from dataclasses import asdict
from botorch.posteriors.posterior import Posterior
from numpy.lib.twodim_base import fliplr
from src.dataset import Dataset
from src.parameters import Parameters
from imports.general import *
from imports.ml import *
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from botorch.models.utils import validate_input_scaling


class DummySurrogate(BatchedMultiOutputGPyTorchModel):
    """Dummy surrogate class. Idea: make simple regression model e.g.: 
    mean: (KNN + lin. reg.), for large datasets: for subset or via efficient search methods?
    variance: average distance to training points"""

    def __init__(
        self, parameters: Parameters, dataset: Dataset, name: str = "DS",
    ):
        self.__dict__.update(asdict(parameters))
        self.name = name
        self.n_neighbors = 5
        # torch stuff ...
        self._modules = {}
        self._backward_hooks = {}
        self._forward_hooks = {}
        self._forward_pre_hooks = {}
        self.set_hyperparameter_space()
        self._set_dimensions(train_X=dataset.data.X_train, train_Y=dataset.data.y_train)
        self.fit(X_train=dataset.data.X_train, y_train=dataset.data.y_train)

    def set_hyperparameter_space(self):
        pass

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x, covar_x = self.predict(x)
        mean_x = torch.tensor(mean_x.squeeze())
        covar_x = torch.tensor(np.diag(covar_x.squeeze()))
        return MultivariateNormal(mean_x, covar_x)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.knn = KNNsklearn(n_neighbors=self.n_neighbors, radius=np.inf).fit(X_train)
        self.model = self.knn

    def predict(
        self, X_test: np.ndarray, stabilizer: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_test = (
            X_test.cpu().detach().numpy().squeeze()
            if torch.is_tensor(X_test)
            else X_test.squeeze()
        )
        X_test = X_test[:, np.newaxis] if X_test.ndim == 1 else X_test
        mean_x = []
        var_x = []
        for i in range(X_test.shape[0]):
            x_test = X_test[[i], :]
            neigh_dist, neigh_ind = self.knn.kneighbors(
                x_test, self.n_neighbors, return_distance=True
            )
            neigh_ind = neigh_ind.squeeze()
            neigh_dist = neigh_dist.squeeze()
            lr = LinearRegression().fit(
                self.X_train[neigh_ind, :], self.y_train[neigh_ind, :],
            )
            mean_x.append(lr.predict(x_test).squeeze())
            var_x.append(np.exp(0.01 * np.min(neigh_dist)) + stabilizer)

        return np.array(mean_x)[:, np.newaxis], np.array(var_x)[:, np.newaxis]

    def calculate_y_std(self, X: np.ndarray) -> np.ndarray:
        predictions = None
        sigma_predictive = np.nanstd(predictions, axis=0)
        if self.change_std:
            sigma_predictive *= self.std_change
        return sigma_predictive
