from dataclasses import asdict
from botorch.posteriors.posterior import Posterior
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
        self, parameters: Parameters, dataset: Dataset, name: str = "RF",
    ):
        self.__dict__.update(asdict(parameters))
        self.name = name
        # torch stuff ...
        self._modules = {}
        self._backward_hooks = {}
        self._forward_hooks = {}
        self._forward_pre_hooks = {}
        self.set_hyperparameter_space()
        self._set_dimensions(train_X=dataset.data.X, train_Y=dataset.data.y)
        self.fit(X_train=dataset.data.X, y_train=dataset.data.y)

    def set_hyperparameter_space(self):
        pass

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x, covar_x = self.predict(x)
        mean_x = torch.tensor(mean_x.squeeze())
        covar_x = torch.tensor(np.diag(covar_x.squeeze()))
        return MultivariateNormal(mean_x, covar_x)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        pass

    def predict(
        self, X_test: np.ndarray, stabilizer: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def calculate_y_std(self, X: np.ndarray) -> np.ndarray:
        predictions = None
        sigma_predictive = np.nanstd(predictions, axis=0)
        return sigma_predictive
