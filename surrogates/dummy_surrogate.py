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
        if self.vanilla:
            self.rf_params_grid = {
                "n_estimators": [30],
                "max_depth": [10],
            }
        else:
            self.rf_params_grid = {
                "n_estimators": [10, 100, 1000],
                "max_depth": [5, 10, 20],
                "max_samples": [
                    int(self.n_initial / 4),
                    int(self.n_initial / 2),
                    int((3 / 4) * self.n_initial),
                ],
                "max_features": ["auto", "sqrt"],
            }

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x, covar_x = self.predict(x)
        mean_x = torch.tensor(mean_x.squeeze())
        covar_x = torch.tensor(np.diag(covar_x.squeeze()))
        return MultivariateNormal(mean_x, covar_x)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits random forest model with hyperparameter tuning
        Args:
            X_train (np.ndarray): training input
            y_train (np.ndarray): training output
        """
        np.random.seed(2021)
        if not self.vanilla:
            self.rf_params_grid.update(
                {
                    "max_samples": [
                        int(X_train.shape[0] / 4),
                        int(X_train.shape[0] / 2),
                        int((3 / 4) * X_train.shape[0]),
                    ],
                }
            )
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid=self.rf_params_grid,
            cv=self.rf_cv_splits,
            n_jobs=-1,
            verbose=0,
        ).fit(X_train, y_train.squeeze())
        self.model = grid_search.best_estimator_

    def predict(
        self, X_test: np.ndarray, stabilizer: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates mean (prediction) and variance (uncertainty)"""
        X_test = (
            X_test.cpu().detach().numpy().squeeze()
            if torch.is_tensor(X_test)
            else X_test.squeeze()
        )
        X_test = X_test[:, np.newaxis] if X_test.ndim == 1 else X_test
        mu_predictive = self.model.predict(X_test)
        sigma_predictive = self.calculate_y_std(X_test) + stabilizer
        return (mu_predictive[:, np.newaxis], sigma_predictive[:, np.newaxis])

    def calculate_y_std(self, X: np.ndarray) -> np.ndarray:
        predictions = None
        sigma_predictive = np.nanstd(predictions, axis=0)
        return sigma_predictive
