from dataclasses import asdict
from botorch.posteriors.posterior import Posterior
from src.dataset import Dataset
from src.parameters import Parameters
from imports.general import *
from imports.ml import *
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from botorch.models.utils import validate_input_scaling


class RandomForest(BatchedMultiOutputGPyTorchModel):
    """Random forest surrogate class. """

    def __init__(
        self, parameters: Parameters, dataset: Dataset, name: str = "RF",
    ):
        self.__dict__.update(asdict(parameters))
        np.random.seed(self.seed)
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
                # "max_samples": [int(self.n_initial / 2), int(self.n_initial)],
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
        # if not self.vanilla:
        #     self.rf_params_grid.update(
        #         {"max_samples": [int(X_train.shape[0] / 2), int(X_train.shape[0]),]}
        #     )
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
        predictions = self.tree_predictions(X)
        sigma_predictive = np.nanstd(predictions, axis=0)
        return sigma_predictive

    def tree_predictions(self, X: np.ndarray) -> np.ndarray:
        predictions = np.full((len(self.model.estimators_), X.shape[0]), np.nan)
        for i_e, estimator in enumerate(self.model.estimators_):
            predictions[i_e, :] = estimator.predict(X)
        return predictions

    def cdf(self, dataset: Dataset):
        raise NotImplementedError()

    def histogram_sharpness(
        self, X: np.ndarray, n_bins: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates sharpness (negative entropy) from histogram
        """
        predictions = self.tree_predictions(X)
        nentropies = []
        for i_n in range(predictions.shape[1]):
            hist, bins = np.histogram(predictions[:, i_n], bins=n_bins, density=True)
            width = bins[1] - bins[0]
            p = hist * width
            nentropies.append(-entropy(p))
        mean_nentropy = np.mean(nentropies)
        return np.array(nentropies), mean_nentropy
