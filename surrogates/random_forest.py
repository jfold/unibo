from base.surrogate import Surrogate
from src.parameters import Defaults, Parameters
from imports.general import *
from imports.ml import *
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


class RandomForest(Surrogate):
    """Random forest surrogate class. """

    def __init__(
        self, parameters: Parameters = Defaults(), cv_splits: int = 5, name: str = "RF"
    ):
        self.__dict__.update(parameters.__dict__)
        self.cv_splits = cv_splits
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
        self.name = name

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits random forest model with hyperparameter tuning
        Args:
            X_train (np.ndarray): training input
            y_train (np.ndarray): training output
        """
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
            cv=self.cv_splits,
            n_jobs=-1,
            verbose=0,
        ).fit(X_train, y_train.squeeze())
        self.model = grid_search.best_estimator_

    def predict(self, X_test: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        """Calculates mean (prediction) and variance (uncertainty)
        Args:
            X_test [np.ndarray]: input data
        Returns:
            [tuple] [(np.ndarray,np.ndarray)]: predictive mean and variance
        """
        mu_predictive = self.model.predict(X_test)
        sigma_predictive = self.calculate_y_std(X_test)
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

    def histogram_sharpness(
        self, X: np.ndarray, n_bins: int = 50
    ) -> Union[np.ndarray, np.ndarray]:
        """[Calculates sharpness (negative entropy) from histogram]

        Args:
            X (np.ndarray): [data]

        Returns:
            Union[np.ndarray, np.ndarray]: [description]
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
