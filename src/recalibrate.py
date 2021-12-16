from dataclasses import asdict
import json
from numpy.lib.npyio import save
from imports.general import *
from imports.ml import *
from src.parameters import Parameters
from src.calibration import Calibration
from src.optimizer import Optimizer
from base.dataset import Dataset


class Recalibrator(object):
    """Recalibrator  class """

    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))

    def train_recalibrator_model(self, F_t: np.ndarray, P_hat: np.ndarray) -> Model:
        R = IsotonicRegression(y_min=0, y_max=1)
        R.fit(F_t, P_hat)
        return R

    def run(
        self,
        optimizer: Optimizer,
        dataset: Dataset,
        calibration: Calibration,
        n_splits: int = 5,
        K_fold: bool = False,
    ):
        """Returns auxiliary recalibration model R. Assumes calibration analysis 
        has been run."""
        X, y = dataset.data.X, dataset.data.y
        assert X.ndim == 2 and y.ndim == 2

        cv = KFold(n_splits=n_splits) if K_fold else LeaveOneOut()
        dataset_ = copy.deepcopy(dataset)
        F_t = []
        for train_index, test_index in cv.split(X):
            X_train, X_test = X[[train_index], :], X[[test_index], :]
            y_train, y_test = y[[train_index], :], y[[test_index], :]
            dataset_.data.X = X_train
            dataset_.data.y = y_train
            optimizer.fit_surrogate(dataset_)
            f_t = optimizer.surrogate_object.cdf(X_test, y_test).squeeze()
            F_t.extend(f_t)

        P_hat = np.array([calibration.calculate_p_hat(F_t, f_t) for f_t in F_t])
        F_t = np.array(F_t).squeeze()
        P_hat = np.array(P_hat).squeeze()
        R = self.train_recalibrator_model(F_t, P_hat)
        return R
