from dataclasses import asdict
import json
from numpy.lib.npyio import save
from imports.general import *
from imports.ml import *
from src.parameters import Parameters
from src.calibration import Calibration
from base.dataset import Dataset


class Recalibrator(object):
    """Recalibrator  class """

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)
        self.__dict__.update(asdict(parameters))

    def train(self) -> None:
        self.R = IsotonicRegression(y_min=0, y_max=1)
        self.R.fit(self.recalibration_dataset["F_t"], self.recalibration_dataset["y_t"])

    def run(
        self, surrogate_model: object, dataset: Dataset, calibration: Calibration,
    ) -> Model:
        """Returns auxiliary recalibration model R. Assumes calibration analysis 
        has been run."""
        self.calibration_x = surrogate_model.cdf(dataset)
        self.calibration_y = dataset.data.y
        P_hat = calibration.summary["y_calibration"]
        self.recalibration_dataset = {"F_t": self.calibration_x, "P_hat": P_hat}
        self.train()
        return self.R
