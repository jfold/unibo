from src.calibration import Calibration
from src.dataset import Dataset
from src.optimizer import Optimizer
from .parameters import Defaults, Parameters


class Experiment(object):
    def __init__(self, parameters: Parameters = Defaults) -> None:
        self.__dict__.update(parameters.__dict__)
        self.data = Dataset(parameters)
        self.optimizer = Optimizer(self.data, parameters)
        self.calibration = Calibration(parameters, self.optimizer.surrogate)

    def __str__(self):
        return (
            "Experiment:"
            + self.data.__str__
            + "\r\n"
            + self.optimizer.__str__
            + "\r\n"
            + self.calibration.__str__
        )

    def run(self):
        for e in range(1):
            print("Ready to run")
