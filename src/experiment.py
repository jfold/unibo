from src.calibration import Calibration
from src.dataset import Dataset
from src.optimizer import Optimizer
from .parameters import Defaults, Parameters


class Experiment(object):
    def __init__(self, parameters: Parameters = Defaults) -> None:
        self.__dict__.update(parameters.__dict__)
        self.data = Dataset(parameters)
        self.optimizer = Optimizer(parameters)
        self.calibration = Calibration(parameters)

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
        self.calibration.surrogate.fit(self.data.X_train, self.data.y_train)
        self.calibration.surrogate.predict(self.data.X_test)
