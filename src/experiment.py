from src.calibration import Calibration
from src.dataset import Dataset
from src.optimizer import Optimizer
from .parameters import Defaults, Parameters


class Experiment(object):
    def __init__(self, parameters: Parameters = Defaults) -> None:
        self.__dict__.update(parameters.__dict__)
        self.dataset = Dataset(parameters)
        self.optimizer = Optimizer(parameters)
        self.calibration = Calibration(parameters)

    def __str__(self):
        return (
            "Experiment:"
            + self.dataset.__str__
            + "\r\n"
            + self.optimizer.__str__
            + "\r\n"
            + self.calibration.__str__
        )

    def run(self):
        for e in range(1):  # self.n_evals
            self.optimizer.surrogate.fit(self.dataset.data.X, self.dataset.data.y)
            self.dataset.add_X_sample_y(self.optimizer.next_sample(self.dataset))
            self.calibration.analyze(self.optimizer.surrogate, self.dataset)
