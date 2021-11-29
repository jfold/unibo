from dataclasses import asdict
from imports.general import *
from src.calibration import Calibration
from src.dataset import Dataset
from src.optimizer import Optimizer
from .parameters import Parameters


class Experiment(object):
    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
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

    def run_bo(self):
        for e in tqdm(range(self.n_evals), leave=False):
            self.optimizer.surrogate.fit(self.dataset.data.X, self.dataset.data.y)
            x_new = self.optimizer.next_sample(self.dataset)
            self.dataset.add_X_sample_y(x_new)
            self.calibration.analyze(
                self.optimizer.surrogate, self.dataset, save_settings=f"---epoch-{e+1}"
            )

    def run_calibraion_demo(self):
        """Samples all datapoints uniformly"""
        self.dataset.add_X_sample_y(
            self.dataset.data.sample_X(self.n_train - self.n_initial)
        )
        self.optimizer.surrogate.fit(self.dataset.data.X, self.dataset.data.y)
        self.calibration.analyze(self.optimizer.surrogate, self.dataset)


if __name__ == "__main__":
    Experiment()
