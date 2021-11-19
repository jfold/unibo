from imports.general import *
from src.calibration import Calibration
from src.dataset import Dataset
from src.optimizer import Optimizer
from .parameters import Defaults, Parameters


class Experiment(object):
    def __init__(self, parameters: Parameters = Defaults()) -> None:
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
        for e in tqdm(range(self.n_evals), leave=False):
            self.optimizer.surrogate.fit(self.dataset.data.X, self.dataset.data.y)
            x_new = self.optimizer.next_sample(self.dataset)
            self.dataset.add_X_sample_y(x_new)
            self.calibration.analyze(
                self.optimizer.surrogate, self.dataset, save_settings=f"epoch-{e+1}"
            )


if __name__ == "__main__":
    Experiment()
