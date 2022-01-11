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

    def run(self):
        if self.bo:
            self.optimizer.fit_surrogate(self.dataset)
            self.calibration.analyze(
                self.optimizer.surrogate_object,
                self.dataset,
                save_settings="---epoch-0",
            )

            for e in tqdm(range(self.n_evals), leave=False):
                save_settings = f"---epoch-{e+1}" if e < self.n_evals - 1 else ""
                x_new, _ = self.optimizer.bo_iter(self.dataset)
                self.dataset.add_X_get_y(x_new)
                self.optimizer.fit_surrogate(self.dataset)
                self.calibration.analyze(
                    self.optimizer.surrogate_object,
                    self.dataset,
                    save_settings=save_settings,
                )
        else:
            X = self.dataset.data.sample_X(self.n_evals)
            self.dataset.add_X_get_y(X)
            self.optimizer.fit_surrogate(self.dataset)
            self.calibration.analyze(self.optimizer.surrogate_object, self.dataset)


if __name__ == "__main__":
    Experiment()
