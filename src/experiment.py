from dataclasses import asdict
from imports.general import *
from src.calibration import Calibration
from src.dataset import Dataset
from src.optimizer import Optimizer
from src.recalibrate import Recalibrator
from .parameters import Parameters


class Experiment(object):
    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        self.dataset = Dataset(parameters)
        self.optimizer = Optimizer(parameters)
        self.calibration = Calibration(parameters)
        self.recalibration = Recalibrator(parameters)

    def __str__(self):
        return (
            "Experiment:"
            + self.dataset.__str__
            + "\r\n"
            + self.optimizer.__str__
            + "\r\n"
            + self.calibration.__str__
        )

    def run_recalibration_demo(self):
        X = self.dataset.data.sample_X(self.n_evals)
        self.dataset.add_X_get_y(X)
        self.optimizer.fit_surrogate(self.dataset)
        self.calibration.analyze(self.optimizer.surrogate_object, self.dataset)
        self.recalibration.run(self.optimizer, self.dataset, self.calibration)

    def run_calibration_demo(self):
        X = self.dataset.data.sample_X(self.n_evals)
        self.dataset.add_X_get_y(X)
        self.optimizer.fit_surrogate(self.dataset)
        self.calibration.analyze(self.optimizer.surrogate_object, self.dataset)

    def run(self):
        if self.bo:
            for e in tqdm(range(self.n_evals), leave=False):
                save_settings = f"---epoch-{e+1}" if e < self.n_evals - 1 else ""
                x_new, _ = self.optimizer.bo_iter(self.dataset)
                self.dataset.add_X_get_y(x_new)
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
