from dataclasses import asdict
from imports.general import *
from src.metrics import Metrics
from src.dataset import Dataset
from src.optimizer import Optimizer
from .parameters import Parameters


class Experiment(object):
    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        self.dataset = Dataset(parameters)
        self.optimizer = Optimizer(parameters)
        self.metrics = Metrics(parameters)

    def __str__(self):
        return (
            "Experiment:"
            + self.dataset.__str__
            + "\r\n"
            + self.optimizer.__str__
            + "\r\n"
            + self.metrics.__str__
        )

    def run(self) -> None:
        if self.bo:
            # Epoch 0
            self.optimizer.fit_surrogate(self.dataset)
            self.metrics.analyze(
                self.optimizer.surrogate_object,
                self.dataset,
                save_settings="---epoch-0",
            )
            self.dataset.save(save_settings="---epoch-0")

            # Epochs > 0
            for e in tqdm(range(self.n_evals), leave=False):
                save_settings = f"---epoch-{e+1}" if e < self.n_evals - 1 else ""

                # BO iteration
                x_next, acq_val, i_choice = self.optimizer.bo_iter(
                    self.dataset, return_idx=True
                )  # , X_test=self.dataset.data.X_test
                y_next = self.dataset.data.y_test[[i_choice]]
                f_next = self.dataset.data.f_test[[i_choice]]

                # Add to dataset
                self.dataset.add_data(x_next, y_next, f_next)

                # Update dataset
                self.dataset.update_solution()

                # Update surrogate
                self.optimizer.fit_surrogate(self.dataset)
                self.metrics.analyze(
                    self.optimizer.surrogate_object,
                    self.dataset,
                    save_settings=save_settings,
                )
            self.dataset.save()
        else:
            X, y, f = self.dataset.data.sample_data(self.n_evals)
            self.dataset.add_data(X, y, f)
            self.optimizer.fit_surrogate(self.dataset)
            self.metrics.analyze(self.optimizer.surrogate_object, self.dataset)
            self.dataset.save()


if __name__ == "__main__":
    Experiment()
