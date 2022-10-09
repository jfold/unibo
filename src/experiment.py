from dataclasses import asdict
from imports.general import *
from imports.ml import *
from src.metrics import Metrics
from src.dataset import Dataset
from src.optimizer import Optimizer
from .parameters import Parameters
from src.recalibrator import Recalibrator


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
        # Epoch 0
        self.optimizer.fit_surrogate(self.dataset)
        recalibrator = (
            Recalibrator(
                self.dataset, self.optimizer.surrogate_object, mode=self.recal_mode,
            )
            if self.recalibrate
            else None
        )
        self.metrics.analyze(
            self.optimizer.surrogate_object,
            self.dataset,
            recalibrator=recalibrator,
            extensive=True,
        )

        if self.bo:

            # Epochs > 0
            for e in tqdm(range(self.n_evals), leave=False):

                recalibrator = (
                    Recalibrator(
                        self.dataset,
                        self.optimizer.surrogate_object,
                        mode=self.recal_mode,
                    )
                    if self.recalibrate
                    else None
                )
                # BO iteration
                x_next, acq_val, i_choice = self.optimizer.bo_iter(
                    self.dataset,
                    X_test=self.dataset.data.X_test,
                    recalibrator=recalibrator,
                    return_idx=True,
                )
                y_next = self.dataset.data.y_test[[i_choice]]
                f_next = (
                    self.dataset.data.f_test[[i_choice]]
                    if not self.dataset.data.real_world
                    else None
                )

                # add to dataset
                self.dataset.add_data(x_next, y_next, f_next, i_choice=i_choice)

                # Update dataset
                self.dataset.update_solution()

                # Update surrogate
                self.optimizer.fit_surrogate(self.dataset)

                if self.analyze_all_epochs:
                    self.metrics.analyze(
                        self.optimizer.surrogate_object,
                        self.dataset,
                        recalibrator=recalibrator,
                        extensive=self.extensive_metrics,
                    )

            if not self.analyze_all_epochs:
                self.metrics.analyze(
                    self.optimizer.surrogate_object,
                    self.dataset,
                    recalibrator=recalibrator,
                    extensive=True,
                )
        else:
            if self.analyze_all_epochs:
                for e in tqdm(range(self.n_evals), leave=False):
                    X, y, f = self.dataset.data.sample_data(n_samples=1)
                    self.dataset.add_data(X, y, f)
                    self.optimizer.fit_surrogate(self.dataset)
                    recalibrator = (
                        Recalibrator(
                            self.dataset,
                            self.optimizer.surrogate_object,
                            mode=self.recal_mode,
                        )
                        if self.recalibrate
                        else None
                    )
                    self.metrics.analyze(
                        self.optimizer.surrogate_object,
                        self.dataset,
                        recalibrator=recalibrator,
                        extensive=self.extensive_metrics,
                    )
            else:
                X, y, f = self.dataset.data.sample_data(self.n_evals)
                self.dataset.add_data(X, y, f)
                self.optimizer.fit_surrogate(self.dataset)
                recalibrator = (
                    Recalibrator(
                        self.dataset,
                        self.optimizer.surrogate_object,
                        mode=self.recal_mode,
                    )
                    if self.recalibrate
                    else None
                )
            self.metrics.analyze(
                self.optimizer.surrogate_object,
                self.dataset,
                recalibrator=recalibrator,
                extensive=True,
            )

        self.dataset.save()
        self.metrics.save()


if __name__ == "__main__":
    Experiment()
