from imports.general import *
from imports.ml import *
from dataclasses import dataclass, asdict


@dataclass(frozen=False, order=True)
class Defaults:
    seed: bool = 0
    dtype = tf.float64
    save_pth: str = "./"
    D: int = 1
    n_test: int = 3000
    n_train: int = 500
    n_initial: int = 500
    n_evals: int = 500
    rf_cv_splits: int = 5
    plot_data: bool = False
    data_location: str = "data.benchmarks.benchmark"
    data_class: str = "Benchmark"
    problem: str = "Alpine01"
    algorithm: str = "vanilla_gp"
    save_pth: str = "/results/"


# class Defaults(object):
#     def __init__(self):
#         self.seed: bool = 0
#         self.dtype = tf.float64
#         self.save_pth: str = "./"
#         self.D: int = 1
#         self.n_test: int = 3000
#         self.n_train: int = 500
#         self.n_initial: int = 500
#         self.n_evals: int = 500
#         self.rf_cv_splits: int = 5
#         self.plot_data: bool = False
#         self.data_location: str = "data.benchmarks.benchmark"
#         self.data_class: str = "Benchmark"
#         self.problem: str = "Alpine01"
#         self.algorithm: str = "vanilla_gp"
#         self.save_pth: str = "/results/"


class Parameters(Defaults):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
