import json
from typing import Dict
from imports.general import *
from imports.ml import *
from dataclasses import dataclass, asdict, replace


@dataclass
class Parameters:
    seed: bool = 0
    d: int = 1  # number of input dimensions
    n_test: int = 3000  # number of test samples for calibration analysis
    n_initial: int = 10  # number of starting points
    n_evals: int = 20  # number of BO iterations
    rf_cv_splits: int = 5  # number of CV splits for random forest hyperparamtuning
    vanilla: bool = False  # simplest implementation (used for test)
    plot_it: bool = False  # whether to plot during BO loop
    save_it: bool = True  # whether to save progress
    csi: float = 0.0  # exploration parameter for BO
    data_location: str = "datasets.verifications.verification"  # "datasets.benchmarks.benchmark"
    data_class: str = "VerificationData"  # dataclass name
    problem: str = "Alpine01"  # "Alpine01" # subproblem name
    maximization: bool = False
    snr: float = 10.0
    K: int = 1  # number of terms in sum for VerificationData
    surrogate: str = "RF"  # surrogate function name
    acquisition: str = "EI"  # acquisition function name
    savepth: str = os.getcwd() + "/results/"
    experiment: str = ""  # folder name
    bo: bool = False  # performing bo to sample X or merely randomly sample X

    def __init__(self, kwargs: Dict = {}, mkdir: bool = False) -> None:
        self.update(kwargs)
        folder_name = (
            datetime.now().strftime("%d%m%y-%H%M%S")
            + f"|{self.surrogate}-{self.acquisition}|{self.data_class}-{self.problem}"
        )
        folder_name = folder_name + "|BO" if self.bo else folder_name
        setattr(
            self, "experiment", folder_name,
        )
        setattr(self, "savepth", self.savepth + self.experiment + "/")
        if mkdir and not os.path.isdir(self.savepth):
            os.mkdir(self.savepth)
            self.save()

    def update(self, kwargs, save=False):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} not found")
        if save:
            self.save()

    def save(self):
        json_dump = json.dumps(asdict(self))
        with open(self.savepth + "parameters.json", "w") as f:
            f.write(json_dump)
