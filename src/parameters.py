import json
from typing import Dict
from imports.general import *
from imports.ml import *
from dataclasses import dataclass, asdict, replace
import random, string


@dataclass
class Parameters:
    seed: bool = 0
    dtype = tf.float64
    d: int = 1
    n_test: int = 3000
    n_train: int = 500
    n_initial: int = 10
    n_evals: int = 500
    rf_cv_splits: int = 5
    plot_it: bool = False
    save_it: bool = True
    csi: float = 0.0
    data_location: str = "data.benchmarks.benchmark"
    data_class: str = "Benchmark"
    problem: str = "Alpine01"
    minmax: str = "minimization"
    snr: float = 10.0
    K: int = 1
    surrogate: str = "RF"
    acquisition: str = "ExpectedImprovement"
    savepth: str = os.getcwd() + "/results/"
    experiment: str = ""

    def __init__(self, kwargs: Dict = {}, mkdir: bool = False) -> None:
        self.update(kwargs)
        setattr(
            self,
            "experiment",
            datetime.now().strftime("%d%m%y-%H%M%S")
            + "|"
            + "".join(
                random.choice(string.ascii_uppercase + string.digits) for _ in range(4)
            ),
        )
        setattr(self, "savepth", self.savepth + self.experiment + "/")
        if mkdir and not os.path.isdir(self.savepth):
            os.mkdir(self.savepth)
            self.save()

    def update(self, kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save(self):
        json_dump = json.dumps(asdict(self))
        f = open(self.savepth + "parameters.json", "w")
        f.write(json_dump)
        f.close()

