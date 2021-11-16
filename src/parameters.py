import json
from imports.general import *
from imports.ml import *
from dataclasses import dataclass, asdict


@dataclass(frozen=False, order=True)
class Defaults:
    seed: bool = 0
    dtype = tf.float64
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
    savepth: str = "/results/"
    experiment: str = datetime.now().strftime("%H:%M:%S-%d%m%y")


class Parameters(Defaults):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.save()

    def save(self):
        if not os.path.isdir(self.savepth + self.experiment):
            os.mkdir(self.savepth + self.experiment)
        json_dump = json.dumps(asdict(self))
        f = open(self.savepth + self.experiment + "/parameters.json", "a")
        f.write(json_dump)
        f.close()
