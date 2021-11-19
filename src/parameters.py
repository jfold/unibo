import json
from imports.general import *
from imports.ml import *
from dataclasses import dataclass, asdict
import random, string


@dataclass(frozen=False, order=True)
class Defaults:
    seed: bool = 0
    dtype = tf.float64
    d: int = 1
    n_test: int = 3000
    n_train: int = 500
    n_initial: int = 500
    n_evals: int = 500
    rf_cv_splits: int = 5
    plot_data: bool = False
    csi: float = 0.0
    data_location: str = "data.benchmarks.benchmark"
    data_class: str = "Benchmark"
    problem: str = "Alpine01"
    minmax: str = "minimization"
    snr: float = 10.0
    K: int = 1
    surrogate: str = "RandomForest"
    acquisition: str = "ExpectedImprovement"
    savepth: str = os.getcwd() + "/results/"
    experiment: str = datetime.now().strftime("%H:%M:%S-%d%m%y") + "|" + "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(4)
    )


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
