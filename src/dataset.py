from typing import Dict
from src.parameters import *
from imports.general import *
from datasets.verifications.verification import VerificationData
from datasets.benchmarks.benchmark import Benchmark


class Dataset(object):
    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        module = importlib.import_module(parameters.data_location)
        data_class = getattr(module, parameters.data_class)
        self.data = data_class(parameters)
        self.summary = {"problem": self.problem, "d": self.d}
        self.update_solution()

    def update_solution(self) -> None:
        self.opt_idx = (
            np.argmax(self.data.y) if self.maximization else np.argmin(self.data.y)
        )
        self.y_opt = self.data.y[[self.opt_idx], :]
        self.X_opt = self.data.X[[self.opt_idx], :]
        self.summary.update(
            {
                "n_initial": int(self.n_initial),
                "X": self.data.X.tolist(),
                "y": self.data.y.tolist(),
                "opt_idx": int(self.opt_idx),
                "X_opt": self.X_opt.tolist(),
                "y_opt": self.y_opt.tolist(),
            }
        )

    def save(self, save_settings: str = "") -> None:
        json_dump = json.dumps(self.summary)
        with open(self.savepth + f"dataset{save_settings}.json", "w") as f:
            f.write(json_dump)

    def add_X_get_y(self, x_new: np.array) -> None:
        self.data.X = np.append(self.data.X, x_new, axis=0)
        y_new = self.data.get_y(x_new)
        self.data.y = np.append(self.data.y, y_new, axis=0)
        self.update_solution()

    def sample_testset(self, n_samples: int = None) -> Dict[np.ndarray, np.ndarray]:
        n_samples = self.n_test if n_samples is None else n_samples
        X = self.data.sample_X(n_samples=n_samples)
        y = self.data.get_y(X)
        return X, y
