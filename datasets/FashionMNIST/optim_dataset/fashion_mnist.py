from attr import has
from src.parameters import Parameters
from .evalset import test_funcs
import inspect
from imports.general import *
from imports.ml import *


class FashionMNIST(object):
    """ MNIST Benchmark dataset for bayesian optimization
    """

    def __init__(self, parameters: Parameters):
        self.d = parameters.d
        self.seed = parameters.seed
        #self.noisify = parameters.noisify
        #self.snr = parameters.snr
        self.problem="MNIST"
        self.n_test = parameters.n_test
        self.n_validation = parameters.n_validation
        self.n_initial = parameters.n_initial
        np.random.seed(self.seed)
        with open("./optim_dataset/hyperparams.npy", "rb") as f:
            self.X_test = np.load(f)
        with open("./optim_dataset/accuracies.npy", "rb") as f:
            self.y_test = np.load(f)
       # with open("datasets/MNIST/optim_dataset/losses.npy", "rb") as f:
        #self.benchmarks = test_funcs
        #all_problems = inspect.getmembers(self.benchmarks)
        #if parameters.problem not in [a for a, b in all_problems]:
        #    raise NameError(f"Could not find problem: {parameters.problem}")
        #self.benchmark_tags = {}
        #for name, obj in inspect.getmembers(self.benchmarks):
        #    if inspect.isclass(obj):
        #        try:
        #            self.benchmark_tags.update({name: obj(dim=self.d).classifiers})
        #        except:
        #            pass

        #if parameters.problem not in self.benchmark_tags:
        #    raise ValueError(
        #        f"Problem {parameters.problem} does not support dimensionality {self.d}"
        #    )

        #self.problem = getattr(self.benchmarks, parameters.problem)(dim=self.d)
        self.bounds = []
        for i in range(len(self.X_candid[0])):
            bounds.append((np.min(self.X_test[:,i]),np.max(self.X_test[:,i])))
        self.x_lbs = np.array([b[0] for b in self.bounds])
        self.x_ubs = np.array([b[1] for b in self.bounds])

        self.sample_initial_dataset()

    def sample_initial_dataset(self) -> None:
        self.X_train, self.y_train = self.sample_data(
            n_samples=self.n_initial, first_time=True
        )

    def compute_set_properties(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        #self.signal_std = np.std(f)
        #self.noise_std = np.sqrt(self.signal_std ** 2 / self.snr)
        #self.ne_true = -norm.entropy(loc=0, scale=self.noise_std)
        #self.y_mean = self.f_mean
        #self.y_std = np.sqrt(self.f_std ** 2 + self.noise_std ** 2)

    def standardize(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = (X - self.X_mean) / self.X_std  # (f - self.X_mean) / np.max(np.abs(X))  #
        #f = (f - self.f_mean) / self.f_std  # (f - self.f_mean) / np.max(np.abs(f))  #
        y = (y - self.y_mean) / self.y_std  # (f - self.f_mean) / np.max(np.abs(f))  #
        return X, y

    def sample_data(
        self, n_samples: int = 1, first_time: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = []
        y = []
        for i in range(n_samples):
            index = np.random.choice(list(np.range(0, len(self.X_test))), 1)
            X.append(self.X_test[index[0]])
            y.append(self.y_test[index[0]])
            self.X_test = np.delete(self.X_test, index[0], 0)
            self.Y_test = np.delete(self.Y_test, index[0], 0)

        if first_time:
            self.compute_set_properties(X, y)

        X, y = self.standardize(X, y)

        if first_time:
            self.y_min_idx = np.argmin(y)
            self.y_min_loc = X[self.y_min_idx, :]
            self.y_min = y[self.y_min_idx]
            self.y_max = np.max(y)

        return X, y[:, np.newaxis]

    def __str__(self):
        return str(self.problem)

