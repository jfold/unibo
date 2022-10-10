from src.parameters import Parameters
from imports.general import *
from imports.ml import *


class MNIST(object):
    """ MNIST Benchmark dataset for bayesian optimization
    """

    def __init__(self, parameters: Parameters):
        self.d = parameters.d
        self.seed = parameters.seed
        self.problem = "MNIST"
        self.real_world = True
        self.n_test = parameters.n_test
        self.n_validation = parameters.n_validation
        self.n_initial = parameters.n_initial
        np.random.seed(self.seed)
        self.sample_initial_dataset()

    def sample_initial_dataset(self) -> None:
        self.X_test = np.load("./datasets/MNIST/optim_dataset/hyperparams.npy")
        self.y_test = -np.load("./datasets/MNIST/optim_dataset/accuracies.npy")

        idxs = np.random.permutation(self.X_test.shape[0])
        self.X_test = self.X_test[idxs]
        self.y_test = self.y_test[idxs]

        self.X_test = (
            self.X_test[:, np.newaxis] if self.X_test.ndim == 1 else self.X_test
        )
        self.y_test = (
            self.y_test[:, np.newaxis] if self.y_test.ndim == 1 else self.y_test
        )

        self.compute_set_properties(self.X_test, self.y_test)
        self.X_test, self.y_test = self.standardize(self.X_test, self.y_test)

        self.bounds = []
        for i in range(self.X_test.shape[1]):
            self.bounds.append((np.min(self.X_test[:, i]), np.max(self.X_test[:, i])))
        self.x_lbs = np.array([b[0] for b in self.bounds])
        self.x_ubs = np.array([b[1] for b in self.bounds])

        self.X_train, self.y_train, _ = self.sample_data(n_samples=self.n_initial)

    def compute_set_properties(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        self.y_min_idx = np.argmin(y)
        self.y_min_loc = X[self.y_min_idx, :]
        self.y_min = y[self.y_min_idx]

    def standardize(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std
        return X, y

    def sample_data(self, n_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.random.choice(list(range(len(self.X_test))), n_samples)
        X = self.X_test[indices, :]
        y = self.y_test[indices, :]
        self.X_test = np.delete(self.X_test, indices, 0)
        self.y_test = np.delete(self.y_test, indices, 0)
        X, y = self.standardize(X, y)

        return X, y, None

    def __str__(self):
        return str(self.problem)

