from src.parameters import Parameters
from imports.general import *
from imports.ml import *


class News(object):
    """ News Classification Benchmark dataset for bayesian optimization
    """

    def __init__(self, parameters: Parameters):
        self.d = parameters.d
        self.seed = parameters.seed
        self.maximize = parameters.maximization
        self.problem = "News"
        self.real_world = True
        self.n_test = parameters.n_test
        self.n_validation = parameters.n_validation
        self.n_initial = parameters.n_initial
        self.n_pool = parameters.n_pool
        np.random.seed(self.seed)
        self.sample_initial_dataset()

    def sample_initial_dataset(self) -> None:
        self.X_full = np.load("./datasets/News/optim_dataset/hyperparams.npy")
        self.y_full = -np.load("./datasets/News/optim_dataset/accuracies.npy")
        # self.y_test = np.load("./datasets/MNIST/optim_dataset/losses.npy")

        self.X_full = (
            self.X_full[:, np.newaxis] if self.X_full.ndim == 1 else self.X_full
        )
        self.y_full = (
            self.y_full[:, np.newaxis] if self.y_full.ndim == 1 else self.y_full
        )

        test_idxs = np.random.permutation(len(self.X_full))[:self.n_test]
        self.X_test = self.X_full[test_idxs]
        self.y_test = self.y_full[test_idxs]

        self.X_full = np.delete(self.X_full, test_idxs, 0)
        self.y_full = np.delete(self.y_full, test_idxs, 0)

        pool_idxs = np.random.permutation(len(self.X_full))[:self.n_pool]
        self.X_pool = self.X_full[pool_idxs]
        self.y_pool = self.y_full[pool_idxs]

        self.X_full = np.delete(self.X_full, pool_idxs, 0)
        self.y_full = np.delete(self.y_full, pool_idxs, 0)

#        init_idxs = np.random.permutation(len(self.X_pool))[:self.n_initial]
#        self.X_train = self.X_pool[init_idxs]
#        self.y_train = self.y_pool[init_idxs]

#        self.X_pool = np.delete(self.X_pool, init_idxs)
#        self.y_pool = np.delete(self.y_pool, init_idxs)

        self.compute_scaling_properties(self.X_pool, self.y_pool, pool_set=True)
        self.X_pool, self.y_pool =  self.standardize(self.X_pool, self.y_pool)
        self.X_test, self.y_test = self.standardize(self.X_test, self.y_test)
        self.y_min_pool = np.min(self.y_pool)
        self.y_min_test = np.min(self.y_test)
        self.compute_set_properties(self.X_pool, self.y_pool, pool_set=True)
        self.compute_set_properties(self.X_test, self.y_test, pool_set=False)

        self.bounds = []
        for i in range(self.X_pool.shape[1]):
            self.bounds.append((np.min(self.X_pool[:, i]), np.max(self.X_pool[:, i])))
        self.x_lbs = np.array([b[0] for b in self.bounds])
        self.x_ubs = np.array([b[1] for b in self.bounds])

        self.X_train, self.y_train, _ = self.sample_data(n_samples=self.n_initial)

    def compute_set_properties(self, X: np.ndarray, y: np.ndarray, pool_set: bool = False) -> None:
        if pool_set:
            self.X_mean_pool = np.mean(X, axis=0)
            self.X_std_pool = np.std(X, axis=0)
            self.y_mean_pool = np.mean(y)
            self.y_std_pool = np.std(y)
            self.y_min_idx_pool = np.argmin(y)
            self.y_min_loc_pool = X[self.y_min_idx_pool, :]
            self.y_min_pool = y[self.y_min_idx_pool]
        else:
            self.X_mean_test = np.mean(X, axis=0)
            self.X_std_test = np.std(X, axis=0)
            self.y_mean_test = np.mean(y)
            self.y_std_test = np.std(y)
            self.y_min_idx_test = np.argmin(y)
            self.y_min_loc_test = X[self.y_min_idx_test, :]
            self.y_min_test = y[self.y_min_idx_test]

    def compute_scaling_properties(self, X: np.ndarray, y: np.ndarray, pool_set: bool = False) -> None:
        self.X_mean_pool_scaling = np.mean(X, axis=0)
        self.X_std_pool_scaling = np.std(X, axis=0)
        self.y_mean_pool_scaling = np.mean(y)
        self.y_std_pool_scaling = np.std(y)
    
    def standardize(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = (X - self.X_mean_pool_scaling) / self.X_std_pool_scaling
        y = (y - self.y_mean_pool_scaling) / self.y_std_pool_scaling
        return X, y

    def sample_data(
        self, n_samples: int = 1, standardize: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.random.permutation(len(self.X_pool))[:n_samples]
        X = self.X_pool[indices, :]
        y = self.y_pool[indices, :]
        self.X_pool = np.delete(self.X_pool, indices, 0)
        self.y_pool = np.delete(self.y_pool, indices, 0)
        if standardize:
            X, y = self.standardize(X, y)

        return X, y, None

    def __str__(self):
        return str(self.problem)

