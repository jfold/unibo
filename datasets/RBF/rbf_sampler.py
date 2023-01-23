from src.parameters import Parameters
from imports.general import *
from imports.ml import *


class RBFSampler(object):
    """RBFSampler class makes data sampled from a gaussian process and contains f (without noise) and y (with noise).
    """

    def __init__(self, parameters: Parameters):
        self.params = parameters
        np.random.seed(self.params.seed)
        self.kernel = 1.0 * RBF(length_scale=0.1)
        self.gp = GaussianProcessRegressor(kernel=self.kernel)
        self.problem_idx = parameters.problem_idx + 1
        self.x_ubs = np.ones(self.params.d)
        self.x_lbs = -np.ones(self.params.d)
        self.f_min_idx_test = None
        self.y_min_loc_test= None
        self.f_min_idx_valid = None
        self.y_min_loc_valid = None
        self.ne_true = None
        self.real_world = False
        self.sample_initial_dataset()

    def sample_initial_dataset(self, n_samples: int = 3000) -> None:
        self.X_pool, self.y_pool, self.f_pool = self.sample_data(
            n_samples=self.params.n_pool, first_time=True, test_set=False
        )
        self.X_test, self.y_test, self.f_test = self.sample_data(
            n_samples=self.params.n_test, first_time = True, test_set=True
        )
        idxs = np.random.permutation(len(self.X_pool))[:self.params.n_initial]
        self.X_train = self.X_pool[tuple(idxs), :]
        self.f_train = self.f_pool[tuple([idxs])]
        self.y_train = self.y_pool[tuple([idxs])]

        self.X_pool = np.delete(self.X_pool, idxs)
        self.y_pool = np.delete(self.y_pool, idxs)
        self.f_pool = np.delete(self.f_pool, idxs)

        self.X_valid, self.y_valid, self.f_valid = self.sample_data(
            n_samples=self.params.n_validation
        )

    def compute_set_properties(self, X: np.ndarray, f: np.ndarray) -> None:
        self.f_min_idx = int(np.argmin(f))
        self.f_min_loc = X[[self.f_min_idx], :]
        self.f_min = f[[self.f_min_idx]]

        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.f_mean = np.mean(f)
        self.f_std = np.std(f)
        self.signal_std = np.std(f)
        self.noise_std = np.sqrt(self.signal_std ** 2 / self.params.snr)
        self.ne_true = -norm.entropy(loc=0, scale=self.noise_std)
        self.y_mean = self.f_mean
        self.y_std = np.sqrt(self.f_std ** 2 + self.noise_std ** 2)

        self.f_min = (self.f_min - self.f_mean) / self.f_std

    def standardize(
        self, X: np.ndarray, y: np.ndarray, f: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = (X - self.X_mean) / self.X_std  # (f - self.X_mean) / np.max(np.abs(X))  #
        f = (f - self.f_mean) / self.f_std  # (f - self.f_mean) / np.max(np.abs(f))  #
        y = (y - self.y_mean) / self.y_std  # (f - self.f_mean) / np.max(np.abs(f))  #
        return X, y, f

    def sample_data(self, n_samples: int = 1, first_time: bool = False, test_set: bool = False):
        ## Make d-dimensional index set (GP input)
        X = np.random.uniform(
            low=self.x_lbs, high=self.x_ubs, size=(n_samples, self.params.d)
        )
        ## Sample GP at index set (GP output)
        f = self.gp.sample_y(X, self.problem_idx, random_state=self.params.seed)[
            :, [-1]
        ]

        if first_time and not test_set:
            self.compute_set_properties(X, f)

        noise = np.random.normal(loc=0, scale=self.noise_std, size=f.shape)
        y = f + noise

        if first_time and test_set:
            self.y_min_idx_test = int(np.argmin(y))
            self.y_min_loc_test = X[[self.y_min_idx_test], :]
            self.y_min_test = np.min(y)
            self.y_max_test = np.max(y)
        elif first_time and not test_set:
            self.y_min_idx_valid = int(np.argmin(y))
            self.y_min_loc_valid = X[[self.y_min_idx_valid], :]
            self.y_min_valid = np.min(y)
            self.y_max_valid = np.max(y)      

        X, y, f = self.standardize(X, y, f)

        return X, y, f

    # def get_y(self, idxs: list, noisify: bool = True):
    #     f = np.array([self.f_test[idx] for idx in idxs])
    #     if noisify:
    #         noise = np.random.normal(loc=0, scale=np.sqrt(self.noise_std), size=f.shape)
    #         y = f + noise
    #         return y
    #     return f
