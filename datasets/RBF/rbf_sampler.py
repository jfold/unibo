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
        self.x_lbs = np.zeros(self.params.d)
        self.X_mean = None
        self.f_mean = None
        self.y_mean = None
        self.f_min_idx = None
        self.y_min_loc = None
        self.ne_true = None
        self.sample_testset_and_compute_data_stats()
        self.X_train, self.y_train, self.f_train = self.sample_data(
            parameters.n_initial
        )

    def sample_testset_and_compute_data_stats(self, n_samples: int = 3000) -> None:
        self.sample_data(n_samples=n_samples, first_time=True)
        self.X_test, self.y_test, self.f_test = self.sample_data(
            n_samples=self.params.n_test
        )

    def sample_data(self, n_samples: int = 1, first_time: bool = False):
        ## Make d-dimensional index set (GP input)
        X = np.random.uniform(
            low=self.x_lbs, high=self.x_ubs, size=(n_samples, self.params.d)
        )
        ## Sample GP at index set (GP output)
        f = self.gp.sample_y(X, self.problem_idx, random_state=self.params.seed)[
            :, [-1]
        ]

        if first_time:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.f_mean = np.mean(f)
            self.f_std = np.std(f)
            # Compute noise levels w.r.t. SNR
            self.signal_std = self.f_std
            self.noise_std = np.sqrt(self.signal_std ** 2 / self.params.snr)

        ## Noisify output with gaussian additive
        noise = np.random.normal(loc=0, scale=self.noise_std, size=f.shape)
        y = f + noise

        if first_time:
            self.ne_true = -norm.entropy(loc=0, scale=self.noise_std)
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)

        # Find true glob. min
        self.f_min_idx = np.argmin(f)
        self.f_min_loc = X[[self.f_min_idx], :]
        self.f_min = f[[self.f_min_idx]]

        # Find noisy glob. min
        self.y_min_idx = np.argmin(y)
        self.y_min_loc = X[[self.y_min_idx], :]
        self.y_min = y[[self.y_min_idx]]

        # Standardize
        if (
            self.X_mean is not None
            and self.f_mean is not None
            and self.y_mean is not None
        ):
            X = (X - self.X_mean) / self.X_std
            f = (f - self.f_mean) / self.f_std
            y = (y - self.y_mean) / self.y_std

        # ## Sample random initial training points
        # idxs = np.random.choice(list(range(self.params.n_test)), self.params.n_initial)
        # self.X_train = self.X_test[idxs, :]
        # self.f_train = self.f_test[[idxs]]
        # self.y_train = self.y_test[[idxs]]

        return X, y, f

    # def get_y(self, idxs: list, noisify: bool = True):
    #     f = np.array([self.f_test[idx] for idx in idxs])
    #     if noisify:
    #         noise = np.random.normal(loc=0, scale=np.sqrt(self.noise_std), size=f.shape)
    #         y = f + noise
    #         return y
    #     return f
