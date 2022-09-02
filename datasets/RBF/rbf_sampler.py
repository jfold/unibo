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
        self.make_data()

    def make_data(self):
        ## Make d-dimensional index set (GP input)
        self.x_ubs = np.ones(self.params.d)
        self.x_lbs = np.zeros(self.params.d)
        self.X_test = np.random.uniform(
            low=self.x_lbs, high=self.x_ubs, size=(self.params.n_test, self.params.d)
        )

        ## Sample GP at index set (GP output)
        self.f_test = self.gp.sample_y(
            self.X_test, self.problem_idx, random_state=self.params.seed
        )[:, [-1]]
        # Find true glob. min
        self.f_min_idx = np.argmin(self.f_test)
        self.f_min_loc = self.X_test[[self.f_min_idx], :]
        self.f_min = self.f_test[[self.f_min_idx]]

        # Compute noise levels w.r.t. SNR
        self.signal_std = np.std(self.f_test)
        self.noise_std = self.signal_std / self.params.snr

        ## Noisify output with gaussian additive
        self.y_test = self.f_test + np.random.normal(
            loc=0, scale=self.noise_std, size=self.f_test.shape
        )
        # Find noisy glob. min
        self.y_min_idx = np.argmin(self.y_test)
        self.y_min_loc = self.X_test[[self.y_min_idx], :]
        self.y_min = self.y_test[[self.y_min_idx]]

        self.X_mean = np.mean(self.X_test, axis=0)
        self.X_std = np.std(self.X_test, axis=0)
        self.f_mean = np.mean(self.f_test)
        self.f_std = np.std(self.f_test)
        self.y_mean = np.mean(self.y_test)
        self.y_std = np.std(self.y_test)

        # Standardize
        self.X_test = (self.X_test - self.X_mean) / self.X_std
        self.f_test = (self.f_test - self.f_mean) / self.f_std
        self.y_test = (self.y_test - self.y_mean) / self.y_std

        ## Sample random initial training points
        idxs = np.random.choice(list(range(self.params.n_test)), self.params.n_initial)
        self.X_train = self.X_test[idxs, :]
        self.f_train = self.f_test[[idxs]]
        self.y_train = self.y_test[[idxs]]

        self.ne_true = -norm.entropy(loc=0, scale=self.noise_std)

    def get_y(self, idxs: list, noisify: bool = True):
        f = np.array([self.f_test[idx] for idx in idxs])
        if noisify:
            noise = np.random.normal(loc=0, scale=np.sqrt(self.noise_std), size=f.shape)
            y = f + noise
            return y
        return f
