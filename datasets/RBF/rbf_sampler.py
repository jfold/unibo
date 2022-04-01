from src.parameters import Parameters
from imports.general import *
from imports.ml import *


class RBFSampler(object):
    def __init__(self, parameters: Parameters):

        np.random.seed(parameters.seed)
        gp_true = GaussianProcessRegressor(kernel=1.0 * RBF(length_scale=0.1))
        self.X_test = np.linspace(0, 1, parameters.n_test).reshape(-1, 1)
        self.f_test = gp_true.sample_y(
            self.X_test, parameters.problem_idx, random_state=parameters.seed
        )
        # Make dataset
        idxs = np.random.choice(list(range(parameters.n_test)), parameters.n_initial)
        self.X_train = self.X_test[idxs, parameters.problem_idx - 1].reshape(-1, 1)
        self.f_train = self.f_test[idxs, parameters.problem_idx - 1].reshape(-1, 1)
        self.y_test = self.f_test[:, parameters.problem_idx - 1].reshape(
            -1, 1
        ) + np.random.normal(
            loc=0, scale=parameters.sigma_data, size=(self.f_test.size, 1)
        )
        self.y_train = self.f_train + np.random.normal(
            loc=0, scale=parameters.sigma_data, size=(self.f_train.size, 1)
        )
        self.ne_true = -norm.entropy(loc=0, scale=parameters.sigma_data)

