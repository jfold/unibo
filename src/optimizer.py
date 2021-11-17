from base.surrogate import Surrogate
from src.dataset import Dataset
from src.parameters import *
import torch
import botorch
from surrogates.random_forest import RandomForest
from sklearn.utils.validation import check_is_fitted


class Optimizer(object):
    """Optimizer wrapper for botorch"""

    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(parameters.__dict__)
        self.surrogate = RandomForest(parameters)

    def acquisition_function(self, mus, sigmas):
        locs = (mus - self.y_min - self.csi) / (sigmas + 1e-9)
        standard_norm_dist = tfp.distributions.Normal(loc=0, scale=1)
        return standard_norm_dist.cdf(locs)

    def acquire_sample(self, dataset: Dataset) -> None:
        check_is_fitted(self.surrogate.model)
        X_candidates = dataset.sample_X(n_samples=self.n_test)
        mu_posterior, sigma_posterior = self.surrogate.predict(X_candidates)
        probs = self.acquisition_function(mu_posterior, sigma_posterior).numpy()
        idx = np.argmin(probs)
        x_new = X_candidates[idx].numpy()
        return x_new
