from base.surrogate import Surrogate
from src.dataset import Dataset
from src.parameters import *
import torch
import botorch
from surrogates.random_forest import RandomForest
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


class Optimizer(object):
    """TODO: Optimizer wrapper for botorch"""

    #def __init__(self, parameters: Parameters) -> None:
    #    self.__dict__.update(asdict(parameters))
    #    self.surrogate = RandomForest(parameters)

    def __init__(self, parameters: Parameters) -> None:
            self.__dict__.update(asdict(parameters))

    #"""TODO: Implement sampling function for getting some random starting points
    #based on the problem we are doing - solved by dataset sampling function?"""
    #def sample_start_x(self, n):
    #    x_start = [0]

    #def expected_improvement(self, y_min, mus, sigmas):
    #    locs = (mus - y_min - self.csi) / (sigmas + 1e-9)
    #    standard_norm_dist = tfp.distributions.Normal(loc=0, scale=1)
    #    return standard_norm_dist.cdf(locs)

    #def next_sample(self, dataset: Dataset) -> np.array:
    #    X_candidates = dataset.data.sample_X(n_samples=self.n_test)
    #    mu_posterior, sigma_posterior = self.surrogate.predict(X_candidates)
    #    ei = self.expected_improvement(
    #        np.min(dataset.data.y), mu_posterior, sigma_posterior
    #    ).numpy()
    #    idx = np.argmin(ei)
    #    x_new = X_candidates[[idx], :]
    #    return x_new

    #Parameters requires something specifying number of BO iterations?
    #Parameters requires something specifying number of starting points?

    def init_fit_surrogate(self, dataset: Dataset):
        #TODO: Consider computational cost of if statements and object
        #creation here.
        if self.surrogate == "GP":
            self.surrogate_model = botorch.models.SingleTaskGP(dataset.data.X, dataset.data.Y)
            self.likelihood = ExactMarginalLogLikelihood(gp.likelihood, gp)
            botorch.fit.fit_gpytorch_model(mll)
        elif self.surrogate == "RF":
            self.surrogate_model = RandomForest(parameters)
            self.surrogate_model.fit(dataset.data.X, dataset.data.Y)


    def bo_iter(self, dataset: Dataset):
        #For every iteration of BO we must create a new model and fit it to
        #the data - if statement to select between surrogates

        #TODO: Optimize by not calling np.min, but rather keeping track of lowest f
        #so far - perhaps as memberdata in dataset?
        init_fit_surrogate(dataset)

        if self.acquisition == "EI":
            self.acquisiton_function = botorch.acquisiton.analytic.ExpectedImprovement(self.surrogate_model, best_f = np.min(dataset.data.Y))
        elif self.acquisition == "UCB":
            self.acquisition_function = botorch.acquisition.analytic.UpperConfidenceBoundExpectedImprovement(self.surrogate_model, best_f = np.min(dataset.data.Y))

        #TODO: Dataset needs a bounds parameter specifying which values x can take.
        #TODO: Check what happens if gradients are not available in model.
        #TODO: Check what to do with BNNs. Have people done BoTorch with this already?
        candidate, acq_value = optimize_acqf(self.acquisition_function, bounds=dataset.bounds, q=1, num_restarts=5, raw_samples=20)
        #Typecheck candidate here
        return candidate
