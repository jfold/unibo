from botorch import acquisition
from src.dataset import Dataset
from src.parameters import *
from torch import Tensor
import botorch
from surrogates.gaussian_process import GaussianProcess
from gpytorch.mlls import ExactMarginalLogLikelihood
from surrogates.random_forest import RandomForest
from surrogates.bayesian_neural_network import BayesianNeuralNetwork
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import (
    AnalyticAcquisitionFunction,
    ConstrainedExpectedImprovement,
    ExpectedImprovement,
    NoisyExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)


# TODO: Check what happens if gradients are not available in model.
# TODO: Check what to do with BNNs. Have people done BoTorch with this already?
class Optimizer(object):
    """Optimizer class"""

    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        self.parameters = parameters
        self.is_fitted = False

    def fit_surrogate(self, dataset: Dataset) -> None:
        if self.surrogate == "GP":
            self.surrogate_object = GaussianProcess(self.parameters, dataset)
            self.surrogate_model = self.surrogate_object.model
            self.is_fitted = True
        elif self.surrogate == "RF":
            self.surrogate_object = RandomForest(self.parameters, dataset)
            self.surrogate_model = self.surrogate_object
            self.is_fitted = True
        elif self.surrogate == "BNN":
            self.surrogate_object = BayesianNeuralNetwork(self.parameters, dataset)
            self.surrogate_model = self.surrogate_object
            self.is_fitted = True
        else:
            raise ValueError(f"Surrogate function {self.surrogate} not supported.")

    def construct_acquisition_function(self, dataset: Dataset):
        if not self.is_fitted:
            raise ValueError("Surrogate has not been fitted!")

        if self.acquisition == "EI":
            self.acquisition_function = ExpectedImprovement(
                self.surrogate_model, best_f=dataset.y_opt, maximize=self.maximization
            )
        elif self.acquisition == "UCB":
            self.acquisition_function = UpperConfidenceBound(
                self.surrogate_model, best_f=dataset.y_opt, maximize=self.maximization
            )
        else:
            raise ValueError(f"Acquisition function {self.acquisition} not supported.")

    def bo_iter(self, dataset: Dataset) -> Tensor:
        self.fit_surrogate(dataset)
        self.construct_acquisition_function(dataset)
        X_test, _ = dataset.sample_testset(self.n_test)
        X_test_torch = torch.tensor(np.expand_dims(X_test, 1))
        acquisition_values = self.acquisition_function(X_test_torch)
        i_choice = np.argmax(acquisition_values.detach().numpy())
        return X_test[[i_choice], :], acquisition_values[i_choice]

    def bo_iter_test(self, dataset: Dataset) -> Tensor:
        self.construct_acquisition_function(dataset)
        X_test = np.linspace(0.1, 0.9, 100)
        X_test = np.expand_dims(X_test, 1)
        X_test_torch = torch.tensor(np.expand_dims(X_test, 1))
        acquisition_values = self.acquisition_function(X_test_torch)
        i_choice = np.argmax(acquisition_values.detach().numpy())
        return X_test[[i_choice], :], acquisition_values[i_choice]