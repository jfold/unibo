from acquisitions.random_search import RandomSearch
from src.dataset import Dataset
from src.parameters import *
from surrogates.dummy_surrogate import DummySurrogate
from surrogates.gaussian_process import GaussianProcess
from surrogates.random_forest import RandomForest
from surrogates.bayesian_neural_network import BayesianNeuralNetwork
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    UpperConfidenceBound,
    # AnalyticAcquisitionFunction,
    # ConstrainedExpectedImprovement,
    # NoisyExpectedImprovement,
    # PosteriorMean,
    # ProbabilityOfImprovement,
)


class Optimizer(object):
    """# Optimizer class"""

    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        self.parameters = parameters
        self.is_fitted = False

    def fit_surrogate(self, dataset: Dataset) -> None:
        if self.surrogate == "GP":
            self.surrogate_object = GaussianProcess(self.parameters, dataset)
            self.surrogate_model = self.surrogate_object.model
        elif self.surrogate == "RF":
            self.surrogate_object = RandomForest(self.parameters, dataset)
            self.surrogate_model = self.surrogate_object
        elif self.surrogate == "BNN":
            self.surrogate_object = BayesianNeuralNetwork(self.parameters, dataset)
            self.surrogate_model = self.surrogate_object
        elif self.surrogate == "DS":
            self.surrogate_object = DummySurrogate(self.parameters, dataset)
            self.surrogate_model = self.surrogate_object
        elif self.surrogate == "RS":
            self.surrogate_object = None
            self.surrogate_model = None
        else:
            raise ValueError(f"Surrogate function {self.surrogate} not supported.")
        self.is_fitted = True

    def construct_acquisition_function(self, dataset: Dataset) -> None:
        if not self.is_fitted:
            raise ValueError("Surrogate has not been fitted!")
        y_opt_tensor = torch.tensor(dataset.y_opt.squeeze())
        if self.acquisition == "EI":
            self.acquisition_function = ExpectedImprovement(
                self.surrogate_model, best_f=y_opt_tensor, maximize=self.maximization
            )
        elif self.acquisition == "UCB":
            self.acquisition_function = UpperConfidenceBound(
                self.surrogate_model, best_f=y_opt_tensor, maximize=self.maximization
            )
        elif self.acquisition == "RS":
            self.acquisition_function = RandomSearch()
        else:
            raise ValueError(f"Acquisition function {self.acquisition} not supported.")

    def bo_iter(self, dataset: Dataset) -> Dict[np.ndarray, np.ndarray]:
        assert self.is_fitted
        self.construct_acquisition_function(dataset)
        X_test, _ = dataset.sample_testset(self.n_test)
        X_test_torch = torch.tensor(np.expand_dims(X_test, 1))
        acquisition_values = self.acquisition_function(X_test_torch).detach().numpy()
        i_choice = np.argmax(acquisition_values)
        return X_test[[i_choice], :], acquisition_values[i_choice]
