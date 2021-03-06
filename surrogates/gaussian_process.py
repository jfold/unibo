from dataclasses import asdict
import botorch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import *
from src.dataset import Dataset
from src.parameters import Parameters
from imports.general import *
from imports.ml import *
from botorch.models.utils import validate_input_scaling
from gpytorch.kernels import *


class GaussianProcess(object):
    """Gaussian process wrapper surrogate class. """

    def __init__(self, parameters: Parameters, dataset: Dataset, name: str = "GP"):
        self.name = name
        self.gp_kernel = parameters.gp_kernel
        self.change_std = parameters.change_std
        self.std_change = parameters.std_change
        self.model = botorch.models.SingleTaskGP(
            torch.tensor(dataset.data.X).double(),
            torch.tensor(dataset.data.y).double(),
            covar_module=getattr(sys.modules[__name__], self.gp_kernel)(),
        )
        self.likelihood = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        botorch.fit.fit_gpytorch_model(self.likelihood)

    def predict(
        self, X_test: np.ndarray, stabilizer: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates mean (prediction) and variance (uncertainty)"""
        X_test = torch.tensor(X_test).double()
        posterior = self.model.posterior(X_test, observation_noise=True)
        mu_predictive = posterior.mean.cpu().detach().numpy().squeeze()
        sigma_predictive = (
            np.sqrt(posterior.variance.cpu().detach().numpy()) + stabilizer
        ).squeeze()
        mu_predictive = (
            mu_predictive[:, np.newaxis] if mu_predictive.ndim == 1 else mu_predictive
        )
        sigma_predictive = (
            sigma_predictive[:, np.newaxis]
            if sigma_predictive.ndim == 1
            else sigma_predictive
        )
        if self.change_std:
            sigma_predictive *= self.std_change
        return mu_predictive, sigma_predictive
