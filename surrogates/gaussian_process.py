from dataclasses import asdict
import botorch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from src.dataset import Dataset
from src.parameters import Parameters
from imports.general import *
from imports.ml import *
from botorch.models.utils import validate_input_scaling

# TODO: validate that GP tunes: length-scale, kernel noise and additive noise


class GaussianProcess(object):
    """Gaussian process wrapper surrogate class. """

    def __init__(self, parameters: Parameters, dataset: Dataset, name: str = "GP"):
        self.name = name
        self.model = botorch.models.SingleTaskGP(
            torch.tensor(dataset.data.X), torch.tensor(dataset.data.y)
        )
        self.likelihood = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        botorch.fit.fit_gpytorch_model(self.likelihood)

    def predict(
        self, X_test: np.ndarray, stabilizer: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates mean (prediction) and variance (uncertainty)"""
        X_test = torch.tensor(X_test)
        posterior = self.model.posterior(X_test, observation_noise=True)
        mu_predictive = np.squeeze(posterior.mean.cpu().detach().numpy(), axis=-1)
        sigma_predictive = np.squeeze(
            (np.sqrt(posterior.variance.cpu().detach().numpy()) + stabilizer), axis=-1
        )
        mu_predictive = (
            mu_predictive[:, np.newaxis] if mu_predictive.ndim == 1 else mu_predictive
        )
        sigma_predictive = (
            sigma_predictive[:, np.newaxis]
            if sigma_predictive.ndim == 1
            else sigma_predictive
        )
        return mu_predictive, sigma_predictive

    def cdf(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        mu_predictive, sigma_predictive = self.predict(X)
        cdf = np.array(
            [
                norm.cdf(
                    y[i].squeeze(), loc=mu_predictive[i], scale=sigma_predictive[i]
                )
                for i in range(X.shape[0])
            ]
        )
        return cdf
