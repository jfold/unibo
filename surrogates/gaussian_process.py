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
from gpytorch.module import Module
from gpytorch.priors import *


class GaussianProcess(object):
    """Gaussian process wrapper surrogate class. """

    def __init__(
        self,
        parameters: Parameters,
        dataset: Dataset,
        name: str = "GP",
        opt_hyp_pars: bool = True,
    ):
        self.name = name
        self.gp_kernel = parameters.gp_kernel
        self.std_change = parameters.std_change
        self.opt_hyp_pars = opt_hyp_pars
        self.kernel = ScaleKernel(
            RBFKernel(lengthscale_prior=LogNormalPrior(0, 1)),
            outputscale_prior=NormalPrior(1.0, 2.0),
        )
        self.fit(parameters, dataset)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x, covar_x = self.predict(x)
        mean_x = torch.tensor(mean_x.squeeze())
        covar_x = torch.tensor(np.diag(covar_x.squeeze()))
        return MultivariateNormal(mean_x, covar_x)

    def recalibrate(self, stabilizer: float = 1e-8):
        assert self.X_train_ is not None and self.y_train_ is not None

        F_t = []
        y_t = self.y_train_.copy()
        for i, x_test in enumerate(self.X_train_):
            X = np.delete(self.X_train_.copy(), i)
            y = np.delete(self.y_train_.copy(), i)
            model = botorch.models.SingleTaskGP(
                torch.tensor(X).double(),
                torch.tensor(y).double(),
                covar_module=self.kernel,
            )
            X_test_torch = torch.from_numpy(x_test)
            posterior = model.posterior(X_test_torch.double(), observation_noise=True)
            F_t.append(
                norm.cdf(
                    x_test,
                    loc=posterior.mean.cpu().detach().numpy().squeeze(),
                    scale=np.sqrt(posterior.variance.cpu().detach().numpy())
                    + stabilizer,
                )
            )

    def fit(self, parameters, dataset):
        self.X_train_ = dataset.data.X_train
        self.y_train_ = dataset.data.y_train

        torch.manual_seed(parameters.seed)

        self.model = botorch.models.SingleTaskGP(
            torch.tensor(dataset.data.X_train).double(),
            torch.tensor(dataset.data.y_train).double(),
            covar_module=self.kernel,
        )

        self.likelihood = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        if self.opt_hyp_pars:
            botorch.fit.fit_gpytorch_model(self.likelihood)

        # surrogate_model.model.covar_module.base_kernel.lengthscale.detach().numpy().squeeze()
        # surrogate_model.model.covar_module.outputscale.detach().numpy().squeeze()
        # surrogate_model.model.likelihood.noise.detach().numpy().squeeze()

    def predict(
        self, X_test: Any, stabilizer: float = 1e-8, observation_noise: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates mean (prediction) and variance (uncertainty)"""
        X_test_torch = (
            torch.from_numpy(X_test) if isinstance(X_test, np.ndarray) else X_test
        )
        posterior = self.model.posterior(
            X_test_torch.double(), observation_noise=observation_noise
        )
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
        if self.std_change != 1.0:
            sigma_predictive *= self.std_change
        return mu_predictive, sigma_predictive

