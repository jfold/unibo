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
        self.change_std = parameters.change_std
        self.std_change = parameters.std_change
        self.opt_hyp_pars = opt_hyp_pars
        self.fit(parameters, dataset)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x, covar_x = self.predict(x)
        mean_x = torch.tensor(mean_x.squeeze())
        covar_x = torch.tensor(np.diag(covar_x.squeeze()))
        return MultivariateNormal(mean_x, covar_x)

    def fit(self, parameters, dataset):
        torch.manual_seed(parameters.seed)
        # getattr(sys.modules[__name__], self.gp_kernel())
        kernel = ScaleKernel(
            RBFKernel(lengthscale_prior=LogNormalPrior(0, 1)),
            outputscale_prior=NormalPrior(1.0, 2.0),
        )
        self.model = botorch.models.SingleTaskGP(
            torch.tensor(dataset.data.X_train).double(),
            torch.tensor(dataset.data.y_train).double(),
            covar_module=kernel,
        )
        self.likelihood = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        if self.opt_hyp_pars:
            botorch.fit.fit_gpytorch_model(self.likelihood)
        # surrogate_model.model.covar_module.base_kernel.lengthscale.detach().numpy().squeeze()
        # surrogate_model.model.covar_module.outputscale.detach().numpy().squeeze()
        # surrogate_model.model.likelihood.noise.detach().numpy().squeeze()

    def predict(
        self,
        X_test: np.ndarray,
        stabilizer: float = 1e-8,
        observation_noise: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates mean (prediction) and variance (uncertainty)"""
        X_test = torch.tensor(X_test).double()
        posterior = self.model.posterior(X_test, observation_noise=observation_noise)
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


# class GaussianProcess(BatchedMultiOutputGPyTorchModel):
#     """Gaussian process wrapper surrogate class. """

#     def __init__(
#         self,
#         parameters: Parameters,
#         dataset: Dataset,
#         name: str = "GP",
#         opt_hyp_pars: bool = True,
#     ):
#         self.name = name
#         self.gp_kernel = parameters.gp_kernel
#         self.change_std = parameters.change_std
#         self.std_change = parameters.std_change
#         self.opt_hyp_pars = opt_hyp_pars
#         # torch stuff ...
#         self._modules = {}
#         self._backward_hooks = {}
#         self._forward_hooks = {}
#         self._forward_pre_hooks = {}
#         self._set_dimensions(train_X=dataset.data.X_train, train_Y=dataset.data.y_train)
#         self.fit(parameters, dataset)

#     def fit(self, parameters, dataset):
#         np.random.seed(parameters.seed)
#         self.model = GaussianProcessRegressor(
#             kernel=1.0 * RBF(length_scale_bounds=(0.001, 10))
#             + WhiteKernel(noise_level_bounds=(0.0001, 10))
#         )
#         self.model.fit(dataset.data.X_train, dataset.data.y_train)

#     def predict(
#         self, X_test: np.ndarray, stabilizer: float = 1e-8
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """Calculates mean (prediction) and variance (uncertainty)"""
#         X_test = (
#             X_test.cpu().detach().numpy().squeeze()
#             if torch.is_tensor(X_test)
#             else X_test.squeeze()
#         )
#         X_test = X_test[:, np.newaxis] if X_test.ndim == 1 else X_test
#         mu_predictive, sigma_predictive = self.model.predict(X_test, return_std=True)
#         sigma_predictive = sigma_predictive + stabilizer
#         if self.change_std:
#             sigma_predictive *= self.std_change
#         return mu_predictive, sigma_predictive

#     def forward(self, x: Tensor) -> MultivariateNormal:
#         mean_x, covar_x = self.predict(x)
#         mean_x = torch.tensor(mean_x.squeeze())
#         covar_x = torch.tensor(np.diag(covar_x.squeeze()))
#         return MultivariateNormal(mean_x, covar_x)
