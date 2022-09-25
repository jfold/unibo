from argparse import ArgumentError
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
from src.metrics import Metrics


class GaussianProcessSklearn(BatchedMultiOutputGPyTorchModel):
    """Gaussian process wrapper surrogate class. """

    def __init__(
        self,
        parameters: Parameters,
        dataset: Dataset,
        name: str = "GP",
        opt_hyp_pars: bool = True,
    ):
        # torch stuff ...
        self._modules = {}
        self._backward_hooks = {}
        self._forward_hooks = {}
        self._forward_pre_hooks = {}
        self._set_dimensions(train_X=dataset.data.X_train, train_Y=dataset.data.y_train)
        self.name = name
        self.gp_kernel = parameters.gp_kernel
        self.d = parameters.d
        self.std_change = parameters.std_change
        self.recalibrate_with_cv = parameters.recalibrate_with_cv
        self.recalibrate_with_testset = parameters.recalibrate_with_testset
        if self.recalibrate_with_cv or self.recalibrate_with_testset:
            self.metrics = Metrics(parameters)
        self.opt_hyp_pars = opt_hyp_pars
        self.kernel = 1.0 * RBF() + WhiteKernel()
        self.fit(parameters, dataset)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x, covar_x = self.predict(x)
        mean_x = torch.tensor(mean_x.squeeze())
        covar_x = torch.tensor(np.diag(covar_x.squeeze()))
        return MultivariateNormal(mean_x, covar_x)

    def recalibrate_kuleshov(self, dataset: Dataset, stabilizer: float = 1e-8):
        assert dataset.data.X_train is not None and dataset.data.y_train is not None

        F_t = []
        y_t = dataset.data.y_train_.copy()
        for i, x_test in enumerate(dataset.data.X_train):
            X = np.delete(dataset.data.X_train.copy(), i)
            y = np.delete(dataset.data.y_train_copy(), i)
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

    def recalibrate_search(self, dataset: Dataset, stabilizer: float = 1e-8):
        recal_factors = [
            0.01,
            0.10,
            0.50,
            0.75,
            0.80,
            0.90,
            0.95,
            1.00,
            1.05,
            1.10,
            1.25,
            1.50,
            2.50,
            5.0,
            10.0,
        ]
        mses = []
        if self.recalibrate_with_testset:
            mus, sigmas = self.predict(dataset.data.X_test)
            for recal_factor in recal_factors:
                sigmas_ = sigmas * recal_factor
                mses.append(
                    self.metrics.calibration_y_batched(
                        mus, sigmas_, dataset.data.y_test, return_mse=True
                    )
                )
        elif self.recalibrate_with_cv:
            mses_ = np.full((dataset.data.X_train.shape[0], len(recal_factors)), np.nan)
            for i, x_test in enumerate(dataset.data.X_train):
                X = (
                    np.delete(dataset.data.X_train.copy(), i)[:, np.newaxis]
                    if self.d == 1
                    else np.delete(dataset.data.X_train.copy(), i)
                )
                y = np.delete(self.y_train_.copy(), i)[:, np.newaxis]
                # Train model
                model = botorch.models.SingleTaskGP(
                    torch.from_numpy(X).double(),
                    torch.from_numpy(y).double(),
                    covar_module=self.kernel,
                )
                likelihood = ExactMarginalLogLikelihood(model.likelihood, model)
                if self.opt_hyp_pars:
                    botorch.fit.fit_gpytorch_model(likelihood)

                X_test_torch = torch.from_numpy(x_test)
                posterior = model.posterior(
                    X_test_torch.double(), observation_noise=True
                )
                mus = posterior.mean.cpu().detach().numpy().squeeze()
                sigmas = np.sqrt(posterior.variance.cpu().detach().numpy())

                for i_r, recal_factor in enumerate(recal_factors):
                    sigmas_ = sigmas * recal_factor
                    mses_[i, i_r] = self.metrics.calibration_y_batched(
                        mus, sigmas_, self.y_train_[i], return_mse=True
                    )

            mses = np.mean(mses_, axis=0)
        else:
            raise ValueError()

        self.recal_factor = recal_factors[np.argmin(mses)]
        print(self.recal_factor)

    def fit(self, parameters, dataset):

        np.random.seed(parameters.seed)
        self.model = GaussianProcessRegressor(kernel=self.kernel)
        self.model.fit(dataset.data.X_train, dataset.data.y_train)
        if self.recalibrate_with_cv or self.recalibrate_with_testset:
            self.recalibrate_search(dataset)

    def predict(
        self, X_test: Any, stabilizer: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates mean (prediction) and variance (uncertainty)"""
        X_test = (
            X_test.cpu().detach().numpy().squeeze()
            if torch.is_tensor(X_test)
            else X_test.squeeze()
        )
        X_test = X_test[:, np.newaxis] if X_test.ndim == 1 else X_test
        mu_predictive, sigma_predictive = self.model.predict(X_test, return_std=True)
        if self.std_change != 1.0:
            sigma_predictive *= self.std_change
        return mu_predictive, sigma_predictive + stabilizer


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
        self.d = parameters.d
        self.std_change = parameters.std_change
        self.recalibrate_with_cv = parameters.recalibrate_with_cv
        self.recalibrate_with_testset = parameters.recalibrate_with_testset
        if self.recalibrate_with_cv or self.recalibrate_with_testset:
            self.metrics = Metrics(parameters)
        self.opt_hyp_pars = opt_hyp_pars
        self.kernel = ScaleKernel(
            RBFKernel(lengthscale_prior=LogNormalPrior(0, 1)),
            outputscale_prior=NormalPrior(1.0, 2.0),
        )
        self.fit(parameters, dataset)

    def recalibrate_kuleshov(self, stabilizer: float = 1e-8):
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

    def recalibrate_search(self, dataset: Dataset, stabilizer: float = 1e-8):
        recal_factors = [
            0.01,
            0.10,
            0.50,
            0.75,
            0.80,
            0.90,
            0.95,
            1.00,
            1.05,
            1.10,
            1.25,
            1.50,
            2.50,
            5.0,
            10.0,
        ]
        mses = []
        if self.recalibrate_with_testset:
            mus, sigmas = self.predict(dataset.data.X_test)
            for recal_factor in recal_factors:
                sigmas_ = sigmas * recal_factor
                mses.append(
                    self.metrics.calibration_y_batched(
                        mus, sigmas_, dataset.data.y_test, return_mse=True
                    )
                )
        elif self.recalibrate_with_cv:
            mses_ = np.full((self.X_train_.shape[0], len(recal_factors)), np.nan)
            for i, x_test in enumerate(self.X_train_):
                X = (
                    np.delete(self.X_train_.copy(), i)[:, np.newaxis]
                    if self.d == 1
                    else np.delete(self.X_train_.copy(), i)
                )
                y = np.delete(self.y_train_.copy(), i)[:, np.newaxis]
                # Train model
                model = botorch.models.SingleTaskGP(
                    torch.from_numpy(X).double(),
                    torch.from_numpy(y).double(),
                    covar_module=self.kernel,
                )
                likelihood = ExactMarginalLogLikelihood(model.likelihood, model)
                if self.opt_hyp_pars:
                    botorch.fit.fit_gpytorch_model(likelihood)

                X_test_torch = torch.from_numpy(x_test)
                posterior = model.posterior(
                    X_test_torch.double(), observation_noise=True
                )
                mus = posterior.mean.cpu().detach().numpy().squeeze()
                sigmas = np.sqrt(posterior.variance.cpu().detach().numpy())

                for i_r, recal_factor in enumerate(recal_factors):
                    sigmas_ = sigmas * recal_factor
                    mses_[i, i_r] = self.metrics.calibration_y_batched(
                        mus, sigmas_, self.y_train_[i], return_mse=True
                    )

            mses = np.mean(mses_, axis=0)
        else:
            raise ValueError()

        self.recal_factor = recal_factors[np.argmin(mses)]
        print(self.recal_factor)

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

        if self.recalibrate_with_cv or self.recalibrate_with_testset:
            self.recalibrate_search(dataset)

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
        if (self.recalibrate_with_cv or self.recalibrate_with_testset) and hasattr(
            self, "recal_factor"
        ):
            sigma_predictive *= self.recal_factor
        return mu_predictive, sigma_predictive

