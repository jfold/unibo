from imports.general import *
from imports.ml import *
from src.parameters import *
from botorch.acquisition.analytic import AnalyticAcquisitionFunction


class FullyBayes(object):
    """Returns random acquisition values using RandomSearch(X)"""

    def __init__(
        self,
        parameters: Parameters,
        surrogate: BatchedMultiOutputGPyTorchModel,
        acquisition: AnalyticAcquisitionFunction,
        y_opt_tensor: torch.Tensor,
    ) -> None:
        self.__dict__.update(asdict(parameters))
        self.acquisition = acquisition
        self.surrogate = surrogate
        self.y_opt_tensor = y_opt_tensor

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        acquisition_vals = np.full((X.shape[0], self.fully_bayes_samples), np.nan)
        surrogate_densities = np.full((self.fully_bayes_samples,), np.nan)
        kernelparams = self.surrogate.covar_module.base_kernel

        for sample in range(2 * self.d):
            acquisition_function = self.acquisition(
                self.surrogate_model,
                best_f=self.y_opt_tensor,
                maximize=self.maximization,
            )

        return torch.rand(X.size()[0])
