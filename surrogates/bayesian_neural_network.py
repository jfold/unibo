from dataclasses import asdict
from botorch.posteriors.posterior import Posterior
from src.dataset import Dataset
from src.parameters import Parameters
from imports.general import *
from imports.ml import *


class BayesianNeuralNetwork(BatchedMultiOutputGPyTorchModel):
    """Bayesian Neural Network (BNN) surrogate class. """

    def __init__(
        self, parameters: Parameters, dataset: Dataset,
    ):
        self.name = parameters.surrogate
        self.model = None

    def predict(
        self, X_test: np.ndarray, stabilizer: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates mean (prediction) and variance (uncertainty)"""
        X_test = torch.tensor(X_test)
        posterior = None
        return None, None
