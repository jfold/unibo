from dataclasses import asdict
from botorch.posteriors.posterior import Posterior
from src.dataset import Dataset
from src.parameters import Parameters
from imports.general import *
from imports.ml import *


class BayesianNeuralNetwork(BatchedMultiOutputGPyTorchModel):
    """Random forest surrogate class. """

    def __init__(
        self, parameters: Parameters, dataset: Dataset, name: str = "BNN",
    ):
        self.__dict__.update(asdict(parameters))
        raise NotImplementedError()
        self.fit(X_train=dataset.data.X, y_train=dataset.data.y)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        **kwargs: Any
    ) -> Posterior:
        return super().posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            **kwargs
        )

    def batch_shape(self) -> torch.Size:
        pass

    def train(self, mode: bool = True) -> Model:
        pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits model 
        """
        raise NotImplementedError()

    def predict(
        self, X_test: np.ndarray, stabilizer: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates mean (prediction) and variance (uncertainty)
        """
        raise NotImplementedError()
