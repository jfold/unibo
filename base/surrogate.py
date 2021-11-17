from abc import ABC, abstractmethod, abstractproperty
from imports.general import *


class Surrogate(ABC):
    """Calibration base class """

    def __init__(self):
        super().__init__()

    # @abstractproperty
    # def name(self):
    #     pass

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Union[np.ndarray, np.ndarray]:
        pass
