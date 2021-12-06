from abc import ABC, abstractmethod, abstractproperty
from imports.general import *


class Dataset(ABC):
    """Calibration base class """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def add_X_get_y(self, x_new: np.array):
        pass

    @abstractmethod
    def sample_testset(self, n_samples: int = None):
        pass
