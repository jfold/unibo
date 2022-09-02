from datasets.RBF.rbf_sampler import RBFSampler
from imports.general import *
from imports.ml import *
from src.parameters import Parameters


class CustomData(object):
    """Custom data generation class."""

    def __init__(self, parameters: Parameters, data: Dict = None):
        if data is None:
            data = RBFSampler(parameters).__dict__

        self.__dict__.update(data)

    def sample_X(self, n_samples: int, return_idxs: bool = False):
        idxs = np.random.choice(list(range(self.data.params.n_test)), n_samples)
        if return_idxs:
            return self.data.X_test[idxs, :], idxs
        else:
            return self.data.X_test[idxs, :]

    def get_y(self, X: np.ndarray = None, idxs: list = None, add_noise: bool = True):
        if X is None and idxs is not None:
            y = self.data.get_y(idxs, noisify=add_noise)
        elif X is not None and idxs is not None:
            y = self.data.get_y(X, noisify=add_noise)
        else:
            raise ValueError("Either X or idx has to be None, not both.")

        return y
