from src.parameters import *
from datasets.verifications.verification import VerificationData


class Dataset(object):
    def __init__(self, parameters: Parameters = Defaults()) -> None:
        self.__dict__.update(parameters.__dict__)
        self.data = VerificationData(parameters)

    def add_X_sample_y(self, x_new: np.array):
        self.data.X = np.append(self.data.X, x_new, axis=0)
        y_new = self.data.sample_y(x_new)
        self.data.y = np.append(self.data.y, y_new, axis=0)

    def sample_testset(self, n_samples: int = None):
        n_samples = self.n_test if n_samples is None else n_samples
        X = self.data.sample_X(n_samples=n_samples)
        y = self.data.sample_y(X)
        return X, y
