from src.parameters import *
from datasets.verifications.verification import VerificationData


class Dataset(object):
    def __init__(self, parameters: Parameters = Defaults()) -> None:
        super().__init__()
        self.data = VerificationData()
        self.X_dist = tfp.distributions.Uniform(low=-1, high=1)

    def sample_X(self, n_samples: int) -> np.array:
        X = self.X_dist.sample(sample_shape=(n_samples, self.d), seed=self.seed)
        return X

