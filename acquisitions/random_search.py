from imports.general import *
from imports.ml import *


class RandomSearch(object):
    """Returns random acquisition values using RandomSearch(X)"""

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        # First dim of X has to be num_samples
        return torch.rand(X.size()[0])
