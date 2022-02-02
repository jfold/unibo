from imports.general import *
from imports.ml import *


class RandomSearch(object):
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return torch.rand(X.size()[0])
