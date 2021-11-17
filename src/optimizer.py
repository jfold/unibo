from base.surrogate import Surrogate
from src.parameters import *


class Optimizer(object):
    def __init__(self, parameters: Parameters, surrogate: Surrogate) -> None:
        super().__init__()
        self.surrogate = surrogate
