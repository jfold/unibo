from base.surrogate import Surrogate
from src.parameters import *
import torch
import botorch


class Optimizer(object):
    """Optimizer wrapper for botorch"""

    def __init__(self, parameters: Parameters) -> None:
        super().__init__()
