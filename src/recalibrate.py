from dataclasses import asdict
import json
from numpy.lib.npyio import save
from imports.general import *
from imports.ml import *
from src.parameters import Parameters
from base.dataset import Dataset


class Recalibrator(object):
    """Calibration experiment class """

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)
        self.__dict__.update(asdict(parameters))
        self.summary = {}

