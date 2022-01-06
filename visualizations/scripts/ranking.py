from imports.general import *
from imports.ml import *


class Ranking(object):
    def __init__(self, loadpths: list[str] = [], settings: Dict[str, str] = {}):
        self.loadpths = loadpths
        self.surrogates = list(set([pth.split("|")[1] for pth in self.loadpths]))
        self.problems = list(set([pth.split("|")[2] for pth in self.loadpths]))
        self.settings = settings
        self.savepth = (
            os.getcwd()
            + "/visualizations/tables/"
            + str.join("-", [f"{key}-{val}-" for key, val in settings.items()])
        )
        self.metrics = [
            "nmse",
            "elpd",
            "y_calibration_mse",
            "y_calibration_nmse",
            "mean_sharpness",
            "x_opt_mean_dist",
            "x_opt_dist",
            "regret",
        ]
        self.metric_labels = [
            "nMSE",
            "ELPD",
            "Calibration MSE",
            "Calibration nMSE",
            "Sharpness",
            "Solution mean distance",
            "Solution distance",
            "Regret",
        ]

