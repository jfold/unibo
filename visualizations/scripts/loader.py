from imports.general import *
from imports.ml import *


class Loader(object):
    def __init__(self, loadpths: list[str] = [], settings: Dict[str, str] = {}):
        self.loadpths = loadpths
        self.settings = settings
        self.metric_dict = {
            "nmse": ["nMSE", -1, r"$ \mathcal{nMSE}$"],
            "elpd": ["ELPD", 1, r"$ \mathcal{ELPD}$"],
            "y_calibration_mse": [
                "Calibration MSE",
                -1,
                r"$ \mathbb{E}[(\mathcal{C}_{\mathbf{y}}(p) - p)^2] $",
            ],
            # "y_calibration_nmse": ["Calibration nMSE", -1,],
            "mean_sharpness": ["Sharpness", 1, r"$ \mathcal{S}$"],
            "x_opt_mean_dist": [
                "Solution mean distance",
                -1,
                r"$ \mathbb{E}[||\textbf{x}_o - \textbf{x}_s||_2] $",
            ],
            "x_opt_dist": [
                "Solution distance",
                -1,
                r"$ ||\textbf{x}_o - \textbf{x}_s||_2 $",
            ],
            "regret": ["Regret", -1, r"$ \mathcal{R}$"],
        }
        self.check_params = [
            "seed",
            "d",
            "n_test",
            "n_initial",
            "n_evals",
            "problem",
            "change_std",
            "surrogate",
            "acquisition",
            "bo",
        ]
        self.skim_data()

    def skim_data(self):
        self.data_settings = {}
        self.data_summary = {k: [] for k in self.check_params}
        for i_e, experiment in enumerate(p for p in self.loadpths):
            if (
                os.path.isdir(experiment)
                and os.path.isfile(f"{experiment}parameters.json")
                and os.path.isfile(f"{experiment}scores.json")
                and os.path.isfile(f"{experiment}dataset.json")
            ):
                with open(f"{experiment}parameters.json") as json_file:
                    parameters = json.load(json_file)
                with open(f"{experiment}scores.json") as json_file:
                    scores = json.load(json_file)
                with open(f"{experiment}dataset.json") as json_file:
                    dataset = json.load(json_file)

                if not self.settings.items() <= parameters.items() or not all(
                    param in parameters for param in self.check_params
                ):
                    continue

                self.data_settings.update(
                    {experiment: {k: parameters[k] for k in self.check_params}}
                )

                for k in self.check_params:
                    lst = self.data_summary[k]
                    lst.append(parameters[k])
                    self.data_summary.update({k: lst})

    def make_result_object(self):
        self.data_settings

    def load_data(self):
        self.data_settings = {}
        for i_e, experiment in enumerate(p for p in self.loadpths):
            if (
                os.path.isdir(experiment)
                and os.path.isfile(f"{experiment}parameters.json")
                and os.path.isfile(f"{experiment}scores.json")
                and os.path.isfile(f"{experiment}dataset.json")
            ):
                with open(f"{experiment}parameters.json") as json_file:
                    parameters = json.load(json_file)
                with open(f"{experiment}scores.json") as json_file:
                    scores = json.load(json_file)
                with open(f"{experiment}dataset.json") as json_file:
                    dataset = json.load(json_file)

                if not self.settings.items() <= parameters.items():
                    continue

                self.data_settings.update({experiment: parameters})
                i_pro = self.problems.index(parameters["problem"])
                i_see = self.seeds.index(parameters["seed"])
                i_sur = self.surrogates.index(parameters["surrogate"])
                i_acq = self.acquisitions.index(parameters["acquisition"])
                i_dim = self.ds.index(parameters["d"])

                # Running over epochs
                files_in_path = [f for f in os.listdir(experiment) if "scores" in f]
                for file in files_in_path:
                    if "---epoch-" in file:
                        i_epo = int(file.split("---epoch-")[-1].split(".json")[0]) - 1
                    else:
                        i_epo = len(self.epochs) - 1

                    with open(experiment + file) as json_file:
                        scores = json.load(json_file)

                    for i_met, metric in enumerate(self.metrics.keys()):
                        self.results[
                            i_pro, i_sur, i_acq, i_met, i_dim, i_see, i_epo
                        ] = scores[metric]
