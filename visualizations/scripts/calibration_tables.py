from typing import Dict
from imports.general import *
from imports.ml import *


class Tables(object):
    def __init__(self, loadpths: list[str] = [], settings: Dict[str, str] = {}):
        self.loadpths = loadpths
        self.surrogates = list(set([pth.split("|")[1] for pth in self.loadpths]))
        self.settings = settings
        self.savepth = os.getcwd() + "/visualizations/figures/"

    def load_raw(self):
        metrics = [
            "nmse",
            "elpd",
            "y_calibration_mse",
            # "S_MSE",
            # "x_opt_mean_dist",
            # "x_opt_dist",
            # "total_regret",
            # "mean_regret",
        ]
        results = np.full(
            (len(self.surrogates), len(metrics), len(self.loadpths)), np.nan
        )

        for i_s, surrogate in enumerate(self.surrogates):
            for i_e, experiment in enumerate(
                [p for p in self.loadpths if p.split("|")[1] == surrogate]
            ):
                if os.path.isfile(experiment + "scores.json") and os.path.isfile(
                    experiment + "parameters.json"
                ):
                    with open(experiment + "scores.json") as json_file:
                        scores = json.load(json_file)
                    with open(experiment + "parameters.json") as json_file:
                        parameters = json.load(json_file)

                    if self.settings.items() <= parameters.items():
                        for i_m, metric in enumerate(metrics):
                            results[i_s, i_m, i_e] = scores[metric]

        self.means = pd.DataFrame(
            data=np.nanmean(results, axis=-1), index=self.surrogates, columns=metrics
        )
        self.stds = pd.DataFrame(
            data=np.nanstd(results, axis=-1), index=self.surrogates, columns=metrics
        )

        print(self.means)
        print(self.stds)

    def generate(self,):
        self.load_raw()
        self.summary_table()

    def summary_table(self):
        pass
