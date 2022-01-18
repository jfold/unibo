from typing import Dict
from imports.general import *
from imports.ml import *
from visualizations.scripts.loader import Loader


class Tables(Loader):
    def __init__(self, loadpths: list[str] = [], settings: Dict[str, str] = {}):
        self.loadpths = loadpths
        self.surrogates = list(set([pth.split("|")[1] for pth in self.loadpths]))
        self.settings = settings
        self.savepth = (
            os.getcwd()
            + "/visualizations/tables/"
            + str.join("-", [f"{key}-{val}-" for key, val in settings.items()])
        )

    def load_raw(self):
        metrics = [
            "nmse",
            "elpd",
            "y_calibration_mse",
            "y_calibration_nmse",
            "mean_sharpness",
            "x_opt_mean_dist",
            "x_opt_dist",
            "regret",
        ]
        results = np.full(
            (len(self.surrogates), len(metrics), len(self.loadpths)), np.nan
        )
        for i_s, surrogate in enumerate(self.surrogates):
            for i_e, experiment in enumerate(
                [
                    p
                    for p in self.loadpths
                    if p.split("|")[1] == surrogate and "Benchmark" in p
                ]
            ):
                if os.path.isfile(experiment + f"scores.json") and os.path.isfile(
                    experiment + "parameters.json"
                ):

                    with open(experiment + f"scores.json") as json_file:
                        scores = json.load(json_file)
                    with open(experiment + "parameters.json") as json_file:
                        parameters = json.load(json_file)

                    if self.settings.items() <= parameters.items():
                        for i_m, metric in enumerate(metrics):
                            results[i_s, i_m, i_e] = scores[metric]
                else:
                    print(f"No such file: {experiment}scores.json")

        self.means = pd.DataFrame(
            data=np.nanmean(results, axis=-1), index=self.surrogates, columns=metrics
        )
        self.stds = pd.DataFrame(
            data=np.nanstd(results, axis=-1), index=self.surrogates, columns=metrics
        )

        self.means.to_csv(self.savepth + f"means.csv", float_format="%.2e")
        self.stds.to_csv(self.savepth + f"stds.csv", float_format="%.2e")

    def generate(self):
        self.load_raw()
        self.summary_table()

    def surrogates_vs_metrics(self):
        columns = 

