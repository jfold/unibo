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
        self.metrics = {
            "nmse": "nMSE",
            "elpd": "ELPD",
            "y_calibration_mse": "Calibration MSE",
            "y_calibration_nmse": "Calibration nMSE",
            "mean_sharpness": "Sharpness",
            "x_opt_mean_dist": "Solution mean distance",
            "x_opt_dist": "Solution distance",
            "regret": "Regret",
        }
        self.surrogates = ["GP", "RF", "BNN"]
        self.acquisitions = ["EI"]
        self.epochs = list(range(50))
        self.ds = [2]
        self.seeds = list(range(10))

    def f_best_i(self):
        """the best seen objective value after i objective function evaluations"""
        pass

    def auc(self):
        """the best seen objective value after i objective function evaluations"""
        pass

    def borda_ranking_system(self):
        pass

    def no_first_and_least_last(self):
        pass

    def run(self):
        surrogates = []
        seeds = []
        acquisitions = []
        ds = []
        arrays = []
        results = np.full((100, 100, 100), np.nan)
        for experiment in self.loadpths:
            if os.path.isdir(experiment) and os.path.isfile(
                experiment + "parameters.json"
            ):
                with open(experiment + "parameters.json") as json_file:
                    parameters = json.load(json_file)
                if not self.settings.items() <= parameters.items():
                    continue
                seeds.append(parameters["seed"])
                surrogates.append(parameters["surrogate"])
                acquisitions.append(parameters["acquisition"])
                ds.append(parameters["d"])

                files_in_path = [f for f in os.listdir(experiment) if "scores" in f]
                for file in files_in_path:
                    with open(experiment + file) as json_file:
                        scores = json.load(json_file)

                    e_idx = (
                        int(file.split("---epoch-")[-1].split(".json")[0]) - 1
                        if "---epoch-" in file
                        else self.n_epochs - 1
                    )

                    for i_m, metric in enumerate(self.metrics.keys()):
                        results[i_m, e_idx] = scores[metric]
        arrays.append(seeds)
        arrays.append(surrogates)
        arrays.append(acquisitions)
        arrays.append(ds)
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(
            tuples, names=["surrogate", "seeds", "acquisitions", "ds"]
        )
        for i_m, metric in enumerate(self.metrics.keys()):
            results = pd.DataFrame(np.random.randn(8, 4), index=index)
