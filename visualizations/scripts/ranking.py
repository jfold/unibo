from imports.general import *
from imports.ml import *


class Ranking(object):
    def __init__(self, loadpths: list[str] = [], settings: Dict[str, str] = {}):
        self.loadpths = loadpths
        self.settings = settings
        self.problems = list(
            set([pth.split("|")[2].split("-")[-1] for pth in self.loadpths])
        )
        self.surrogates = ["GP", "RF", "BNN"]
        self.acquisitions = ["EI"]
        self.metrics_arr = [
            "nmse",
            "elpd",
            "y_calibration_mse",
            "y_calibration_nmse",
            "mean_sharpness",
            "x_opt_mean_dist",
            "x_opt_dist",
            "regret",
        ]
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
        self.ds = [2]
        self.seeds = list(range(1, 20 + 1))
        self.epochs = list(range(1, 50 + 1))
        self.ranking_metrics = {
            "y_calibration_nmse": -1,
            "mean_sharpness": 1,
            "regret": -1,
            "x_opt_dist": -1,
        }  # -1 indicates if small is good, 1 indicates if large is good
        self.savepth = (
            os.getcwd()
            + "/visualizations/tables/"
            + str.join("-", [f"{key}-{val}-" for key, val in settings.items()])
        )

    def f_best_i(self):
        """the best seen objective value after i objective function evaluations"""
        pass

    def auc(self):
        """aggregated score"""
        pass

    def borda_ranking_system(self):
        pass

    def num_of_first_and_least_last(self):
        pass

    def shuffle_argsort(self, array: np.ndarray):
        numerical_noise = np.random.uniform(0, 1e-7, size=array.shape)
        if not (np.all(np.argsort(array + numerical_noise) == np.argsort(array))):
            print("Tie!")
        return np.argsort(array + numerical_noise)

    def extract(self):
        self.results = np.full(
            (
                len(self.problems),
                len(self.surrogates),
                len(self.acquisitions),
                len(self.metrics),
                len(self.ds),
                len(self.seeds),
                len(self.epochs),
            ),
            np.nan,
        )
        for i_e, experiment in enumerate(p for p in self.loadpths):
            if os.path.isdir(experiment) and os.path.isfile(
                experiment + "parameters.json"
            ):
                with open(experiment + "parameters.json") as json_file:
                    parameters = json.load(json_file)

                if not self.settings.items() <= parameters.items():
                    continue

                try:
                    i_pro = self.problems.index(parameters["problem"])
                    i_see = self.seeds.index(parameters["seed"])
                    i_sur = self.surrogates.index(parameters["surrogate"])
                    i_acq = self.acquisitions.index(parameters["acquisition"])
                    i_dim = self.ds.index(parameters["d"])
                    # Running over epochs
                    files_in_path = [f for f in os.listdir(experiment) if "scores" in f]
                    for file in files_in_path:
                        if "---epoch-" in file:
                            i_epo = (
                                int(file.split("---epoch-")[-1].split(".json")[0]) - 1
                            )
                        else:
                            i_epo = len(self.epochs) - 1

                        with open(experiment + file) as json_file:
                            scores = json.load(json_file)

                        for i_met, metric in enumerate(self.metrics.keys()):
                            self.results[
                                i_pro, i_sur, i_acq, i_met, i_dim, i_see, i_epo
                            ] = scores[metric]
                except:
                    print("ERROR with:", parameters)
                    continue

    def calc_ranking(self):
        self.rankings = np.full(self.results.shape, np.nan)
        for i_pro, problem in enumerate(self.problems):
            for i_acq, acquisition in enumerate(self.acquisitions):
                for i_met, metric in enumerate(self.ranking_metrics.keys()):
                    for i_dim, d in enumerate(self.ds):
                        for i_see, seed in enumerate(self.seeds):
                            if np.any(  # demanding all surrogates have carried out all epochs
                                np.isnan(
                                    self.results[
                                        i_pro, :, i_acq, i_met, i_dim, i_see, :
                                    ]
                                )
                            ):
                                continue
                            scores = self.results[
                                i_pro, :, i_acq, i_met, i_dim, i_see, :,
                            ].T
                            rankings = []
                            if self.ranking_metrics[metric] == 1:
                                for score in scores:
                                    rankings.append(self.shuffle_argsort(score)[::-1])
                            else:
                                for score in scores:
                                    rankings.append(self.shuffle_argsort(score))
                            rankings = np.array(rankings)
                            self.rankings[
                                i_pro, :, i_acq, i_met, i_dim, i_see, :
                            ] = rankings.T

        for i_sur, surrogate in enumerate(self.surrogates):
            for i_met, metric in enumerate(self.ranking_metrics.keys()):
                mean_rank = np.nanmean(self.rankings[:, i_sur, :, i_met, :, :])
                std_rank = np.nanstd(self.rankings[:, i_sur, :, i_met, :, :])
                no_trials = np.sum(np.isfinite(self.rankings[:, i_sur, :, i_met, :, :]))
                self.mean_ranking_table[metric][surrogate] = 1 + mean_rank
                self.std_ranking_table[metric][surrogate] = 1 + std_rank
                self.no_ranking_table[metric][surrogate] = no_trials

    def init_tables(self):
        rows = self.surrogates
        cols = self.ranking_metrics.keys()
        self.mean_ranking_table = pd.DataFrame(columns=cols, index=rows)
        self.std_ranking_table = pd.DataFrame(columns=cols, index=rows)
        self.no_ranking_table = pd.DataFrame(columns=cols, index=rows)

    def calc_corr_coef(self):
        met_1 = self.metrics_arr.index("y_calibration_nmse")
        met_2 = self.metrics_arr.index("regret")
        print(self.rankings.shape, met_1, met_2)
        for i_sur, surrogate in enumerate(self.surrogates):
            x = self.rankings[:, i_sur, :, met_1, :, :].flatten()
            y = self.rankings[:, i_sur, :, met_2, :, :].flatten()
            print(np.sum(np.isfinite(x)), np.sum(np.isfinite(y)))
            # pearson = np.corrcoef(x, y)
            # mi = mutual_info_regression(x[:, np.newaxis], y)
            # print(surrogate, pearson, mi)

    def run(self):
        self.extract()
        self.init_tables()
        self.calc_ranking()
        self.calc_corr_coef()

        self.mean_ranking_table.applymap("{:.4f}".format).to_csv(
            f"{self.savepth}means.csv",
        )
        self.std_ranking_table.applymap("{:.4f}".format).to_csv(
            f"{self.savepth}stds.csv"
        )
        print(self.mean_ranking_table)
        print(self.std_ranking_table)
        print(self.no_ranking_table)


# TODO:
# Rank (1,2,3) som funktion af epoker for:
# y_calibration_nmse, mean_sharpness, regret, x_opt_dist
# I hver epoke regner vi korrelationscoefficienten (mutual information?)
# mellem

# Regret (x-axis) versus calibration error (y-axis) for halvvejs og slut: plot de 10 seeds

# DONE Normaliser y inden således regret kan sammenlignes: del med største numeriske værdi
