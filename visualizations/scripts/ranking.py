from imports.general import *
from imports.ml import *


class Ranking(object):
    def __init__(self, loadpths: list[str] = [], settings: Dict[str, str] = {}):
        np.random.seed(2022)
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
        self.seeds = list(range(1, 30 + 1))
        self.epochs = list(range(1, 50 + 1))
        self.ranking_direction = {  # -1 indicates if small is good, 1 indicates if large is good
            "nmse": -1,
            "elpd": 1,
            "y_calibration_mse": -1,
            "y_calibration_nmse": -1,
            "mean_sharpness": 1,
            "regret": -1,
            "x_opt_dist": -1,
            "x_opt_mean_dist": -1,
        }
        self.savepth = (
            os.getcwd()
            + "/visualizations/tables/"
            + str.join("-", [f"{key}-{val}-" for key, val in settings.items()])
        )
        self.figsavepth = (
            os.getcwd()
            + "/visualizations/figures/"
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

    def shuffle_argsort(self, array: np.ndarray) -> np.ndarray:
        numerical_noise = np.random.uniform(0, 1e-7, size=array.shape)
        if not (np.all(np.argsort(array + numerical_noise) == np.argsort(array))):
            # print("Tie! Picking winner at random.")
            pass
        return np.argsort(array + numerical_noise)

    def extract(self) -> None:
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

    def calc_ranking(self) -> None:
        self.rankings = np.full(self.results.shape, np.nan)
        for i_pro, problem in enumerate(self.problems):
            for i_acq, acquisition in enumerate(self.acquisitions):
                for i_met, metric in enumerate(self.ranking_direction.keys()):
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
                            if self.ranking_direction[metric] == 1:
                                for score in scores:
                                    rankings.append(self.shuffle_argsort(score)[::-1])
                            else:
                                for score in scores:
                                    rankings.append(self.shuffle_argsort(score))
                            rankings = np.array(rankings)
                            self.rankings[
                                i_pro, :, i_acq, i_met, i_dim, i_see, :
                            ] = rankings.T

        self.missing_experiments_per = (
            np.sum(np.isnan(self.rankings)) / self.rankings.size
        )

        for i_sur, surrogate in enumerate(self.surrogates):
            for i_met, metric in enumerate(self.ranking_direction.keys()):
                mean_rank = np.nanmean(self.rankings[:, i_sur, :, i_met, :, :])
                std_rank = np.nanstd(self.rankings[:, i_sur, :, i_met, :, :])
                no_trials = np.sum(np.isfinite(self.rankings[:, i_sur, :, i_met, :, :]))
                self.mean_ranking_table[metric][surrogate] = 1 + mean_rank
                self.std_ranking_table[metric][surrogate] = 1 + std_rank
                self.no_ranking_table[metric][surrogate] = no_trials

    def init_tables(self) -> None:
        rows = self.surrogates
        cols = self.ranking_direction.keys()
        self.mean_ranking_table = pd.DataFrame(columns=cols, index=rows)
        self.std_ranking_table = pd.DataFrame(columns=cols, index=rows)
        self.no_ranking_table = pd.DataFrame(columns=cols, index=rows)

    def remove_nans(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert x.shape == y.shape
        is_nan_idx = np.isnan(x)
        y = y[np.logical_not(is_nan_idx)]
        x = x[np.logical_not(is_nan_idx)]
        is_nan_idx = np.isnan(y)
        x = x[np.logical_not(is_nan_idx)]
        y = y[np.logical_not(is_nan_idx)]
        return x, y

    def remove_extremes(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert x.shape == y.shape
        remove_idx = y > np.quantile(y, 0.95)
        y = y[np.logical_not(remove_idx)]
        x = x[np.logical_not(remove_idx)]
        remove_idx = y < np.quantile(y, 0.05)
        x = x[np.logical_not(remove_idx)]
        y = y[np.logical_not(remove_idx)]
        return x, y

    def calc_plot_metric_dependence(
        self,
        metric_1: str = "regret",
        metric_2: str = "y_calibration_mse",
        n_epoch: int = -1,
    ):
        rho = r"$\rho$"
        met_1 = self.metrics_arr.index(metric_1)
        met_2 = self.metrics_arr.index(metric_2)
        self.metric_dependence = {}
        for i_p, problem in enumerate(self.problems):
            fig = plt.figure()
            for i_sur, surrogate in enumerate(self.surrogates):
                x = self.results[i_p, i_sur, :, met_1, :, :, n_epoch].flatten()
                y = self.results[i_p, i_sur, :, met_2, :, :, n_epoch].flatten()
                x, y = self.remove_nans(x, y)
                x, y = self.remove_extremes(x, y)
                pearson, p_value = pearsonr(x, y)
                mi = mutual_info_regression(x[:, np.newaxis], y)[0]
                self.metric_dependence.update(
                    {
                        f"{surrogate}:linearcorr": (pearson, p_value),
                        f"{surrogate}:mi": mi,
                    }
                )
                plt.plot(
                    x,
                    y,
                    "o",
                    label=f"{surrogate} ({rho}={np.round(pearson,2)},mi={np.round(mi,2)})",
                    alpha=1 - i_sur / (i_sur + 2),
                )
            plt.xlabel(self.metrics[metric_1])
            plt.ylabel(self.metrics[metric_2])
            plt.legend()
            plt.xscale("log")
            m1 = metric_1.replace("_", "-")
            m2 = metric_2.replace("_", "-")
            fig.savefig(
                f"{self.figsavepth}{m1}-vs-{m2}-{problem}-epoch-{n_epoch}.pdf".replace(
                    ",", "-"
                )
            )

    def run(self):
        self.extract()
        self.init_tables()
        self.calc_ranking()
        self.calc_plot_metric_dependence(
            metric_1="regret", metric_2="y_calibration_mse", n_epoch=-1
        )
        self.calc_plot_metric_dependence(metric_1="regret", metric_2="elpd", n_epoch=-1)

        self.mean_ranking_table.applymap("{:.4f}".format).to_csv(
            f"{self.savepth}means.csv",
        )
        self.std_ranking_table.applymap("{:.4f}".format).to_csv(
            f"{self.savepth}stds.csv"
        )
        print(self.mean_ranking_table)
        print(self.std_ranking_table)


# TODO:
# Rank (1,2,3) som funktion af epoker for:
# y_calibration_nmse, mean_sharpness, regret, x_opt_dist
# I hver epoke regner vi korrelationscoefficienten (mutual information?)
# mellem

# 1)
# Contour plot i 2D -> init som punkter + epokevalg som tal
# 2)
# 3x1 subplot plot:
# Contour plot i 2D + punkter op til en specifik iteration
# posterior middel for en specifik iteration
# posterior varians for en specifik iteration
# acquisition for en specifik iteration
# 3)
# Inkluder GP med ændret std
# 4) Mikkel
# Generer data fra en GP med kendte hyperparametre

