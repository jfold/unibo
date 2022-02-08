from imports.general import *
from imports.ml import *
from numpy.core import numeric as _nx
from visualizations.scripts.loader import Loader


class Ranking(Loader):
    def __init__(
        self,
        loadpths: list[str] = [],
        settings: Dict[str, str] = {},
        update_data: bool = True,
    ):
        super(Ranking, self).__init__(loadpths, settings, update=update_data)
        if not self.data_was_loaded:
            raise ValueError("No data could be loaded")
        self.savepth_figs = (
            os.getcwd()
            + "/visualizations/figures/"
            + str.join("-", [f"{key}-{val}-" for key, val in settings.items()])
        )
        self.savepth_tables = (
            os.getcwd()
            + "/visualizations/figures/"
            + str.join("-", [f"{key}-{val}-" for key, val in settings.items()])
        )
        # self.init_tables()

    def init_tables(self) -> None:
        rows = self.metric_dict.keys()
        cols = self.loader_summary["surrogate"]["vals"]
        self.mean_ranking_table = pd.DataFrame(columns=cols, index=rows)
        self.median_ranking_table = pd.DataFrame(columns=cols, index=rows)
        self.std_ranking_table = pd.DataFrame(columns=cols, index=rows)
        self.no_ranking_table = pd.DataFrame(columns=cols, index=rows)

    def get_indexer(self, settings: Dict) -> Tuple[int]:
        axes = {}
        for key, val in settings.items():
            axes.update({key: [self.loader_summary[key]["axis"], val]})

        indexer = [np.s_[:]] * self.data.ndim
        for name, lst in axes.items():
            vals = self.loader_summary[name]["vals"]
            indexer[lst[0]] = np.s_[vals.index(lst[1]) : 1 + vals.index(lst[1])]

        return tuple(indexer)

    def nansum(self, a, **kwargs):
        mx = np.isnan(a).all(**kwargs)
        res = np.nansum(a, **kwargs)
        res[mx] = np.nan
        return res

    def calc_surrogate_ranks(self, bo: bool = True, save: bool = True):
        rankings = np.full(self.data.shape, np.nan)
        rank_axis = self.loader_summary["surrogate"]["axis"]
        for problem in self.loader_summary["problem"]["vals"]:
            for dim in self.loader_summary["d"]["vals"]:
                for seed in self.loader_summary["seed"]["vals"]:
                    for metric in self.loader_summary["metric"]["vals"]:
                        settings = {
                            "problem": problem,
                            "d": dim,
                            "metric": metric,
                            "bo": bo,
                            "seed": seed,
                        }
                        data = self.extract(settings=settings)

                        if np.all(np.isnan(data)):
                            continue

                        if (
                            "RS" in self.loader_summary["acquisition"]["vals"]
                            and len(self.loader_summary["acquisition"]["vals"]) == 2
                        ):
                            data = self.nansum(
                                data,
                                **{
                                    "axis": self.loader_summary["acquisition"]["axis"],
                                    "keepdims": True,
                                },
                            )
                        if self.metric_dict[metric][1] == 1:
                            order = np.flip(
                                self.shuffle_argsort(data, axis=rank_axis),
                                axis=rank_axis,
                            )
                        else:
                            order = self.shuffle_argsort(data, axis=rank_axis)
                        ranks = np.argsort(order, axis=rank_axis) + 1
                        indexer = self.get_indexer(settings)
                        rankings[indexer] = ranks

                        # if metric == self.loader_summary["metric"]["vals"][-1]:
                        #     raise ValueError()

        if save:
            with open(os.getcwd() + "/rankings.npy", "wb") as f:
                np.save(f, rankings)
        return rankings

    def rank_metrics_vs_epochs(
        self,
        avg_names: list[str] = ["seed", "problem", "d"],
        settings: Dict = {"bo": True},
    ):
        matplotlib.rcParams["font.size"] = 18
        matplotlib.rcParams["figure.figsize"] = (12, 18)
        # rankings = self.calc_surrogate_ranks()
        rankings = np.load(os.getcwd() + "/rankings.npy")
        rankings = self.extract(rankings, settings=settings)
        avg_dims = tuple([self.loader_summary[name]["axis"] for name in avg_names])
        n_avgs = 10 * 5 * 8
        epochs = self.loader_summary["epoch"]["vals"]
        ranking_mean = np.nanmean(rankings, axis=avg_dims, keepdims=True)
        ranking_std = np.nanstd(rankings, axis=avg_dims, keepdims=True)

        surrogates = self.loader_summary["surrogate"]["vals"]
        surrogate_axis = self.loader_summary["surrogate"]["axis"]
        metrics = self.loader_summary["metric"]["vals"]  # [:7]
        metric_axis = self.loader_summary["metric"]["axis"]

        indexer = [np.s_[:]] * ranking_mean.ndim
        if (
            "RS" in self.loader_summary["acquisition"]["vals"]
            and len(self.loader_summary["acquisition"]["vals"]) == 2
        ):
            indexer[self.loader_summary["acquisition"]["axis"]] = np.s_[0:1]
        fig = plt.figure()
        for i_m, metric in enumerate(metrics):
            indexer[metric_axis] = np.s_[i_m : i_m + 1]
            ax = plt.subplot(len(metrics), 1, i_m + 1)
            print(metric)
            for i_s, surrogate in enumerate(surrogates):
                indexer[surrogate_axis] = np.s_[i_s : i_s + 1]
                means = ranking_mean[indexer].squeeze()
                stds = ranking_std[indexer].squeeze()

                if surrogate != "RS" or (
                    surrogate == "RS"
                    and metric in ["regret", "true_regret", "x_opt_mean_dist"]
                ):
                    plt.plot(
                        epochs,
                        means,
                        color=ps[surrogate]["c"],
                        marker=ps[surrogate]["m"],
                        label=f"${surrogate}$",
                    )
                    plt.fill_between(
                        epochs,
                        means + 1 * stds / n_avgs,
                        means - 1 * stds / n_avgs,
                        color=ps[surrogate]["c"],
                        alpha=0.1,
                    )

            if i_m < len(metrics) - 1:
                ax.set_xticklabels([])
            plt.title(self.metric_dict[metric][-1])
            plt.xlim([epochs[0] - 0.1, epochs[-1] + 0.1])
            plt.ylim([1 + 0.1, len(surrogates) + 0.1])
            if metric not in ["regret", "true_regret", "x_opt_distance"]:
                plt.ylim([1 + 0.1, len(surrogates) - 1 + 0.1])
            plt.yticks(range(1, 1 + len(surrogates)))

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=len(surrogates),
        )
        plt.xlabel("Iterations")
        plt.tight_layout()
        fig.savefig(f"{self.savepth_figs}ranking-metrics-vs-epochs---{settings}.pdf")
        plt.close()

    def shuffle_argsort(self, array: np.ndarray, axis: int = None) -> np.ndarray:
        numerical_noise = np.random.uniform(0, 1e-10, size=array.shape)
        if not (np.all(np.argsort(array + numerical_noise) == np.argsort(array))):
            print("Tie! Picking winner at random.")
        if axis is None:
            return np.argsort(array + numerical_noise)
        else:
            return np.argsort(array + numerical_noise, axis=axis)

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
