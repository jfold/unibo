from sklearn.neighbors import VALID_METRICS
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

    def get_indexer(self, settings: Dict) -> Tuple[int]:
        axes = {}
        for key, val in settings.items():
            axes.update({key: [self.loader_summary[key]["axis"], val]})

        indexer = [np.s_[:]] * self.data.ndim
        for name, lst in axes.items():
            vals = self.loader_summary[name]["vals"]
            indexer[lst[0]] = np.s_[vals.index(lst[1]) : 1 + vals.index(lst[1])]

        return tuple(indexer)

    def nansum(self, a, **kwargs) -> np.ndarray:
        mx = np.isnan(a).all(**kwargs)
        res = np.nansum(a, **kwargs)
        res[mx] = np.nan
        return res

    def rank(
        self, data: np.ndarray, settings: Dict, rank_axis: int, direction: int
    ) -> Tuple[np.array, Tuple]:
        idx = self.get_indexer(settings)
        if direction == 1:
            order = self.shuffle_argsort(-data, axis=rank_axis)
        else:
            order = self.shuffle_argsort(data, axis=rank_axis)
        ranks = self.shuffle_argsort(order, axis=rank_axis) + 1
        ranks = np.argsort(order, axis=rank_axis) + 1.0
        ranks[np.isnan(data)] = np.nan
        return ranks, idx

    def calc_surrogate_ranks(self, with_bo: bool, save: bool = True) -> np.ndarray:
        rank_axis = self.loader_summary["surrogate"]["axis"]
        acq_axis = self.loader_summary["acquisition"]["axis"]
        self.rankings = np.full(self.data.shape, np.nan)
        start_time = time.time()
        settings = {"bo": with_bo}
        if not with_bo:
            settings.update({"epoch": 90})

        add_str = "with" if with_bo else "no"

        self.rankings = np.full(self.data.shape, np.nan)
        for metric in self.loader_summary["metric"]["vals"]:
            settings.update({"metric": metric})
            data = self.extract(settings=settings)
            data = np.nanmean(data, axis=acq_axis, keepdims=True)
            ranking, idx = self.rank(
                data, settings, rank_axis, self.metric_dict[metric][1]
            )
            self.rankings[idx] = ranking

        print("missing:", np.sum(np.isnan(self.rankings)) / self.rankings.size)
        # raise ValueError()
        # for problem in self.loader_summary["problem"]["vals"]:
        #     for dim in self.loader_summary["d"]["vals"]:
        #         if f"({dim}){problem}" not in all_probs:
        #             continue
        #         for seed in self.loader_summary["seed"]["vals"]:
        #             for change_std in self.loader_summary["change_std"]["vals"]:
        #                 for metric in self.loader_summary["metric"]["vals"]:
        #                     settings.update(
        #                         {
        #                             "problem": problem,
        #                             "d": dim,
        #                             "metric": metric,
        #                             "change_std": change_std,
        #                             "seed": seed,
        #                             "acquisition": "EI",
        #                         }
        #                     )
        #                     data = self.extract(settings=settings)
        #                     ranking, idx = self.rank(
        #                         data, settings, rank_axis, self.metric_dict[metric][1]
        #                     )
        #                     self.rankings[idx] = ranking
        #                     ############### When debugging:
        #                 #     print(metric, self.metric_dict[metric])
        #                 #     print(data.squeeze()[:, :2])
        #                 #     print(ranking.squeeze()[:, :2])
        #                 # raise ValueError()

        if save:
            with open(f"{os.getcwd()}/results/rankings-{add_str}-bo.npy", "wb") as f:
                np.save(f, self.rankings)
        print(
            f"Ranking {add_str} BO took: --- %s seconds ---"
            % (time.time() - start_time)
        )
        return self.rankings

    def table_ranking_no_bo(self, save: bool = True, update: bool = True) -> None:
        rankings = np.load(os.getcwd() + "/results/rankings-no-bo.npy")
        metrics = [
            "nmse",
            "elpd",
            "mean_sharpness",
            "y_calibration_mse",
            "uct-avg_calibration-miscal_area",
        ]
        table = pd.DataFrame(
            columns=metrics, index=self.loader_summary["surrogate"]["vals"],
        )
        for metric in metrics:
            for surrogate in self.loader_summary["surrogate"]["vals"]:
                settings = {
                    "surrogate": surrogate,
                    "metric": metric,
                    "bo": False,
                    "epoch": 90,
                }
                data = self.extract(rankings, settings=settings)
                sem_div = 1.96 / np.sqrt(np.sum(np.isfinite(data)))
                table[metric][surrogate] = (
                    "${:.2f} \pm {:.2E}".format(
                        np.nanmean(data), np.nanstd(data) * sem_div
                    ).replace("E", "\cdot 10^{")
                    + "}$"
                )
        table = table.rename(columns={x: self.metric_dict[x][-1] for x in metrics})
        if save:
            self.save_to_tex(table, name="ranking-no-bo")

    def table_ranking_with_bo(self, save: bool = True, update: bool = True) -> None:
        rankings = np.load(os.getcwd() + "/results/rankings-with-bo.npy")
        metrics = [
            "nmse",
            "elpd",
            "mean_sharpness",
            "y_calibration_mse",
            "uct-avg_calibration-miscal_area",
            "true_regret",
            "mahalanobis_dist",
        ]
        table = pd.DataFrame(
            columns=metrics, index=self.loader_summary["surrogate"]["vals"],
        )
        for metric in metrics:
            for surrogate in self.loader_summary["surrogate"]["vals"]:
                settings = {
                    "surrogate": surrogate,
                    "metric": metric,
                    "bo": True,
                    "acquisition": "EI",
                }
                data = self.extract(rankings, settings=settings)
                sem_div = 1.96 / np.sqrt(np.sum(np.isfinite(data)))
                table[metric][surrogate] = (
                    "${:.2f} \pm {:.2E}".format(
                        np.nanmean(data), np.nanstd(data) * sem_div
                    ).replace("E", "\cdot 10^{")
                    + "}$"
                )
        table = table.rename(columns={x: self.metric_dict[x][-1] for x in metrics})
        if save:
            self.save_to_tex(table, name="ranking-with-bo")

    def rank_vs_epochs(
        self,
        avg_names: list[str] = ["seed", "problem", "d", "acquisition"],
        settings: Dict = {"bo": True, "change_std": False},
        metrics: list[str] = [
            # "nmse",
            # "elpd",
            "mean_sharpness",
            "y_calibration_mse",
            # "regret",
            "true_regret",
            "x_opt_mean_dist",
            "mahalanobis_dist",
        ],
        save_str: str = "",
    ):
        matplotlib.rcParams["font.size"] = 18
        matplotlib.rcParams["figure.figsize"] = (12, 7)  # (10, 16)
        rankings = np.load(os.getcwd() + "/results/rankings-with-bo.npy")
        rankings = self.extract(rankings, settings=settings)

        avg_dims = tuple([self.loader_summary[name]["axis"] for name in avg_names])
        sem_multi = 1.96 / np.sqrt(10 * 5 * 8)
        epochs = self.loader_summary["epoch"]["vals"]
        surrogates = self.loader_summary["surrogate"]["vals"]

        fig = plt.figure()
        for i_m, metric in enumerate(metrics):
            ax = plt.subplot(len(metrics), 1, i_m + 1)
            for surrogate in surrogates:
                rankings_ = self.extract(
                    rankings, settings={"surrogate": surrogate, "metric": metric}
                )
                means = np.nanmean(rankings_, axis=avg_dims).squeeze()
                stds = np.nanstd(rankings_, axis=avg_dims).squeeze()
                plt.plot(
                    epochs,
                    means,
                    color=ps[surrogate]["c"],
                    marker=ps[surrogate]["m"],
                    label=f"${surrogate}$",
                )
                plt.fill_between(
                    epochs,
                    means + stds * sem_multi,
                    means - stds * sem_multi,
                    color=ps[surrogate]["c"],
                    alpha=0.1,
                )

            if i_m < len(metrics) - 1:
                ax.set_xticklabels([])
            plt.title(self.metric_dict[metric][-1])
            plt.xlim([epochs[0] - 0.1, epochs[-1] + 0.1])
            plt.ylim([1 + 0.1, len(surrogates) + 0.1])
            plt.ylabel("Rank")
            if metric not in ["regret", "true_regret", "x_opt_distance"]:
                plt.ylim([1 + 0.1, len(surrogates) - 1 + 0.1])
            plt.yticks(range(1, 1 + len(surrogates)))

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc=(0.1, -0.01),
            ncol=len(surrogates),
            # bbox_to_anchor=(-0.05, -0.05),
            fancybox=True,
            shadow=True,
        )
        plt.xlabel("Iterations")
        plt.tight_layout()
        plt.show()
        fig.savefig(
            f"{self.savepth_figs}ranking-vs-epochs---{settings}---{save_str}.pdf"
        )
        plt.close()

    def shuffle_argsort(self, array: np.ndarray, axis: int = None) -> np.ndarray:
        numerical_noise = np.random.uniform(0, 1e-10, size=array.shape)
        is_nan = np.isnan(array)
        if not (np.all(np.argsort(array + numerical_noise) == np.argsort(array))):
            print("Tie! Picking winner at random.")
        if axis is None:
            sorted_array = np.argsort(array + numerical_noise)
        else:
            sorted_array = np.argsort(array + numerical_noise, axis=axis)
        sorted_array = sorted_array.astype(float)
        sorted_array[is_nan] = np.nan
        return sorted_array

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

    def TRIAL_RANKING_no_bo(self, save: bool = True, update: bool = True) -> None:
        rankings = np.load(os.getcwd() + "/results/rankings-no-bo.npy")
        print(self.loader_summary["surrogate"]["vals"])
        for d in range(2, 3):
            settings = {
                "metric": "y_calibration_mse",
                "problem": "BartelsConn",
                "bo": False,
                "epoch": 90,
                "d": d,
                "change_std": False,
                "acquisition": "EI",
            }
            ranking = self.extract(rankings, settings=settings)
            data = self.extract(settings=settings)
            mean_ranking = np.nanmean(
                ranking,
                axis=(
                    self.loader_summary["seed"]["axis"],
                    self.loader_summary["problem"]["axis"],
                ),
            ).squeeze()
            print(mean_ranking)
            print(data.squeeze())
