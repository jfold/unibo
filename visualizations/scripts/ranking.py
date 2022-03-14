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
        all_probs = "".join(list(self.data_settings.keys()))
        self.rankings = np.full(self.data.shape, np.nan)
        start_time = time.time()
        settings = {"bo": with_bo}
        if not with_bo:
            settings.update({"epoch": 90})

        for problem in self.loader_summary["problem"]["vals"]:
            for dim in self.loader_summary["d"]["vals"]:
                if f"({dim}){problem}" not in all_probs:
                    continue
                for seed in self.loader_summary["seed"]["vals"]:
                    for metric in self.loader_summary["metric"]["vals"]:
                        settings.update(
                            {
                                "problem": problem,
                                "d": dim,
                                "metric": metric,
                                "seed": seed,
                                "acquisition": "EI",
                            }
                        )
                        data = self.extract(settings=settings)
                        ranking, idx = self.rank(
                            data, settings, rank_axis, self.metric_dict[metric][1]
                        )
                        self.rankings[idx] = ranking

                        ############### When debugging:
                    #     print(metric, self.metric_dict[metric])
                    #     print(data.squeeze()[:, :2])
                    #     print(ranking.squeeze()[:, :2])
                    # raise ValueError()
        print(f"Ranking took: --- %s seconds ---" % (time.time() - start_time))
        add_str = "with" if with_bo else "no"
        if save:
            with open(f"{os.getcwd()}/results/rankings-{add_str}-bo.npy", "wb") as f:
                np.save(f, self.rankings)
        return self.rankings

    def table_ranking_no_bo(self, save: bool = True, update: bool = True) -> None:
        rankings = (
            self.calc_surrogate_ranks(with_bo=False, save=save)
            if update
            else np.load(os.getcwd() + "/results/rankings-no-bo.npy")
        )
        non_bo_metrics = [
            "nmse",
            "elpd",
            "mean_sharpness",
            "y_calibration_mse",
            "uct-avg_calibration-miscal_area",
        ]
        table = pd.DataFrame(
            columns=non_bo_metrics, index=self.loader_summary["surrogate"]["vals"],
        )
        for metric in non_bo_metrics:
            for surrogate in self.loader_summary["surrogate"]["vals"]:
                settings = {
                    "surrogate": surrogate,
                    "metric": metric,
                    "bo": False,
                    "epoch": 90,
                }
                data = self.extract(rankings, settings=settings)
                sem_div = 1.96 / np.sqrt(np.sum(np.isfinite(data)))
                table[metric][surrogate] = "${:.2f} \pm {:.2f}$".format(
                    np.nanmean(data), np.nanstd(data) * sem_div
                )
        table = table.rename(
            columns={x: self.metric_dict[x][-1] for x in non_bo_metrics}
        )
        if save:
            self.save_to_tex(table, name="ranking-no-bo")

    def table_ranking_with_bo(self, save: bool = True, update: bool = False) -> None:
        rankings = (
            self.calc_surrogate_ranks(with_bo=True, save=save)
            if update
            else np.load(os.getcwd() + "/results/rankings-with-bo.npy")
        )
        bo_metrics = [
            "nmse",
            "elpd",
            "mean_sharpness",
            "y_calibration_mse",
            "uct-avg_calibration-miscal_area",
            "true_regret",
            "mahalanobis_dist",
        ]
        table = pd.DataFrame(
            columns=bo_metrics, index=self.loader_summary["surrogate"]["vals"],
        )
        for metric in bo_metrics:
            for surrogate in self.loader_summary["surrogate"]["vals"]:
                settings = {
                    "surrogate": surrogate,
                    "metric": metric,
                    "bo": True,
                    "acquisition": "EI",
                }
                data = self.extract(rankings, settings=settings)
                sem_div = 1.96 / np.sqrt(np.sum(np.isfinite(data)))
                table[metric][surrogate] = "${:.2f} \pm {:.2f}$".format(
                    np.nanmean(data), np.nanstd(data) * sem_div
                )
        table = table.rename(columns={x: self.metric_dict[x][-1] for x in bo_metrics})
        if save:
            self.save_to_tex(table, name="ranking-with-bo")

    def rank_metrics_vs_epochs(
        self,
        avg_names: list[str] = ["seed", "problem", "d", "acquisition"],
        settings: Dict = {"bo": True},
        update: bool = False,
        metrics: list[str] = [
            "y_calibration_mse",
            "nmse",
            "elpd",
            "mean_sharpness",
            "x_opt_mean_dist",
            "regret",
            "true_regret",
            "mahalanobis_dist",
        ],
    ):
        matplotlib.rcParams["font.size"] = 18
        matplotlib.rcParams["figure.figsize"] = (10, 16)
        rankings = (
            self.calc_surrogate_ranks(with_bo=True, save=True)
            if update
            else np.load(os.getcwd() + "/results/rankings-with-bo.npy")
        )
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
                means = np.nanmean(rankings_, axis=avg_dims, keepdims=True).squeeze()
                stds = np.nanstd(rankings_, axis=avg_dims, keepdims=True).squeeze()
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
            if metric not in ["regret", "true_regret", "x_opt_distance"]:
                plt.ylim([1 + 0.1, len(surrogates) - 1 + 0.1])
            plt.yticks(range(1, 1 + len(surrogates)))

        plt.tight_layout()
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=len(surrogates),
        )
        plt.xlabel("Iterations")
        plt.show()
        fig.savefig(f"{self.savepth_figs}ranking-metrics-vs-epochs---{settings}.pdf")
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
