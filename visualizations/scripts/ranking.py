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
            + "/visualizations/tables/"
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

    def nansum(self, a, **kwargs) -> np.ndarray:
        mx = np.isnan(a).all(**kwargs)
        res = np.nansum(a, **kwargs)
        res[mx] = np.nan
        return res

    def calc_surrogate_ranks(self, save: bool = True):
        rank_axis = self.loader_summary["surrogate"]["axis"]
        all_probs = "".join(list(self.data_settings.keys()))
        self.rankings = np.full(self.data.shape, np.nan)
        for problem in self.loader_summary["problem"]["vals"]:
            for dim in self.loader_summary["d"]["vals"]:
                if f"({dim}){problem}" not in all_probs:
                    continue
                for seed in self.loader_summary["seed"]["vals"]:
                    for metric in self.loader_summary["metric"]["vals"]:
                        t0 = time.time()
                        settings = {
                            "problem": problem,
                            "d": dim,
                            "metric": metric,
                            "bo": True,
                            "seed": seed,
                        }
                        data = self.extract(settings=settings)

                        if not np.all(np.isnan(data)):
                            if (
                                "RS" in self.loader_summary["acquisition"]["vals"]
                                and len(self.loader_summary["acquisition"]["vals"]) == 2
                            ):
                                data = self.nansum(
                                    data,
                                    **{
                                        "axis": self.loader_summary["acquisition"][
                                            "axis"
                                        ],
                                        "keepdims": True,
                                    },
                                )

                            if self.metric_dict[metric][1] == 1:
                                order = self.shuffle_argsort(-data, axis=rank_axis)
                            else:
                                order = self.shuffle_argsort(data, axis=rank_axis)
                            ranks = self.shuffle_argsort(order, axis=rank_axis) + 1
                            # check when debugging:
                            # print(metric)
                            # print(data.squeeze()[:, :3])
                            # print(ranks.squeeze()[:, :3])
                            # raise ValueError()
                            indexer = self.get_indexer(settings)
                            self.rankings[indexer] = ranks

        if save:
            with open(os.getcwd() + "/results/rankings.npy", "wb") as f:
                np.save(f, self.rankings)

    def rank_correlation(
        self, settings: Dict = {"metric": ["regret", "y_calibration_mse"]}
    ) -> None:
        rankings = np.load(os.getcwd() + "/results/rankings.npy")
        ranking_dir = list(settings.keys())[0]
        assert len(list(settings.keys())) == 1 and len(settings[ranking_dir]) == 2
        x = self.extract(
            rankings, settings={"bo": True, ranking_dir: settings[ranking_dir][0]},
        )
        y = self.extract(
            rankings, settings={"bo": True, ranking_dir: settings[ranking_dir][1]},
        )

        x, y = self.remove_nans(x.squeeze(), y.squeeze())
        rho, pval = spearmanr(x, y)
        print(rho, pval)

    def metric_correlation(
        self, settings: Dict = {"metric": ["regret", "y_calibration_mse"]}
    ) -> None:
        ranking_dir = list(settings.keys())[0]
        assert len(list(settings.keys())) == 1 and len(settings[ranking_dir]) == 2
        x = self.extract(settings={"bo": True, ranking_dir: settings[ranking_dir][0]})
        y = self.extract(settings={"bo": True, ranking_dir: settings[ranking_dir][1]})
        x, y = self.remove_nans(x.squeeze(), y.squeeze())
        rho, pval = pearsonr(x, y)
        print(rho, pval)

    def do_ranking(
        self, data: np.ndarray, settings: Dict, rank_axis: int, direction: int
    ) -> Tuple[np.array, Tuple]:
        idx = self.get_indexer(settings)
        if direction == 1:
            order = self.shuffle_argsort(-data, axis=rank_axis)
        else:
            order = self.shuffle_argsort(data, axis=rank_axis)
        ranks = self.shuffle_argsort(order, axis=rank_axis) + 1
        ranks = np.argsort(order, axis=rank_axis) + 1
        return ranks, idx

    def calc_surrogate_ranks_no_bo(self, save: bool = True) -> np.ndarray:
        rank_axis = self.loader_summary["surrogate"]["axis"]
        all_probs = "".join(list(self.data_settings.keys()))
        self.rankings = np.full(self.data.shape, np.nan)
        start_time = time.time()
        for problem in self.loader_summary["problem"]["vals"]:
            for dim in self.loader_summary["d"]["vals"]:
                if f"({dim}){problem}" not in all_probs:
                    continue
                for seed in self.loader_summary["seed"]["vals"]:
                    for metric in self.loader_summary["metric"]["vals"]:
                        settings = {
                            "problem": problem,
                            "d": dim,
                            "metric": metric,
                            "seed": seed,
                            "acquisition": "EI",
                            "bo": False,
                            "epoch": 90,
                        }
                        data = self.extract(settings=settings)
                        ranking, idx = self.do_ranking(
                            data, settings, rank_axis, self.metric_dict[metric][1]
                        )
                        self.rankings[idx] = ranking

        print(f"Ranking took: --- %s seconds ---" % (time.time() - start_time))
        if save:
            with open(os.getcwd() + "/results/rankings-no-bo.npy", "wb") as f:
                np.save(f, self.rankings)
        return self.rankings

    def save_to_tex(self, df: pd.DataFrame, name: str):
        with open(f"{self.savepth_tables}/{name}.tex", "w") as file:
            file.write(df.to_latex(escape=False))
        file.close()

    def ranking_no_bo(self, save: bool = True, update: bool = False) -> None:
        rankings = (
            self.calc_surrogate_ranks_no_bo(save)
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
                table.loc[[surrogate], [metric]] = "${:.2f} \pm {:.2f}$".format(
                    np.nanmean(data), np.nanstd(data) * sem_div
                )
        table = table.rename(
            columns={x: self.metric_dict[x][-1] for x in non_bo_metrics}
        )
        if save:
            self.save_to_tex(table, name="ranking-no-bo")

    def rank_metrics_vs_epochs(
        self,
        avg_names: list[str] = ["seed", "problem", "d"],
        settings: Dict = {"bo": True},
        calc_sur_ranks: bool = False,
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
            self.calc_surrogate_ranks()
            if calc_sur_ranks
            else np.load(os.getcwd() + "/results/rankings.npy")
        )
        rankings = self.extract(rankings, settings=settings)
        avg_dims = tuple([self.loader_summary[name]["axis"] for name in avg_names])
        sem_multi = 1.96 / np.sqrt(10 * 5 * 8)
        epochs = self.loader_summary["epoch"]["vals"]
        ranking_mean = np.nanmean(rankings, axis=avg_dims, keepdims=True)
        ranking_std = np.nanstd(rankings, axis=avg_dims, keepdims=True)

        surrogates = self.loader_summary["surrogate"]["vals"]
        surrogate_axis = self.loader_summary["surrogate"]["axis"]
        metric_axis = self.loader_summary["metric"]["axis"]

        indexer = [np.s_[:]] * ranking_mean.ndim
        if (
            "RS" in self.loader_summary["acquisition"]["vals"]
            and len(self.loader_summary["acquisition"]["vals"]) == 2
        ):
            indexer[self.loader_summary["acquisition"]["axis"]] = np.s_[0:1]
        fig = plt.figure()
        for i_m, metric in enumerate(metrics):
            print(metric)
            indexer[metric_axis] = np.s_[i_m : i_m + 1]
            ax = plt.subplot(len(metrics), 1, i_m + 1)
            for i_s, surrogate in enumerate(surrogates):
                if surrogate != "RS" or (
                    surrogate == "RS"
                    and metric
                    in [
                        "regret",
                        "true_regret",
                        "x_opt_mean_dist",
                        "mahalanobis_dist",
                        "running_inner_product",
                    ]
                ):
                    indexer[surrogate_axis] = np.s_[i_s : i_s + 1]
                    means = ranking_mean[tuple(indexer)].squeeze()
                    stds = ranking_std[tuple(indexer)].squeeze()
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
