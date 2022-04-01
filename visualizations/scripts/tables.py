from typing import Dict
from imports.general import *
from imports.ml import *
from visualizations.scripts.loader import Loader


class Tables(Loader):
    def __init__(
        self,
        loadpths: list[str] = [],
        settings: Dict[str, str] = {},
        update_data: bool = True,
    ):
        super(Tables, self).__init__(loadpths, settings, update=update_data)
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
                if os.path.isfile(experiment + f"metrics.json") and os.path.isfile(
                    experiment + "parameters.json"
                ):

                    with open(experiment + f"metrics.json") as json_file:
                        metrics = json.load(json_file)
                    with open(experiment + "parameters.json") as json_file:
                        parameters = json.load(json_file)

                    if self.settings.items() <= parameters.items():
                        for i_m, metric in enumerate(metrics):
                            results[i_s, i_m, i_e] = metrics[metric]
                else:
                    print(f"No such file: {experiment}metrics.json")

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

    def rank_regression(
        self,
        avg_names: list[str] = ["seed", "problem", "d"],
        settings: Dict = {"bo": False},
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

    def format_p_val(self, pval: float) -> str:
        pval_str = (
            "<10^{-10}" if pval < 1e-10 else rf"={pval:.2E}".replace("E", "10^{") + "}"
        )
        return pval_str

    def correlation_table(self, save: bool = True) -> None:
        surrogates = [
            x for x in self.loader_summary["surrogate"]["vals"] if not x == "RS"
        ]
        rows = ["Metric", "Rank"]
        table = pd.DataFrame(columns=surrogates, index=rows)
        ranking_dir = "metric"
        ranking_vals = ["regret", "y_calibration_mse"]
        for surrogate in surrogates:
            rho, pval = self.metric_correlation(
                {"surrogate": surrogate},
                ranking_dir=ranking_dir,
                ranking_vals=ranking_vals,
            )
            table[surrogate][rows[0]] = rf"{rho:.2f} ($p{self.format_p_val(pval)}$)"

            rho, pval = self.rank_correlation(
                {"surrogate": surrogate},
                ranking_dir=ranking_dir,
                ranking_vals=ranking_vals,
            )
            table[surrogate][rows[1]] = rf"{rho:.2f} ($p{self.format_p_val(pval)}$)"
        if save:
            self.save_to_tex(table.transpose(), name="correlation-table")

    def rank_correlation(
        self, settings: Dict = {}, ranking_dir: str = "", ranking_vals: list = []
    ) -> Dict[float, float]:
        if "bo" not in settings.keys():
            settings.update({"bo": True})
            add_str = "with"
        else:
            add_str = "with" if settings["bo"] else "no"
        if ranking_dir == "" or ranking_vals == []:
            ranking_dir = "metric"
            ranking_vals = ["regret", "y_calibration_mse"]
        assert len(ranking_vals) == 2
        rankings = np.load(f"{os.getcwd()}/results/rankings-{add_str}-bo.npy")

        settings.update({ranking_dir: ranking_vals[0]})
        x = self.extract(rankings, settings=settings).flatten().squeeze()

        settings.update({ranking_dir: ranking_vals[1]})
        y = self.extract(rankings, settings=settings).flatten().squeeze()
        x, y = self.remove_nans(x, y)
        return spearmanr(x, y)

    def metric_correlation(
        self, settings: Dict = {}, ranking_dir: str = "", ranking_vals: list = []
    ) -> Dict[float, float]:
        if "bo" not in settings.keys():
            settings.update({"bo": True})
        if ranking_dir == "" or ranking_vals == []:
            ranking_dir = "metric"
            ranking_vals = ["regret", "y_calibration_mse"]
        assert len(ranking_vals) == 2

        settings.update({ranking_dir: ranking_vals[0]})
        x = self.extract(settings=settings).flatten().squeeze()

        settings.update({ranking_dir: ranking_vals[1]})
        y = self.extract(settings=settings).flatten().squeeze()

        x, y = self.remove_nans(x.squeeze(), y.squeeze())
        x, y = self.remove_extremes(x.squeeze(), y.squeeze())
        return pearsonr(x, y)

    def expected_vs_actual_improv_correlation(self):
        return None
