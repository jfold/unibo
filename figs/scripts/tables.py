from typing import Dict
from imports.general import *
from imports.ml import *
from figs.scripts.loader import Loader


class Tables(Loader):
    def __init__(self, loader: Loader):
        self.loader = loader
        self.savepth = os.getcwd() + "/figs/tables/"

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

    def to_scientific(self, x):
        if np.abs(x) < 10 ** (-2):
            return (
                f"{x:.1E}".replace("E-0", "E-")
                .replace("E+0", "E+")
                .replace("E-", "\cdot 10^{-")
                .replace("E+", "10^{")
                + "}"
            )
        else:
            return f"{x:.2f}"

    def table_linear_correlation(
        self,
        settings: Dict = {"data_name": "benchmark"},
        ds: list = [1, 5, 10],
        surrogates: list = ["GP", "RF", "BNN", "DE", "RS"],
        metrics=["f_regret", "y_calibration_mse"],
    ):
        settings_ = dict(settings)
        for d in ds:
            plt.figure()
            for surrogate in surrogates:
                # Instant regret
                settings = dict(settings_)
                settings.update(
                    {
                        "d": d,
                        "bo": True,
                        "surrogate": surrogate,
                        "metric": metrics[0],
                        "epoch": 90,
                    }
                )
                instant_regret = self.loader.extract(settings=settings).flatten()

                # Accumulated regret
                # settings = dict(settings_)
                # settings.update(
                #     {"d": d, "bo": True, "surrogate": surrogate, "metric": metrics[0]}
                # )
                # regret = self.loader.extract(settings=settings)
                # regret = np.nancumsum(regret, axis=self.loader.names.index("epoch"))

                if surrogate not in ["RS", "DS"]:
                    # Calibration (regression)
                    settings = dict(settings_)
                    settings.update(
                        {
                            "d": d,
                            "bo": False,
                            "surrogate": surrogate,
                            "metric": metrics[1],
                            "epoch": 90,
                        }
                    )
                    calibration = self.loader.extract(settings=settings).flatten()

                    # Calibration (BO)
                    settings = dict(settings_)
                    settings.update(
                        {
                            "d": d,
                            "bo": True,
                            "surrogate": surrogate,
                            "metric": metrics[1],
                            "epoch": 90,
                        }
                    )
                    calibration_bo = self.loader.extract(settings=settings).flatten()

                    # Correlation test (regression)
                    instant_regret_, calibration_ = self.loader.remove_nans(
                        instant_regret, calibration
                    )
                    rho_reg, p_reg = pearsonr(instant_regret_, calibration_)
                    plt.plot(instant_regret_, calibration_, ".", label=surrogate)

                    # Correlation test (BO)
                    instant_regret_, calibration_ = self.loader.remove_nans(
                        instant_regret, calibration_bo
                    )
                    rho_bo, p_bo = pearsonr(instant_regret_, calibration_)

                # Make table
                row = (
                    f"{surrogate}({d})&$"
                    + self.to_scientific(np.mean(instant_regret_))
                    + " (\pm "
                    + self.to_scientific(np.std(instant_regret_))
                    + ")$"
                )
                if surrogate not in ["RS", "DS"]:
                    # row += "&"  # + self.to_scientific(np.nanmean(regret)) + "(\pm "+ self.to_scientific(np.std(regret)) + ")"
                    row += (
                        "&$"
                        + self.to_scientific(np.mean(calibration_))
                        + " (\pm "
                        + self.to_scientific(np.std(calibration_))
                        + ")$"
                    )
                    row += (
                        "&$"
                        + self.to_scientific(np.nanmean(calibration_bo))
                        + " (\pm "
                        + self.to_scientific(np.nanstd(calibration_bo))
                        + ")$"
                    )
                    # regression rho
                    row += "&$" + self.to_scientific(rho_reg)
                    stars = ""
                    if p_reg < 0.05:
                        stars = "^{*}"
                    if p_reg < 0.001:
                        stars = "^{**}"
                    if p_reg < 0.0001:
                        stars = "^{***}"
                    row += stars + "$"
                    # bo rho
                    row += "&$" + self.to_scientific(rho_bo)
                    stars = ""
                    if p_bo < 0.05:
                        stars = "^{*}"
                    if p_bo < 0.001:
                        stars = "^{**}"
                    if p_bo < 0.0001:
                        stars = "^{***}"
                    row += stars + "$"
                else:
                    row += "&-&-&-&-"
                print(row + "\\\\")
            print("\\hline")
            plt.xscale("log")
            plt.legend()
