from typing import Dict
from datasets.benchmarks.evalset.test_funcs import Rastrigin
from imports.general import *
from imports.ml import *
from figs.scripts.loader import Loader


class Tables(Loader):
    def __init__(self, scientific_notation: bool = False):
        super().__init__()
        self.scientific_notation = scientific_notation
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

    def table_linear_correlation_all(self):
        cnx = sqlite3.connect("./results.db")
        table_name = "results_regret_vs_calibration"
        query = self.dict2query(FROM=table_name, SELECT=["surrogate"])
        df = pd.read_sql(query, cnx)
        surrogates = sorted(df["surrogate"].unique())
        snr = 100

        row = ""
        for recal in [False]:
            x1 = pd.read_sql(
                self.dict2query(
                    FROM=table_name,
                    SELECT=[
                        "f_regret",
                        "y_calibration_mse",
                        "surrogate",
                        "mean_sharpness",
                        "elpd",
                    ],
                    WHERE={"bo": False, "recalibrate": recal, "snr": snr,},
                    ORDERBY=["seed", "d", "problem", "surrogate",],
                ),
                cnx,
            )
            x2 = pd.read_sql(
                self.dict2query(
                    FROM=table_name,
                    SELECT=[
                        "f_regret",
                        "y_calibration_mse",
                        "surrogate",
                        "mean_sharpness",
                        "elpd",
                    ],
                    WHERE={"bo": True, "recalibrate": recal, "snr": snr,},
                    ORDERBY=["seed", "d", "problem", "surrogate",],
                ),
                cnx,
            )

            x1 = x1.replace("NULL", np.nan, regex=True)
            x1 = x1.dropna(how="any", axis=0)
            sur_one_hot = pd.get_dummies(x1[["surrogate"]])
            x1.drop("surrogate", axis=1, inplace=True)
            for sur in surrogates:
                if sur not in ["RS", "DS"]:
                    x1[f"surrogate_{sur}"] = sur_one_hot[[f"surrogate_{sur}"]]

            x1 = x1.dropna(how="any", axis=0)
            y = x1[["f_regret"]]
            X = x1.drop("f_regret", axis=1)
            p_values, c025, c975 = self.fit_linearmodel(X, y)
            row = "All"
            for predictor in X.columns.values:
                c1 = self.format_num(c025[predictor])
                c2 = self.format_num(c975[predictor])
                row += f"&$({c1},{c2})" + self.p2stars(p_values[predictor], 15) + "$"
            print(row + "\\\\")

            x2 = x2.replace("NULL", np.nan, regex=True)
            x2 = x2.dropna(how="any", axis=0)
            sur_one_hot = pd.get_dummies(x2[["surrogate"]])
            x2.drop("surrogate", axis=1, inplace=True)
            for sur in surrogates:
                if sur not in ["RS", "DS"]:
                    x2[f"surrogate_{sur}"] = sur_one_hot[[f"surrogate_{sur}"]]

            x2 = x2.dropna(how="any", axis=0)
            y = x2[["f_regret"]]
            X = x2.drop("f_regret", axis=1)
            p_values, c025, c975 = self.fit_linearmodel(X, y)
            row = "All"
            for predictor in X.columns.values:
                c1 = self.format_num(c025[predictor])
                c2 = self.format_num(c975[predictor])
                row += f"&$({c1},{c2})" + self.p2stars(p_values[predictor], 15) + "$"
            print(row + "\\\\")
            raise ValueError()

            #################################################################################
            #################################################################################

            print(row + "\\\\")

    def table_linear_correlation(
        self, table_name: str = "results_regret_vs_calibration"
    ):
        cnx = sqlite3.connect("./results.db")
        groups = "surrogate"
        query = self.dict2query(FROM=table_name, SELECT=[groups])
        df = pd.read_sql(query, cnx)
        groups_ = sorted(df[groups].unique())
        snr = 100

        for recal in [False, True]:
            for group in groups_:
                row = f"{group}"

                #################################################################################
                #################################################################################

                data = pd.read_sql(
                    self.dict2query(
                        FROM=table_name,
                        SELECT=["f_regret"],
                        WHERE=self.merge_two_dicts(
                            {groups: group},
                            {"bo": True, "recalibrate": recal, "snr": snr},
                        ),
                    ),
                    cnx,
                )[["f_regret"]].to_numpy()

                row += (
                    "&$"
                    + self.format_num(np.nanmean(data))
                    + "\,\,(\pm "
                    + self.format_num(np.nanstd(data))
                    + ")$"
                )

                #################################################################################
                #################################################################################
                data = pd.read_sql(
                    self.dict2query(
                        FROM=table_name,
                        SELECT=["f_regret_total"],
                        WHERE=self.merge_two_dicts(
                            {groups: group},
                            {"bo": True, "recalibrate": recal, "snr": snr},
                        ),
                    ),
                    cnx,
                )[["f_regret_total"]].to_numpy()
                row += (
                    "&$"
                    + self.format_num(np.nanmean(data))
                    + "\,\,(\pm "
                    + self.format_num(np.nanstd(data))
                    + ")$"
                )

                if group in ["RS", "DS"]:
                    row += "&-&-"
                    continue

                #################################################################################
                #################################################################################
                data = pd.read_sql(
                    self.dict2query(
                        FROM=table_name,
                        SELECT=["y_calibration_mse"],
                        WHERE=self.merge_two_dicts(
                            {groups: group},
                            {"bo": False, "recalibrate": recal, "snr": snr},
                        ),
                    ),
                    cnx,
                )[["y_calibration_mse"]].to_numpy()
                row += (
                    "&$"
                    + self.format_num(np.nanmean(data))
                    + "\,\,(\pm "
                    + self.format_num(np.nanstd(data))
                    + ")$"
                )

                #################################################################################
                #################################################################################
                data = pd.read_sql(
                    self.dict2query(
                        FROM=table_name,
                        SELECT=["y_calibration_mse"],
                        WHERE=self.merge_two_dicts(
                            {groups: group},
                            {"bo": True, "recalibrate": recal, "snr": snr},
                        ),
                    ),
                    cnx,
                )[["y_calibration_mse"]].to_numpy()
                row += (
                    "&$"
                    + self.format_num(np.nanmean(data))
                    + "\,\,(\pm "
                    + self.format_num(np.nanstd(data))
                    + ")$"
                )

                #################################################################################
                #################################################################################
                x1 = (
                    pd.read_sql(
                        self.dict2query(
                            FROM=table_name,
                            SELECT=["y_calibration_mse"],
                            WHERE={
                                groups: group,
                                "bo": False,
                                "recalibrate": recal,
                                "snr": snr,
                            },
                            ORDERBY=["seed", "d", "problem",],
                        ),
                        cnx,
                    )
                    .to_numpy()
                    .squeeze()
                )
                x2 = (
                    pd.read_sql(
                        self.dict2query(
                            FROM=table_name,
                            SELECT=["y_calibration_mse"],
                            WHERE={
                                groups: group,
                                "bo": True,
                                "recalibrate": recal,
                                "snr": snr,
                            },
                            ORDERBY=["seed", "d", "problem",],
                        ),
                        cnx,
                    )
                    .to_numpy()
                    .squeeze()
                )

                y = (
                    pd.read_sql(
                        self.dict2query(
                            FROM=table_name,
                            SELECT=["f_regret"],
                            WHERE={
                                groups: group,
                                "bo": True,
                                "recalibrate": recal,
                                "snr": snr,
                            },
                            ORDERBY=["seed", "d", "problem",],
                        ),
                        cnx,
                    )
                    .to_numpy()
                    .squeeze()
                )
                rho, p_val = pearsonr(x1, y)
                row += "&$" + self.format_num(rho) + self.p2stars(p_val, 15) + "$"
                rho, p_val = pearsonr(x2, y)
                row += "&$" + self.format_num(rho) + self.p2stars(p_val, 15) + "$"
                #################################################################################
                #################################################################################

                print(row + "\\\\")

    def extract_pandasframes(
        self,
        loader: Loader,
        settings_X: Dict,
        settings_y: Dict,
        standardize: bool = True,
    ):
        predictors = list(settings_X["metric"])
        X, names = loader.extract(settings=settings_X, return_values=True)
        y = loader.extract(settings=settings_y)
        y = y.reshape(*[-1], y.shape[-1])
        X = X.reshape(*[-1], X.shape[-1])
        if standardize:
            X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
            y = (y - np.nanmean(y, axis=0)) / np.nanstd(y, axis=0)
        variables = np.append(names, settings_y["metric"])
        X_y = np.append(X, y, axis=-1)
        data = pd.DataFrame(X_y, columns=variables)
        data = data.dropna()
        X = data[predictors]
        y = data[settings_y["metric"]]
        predictors.append("const")
        return X, y, predictors

    def fit_linearmodel(
        self, X: np.ndarray, y: np.ndarray, return_ci: bool = True, ci: bool = 0.05,
    ):
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)
        X = sm.add_constant(X)
        mod = sm.OLS(y, X)
        res = mod.fit()
        # print(res.summary())
        confidence_intervals = res.conf_int(ci)
        p_values = res.pvalues.to_dict()
        if return_ci:
            c025 = confidence_intervals[0].to_dict()
            c975 = confidence_intervals[1].to_dict()
            return p_values, c025, c975
        else:
            coeffs = res.params.to_dict()
            return p_values, coeffs

    def table_linear_model_dims(
        self,
        target: str = "f_regret",
        predictors: list = ["y_calibration_mse", "mean_sharpness", "elpd",],
        X_bo: bool = False,
        y_bo: bool = True,
    ):
        loader = self.loader
        predictors_ = list(predictors)

        bonferoni_correction = (
            loader.loader_summary["surrogate"]["d"] + loader.loader_summary["d"]["d"]
        )

        # Surrogates, dimensions
        for d in [1, 5, 10]:
            for sur in ["BNN", "DE", "GP", "RF"]:
                settings_X = {
                    "bo": X_bo,
                    "d": d,
                    "surrogate": sur,
                    "metric": predictors_,
                }
                settings_y = {"bo": y_bo, "d": d, "surrogate": sur, "metric": target}
                X, y, predictors = self.extract_pandasframes(
                    loader, settings_X, settings_y
                )
                p_values, c025, c975 = self.fit_linearmodel(X, y)
                row = f"{sur}({d})"
                for predictor in predictors:
                    c1 = self.format_num(c025[predictor])
                    c2 = self.format_num(c975[predictor])
                    row += (
                        f"&$({c1},{c2})"
                        + self.p2stars(p_values[predictor], bonferoni_correction)
                        + "$"
                    )
                print(row + "\\\\")
            print("\\hline")

        # Surrogates
        for sur in ["BNN", "DE", "GP", "RF"]:
            settings_X = {"bo": X_bo, "surrogate": sur, "metric": predictors_}
            settings_y = {"bo": y_bo, "surrogate": sur, "metric": target}
            X, y, predictors = self.extract_pandasframes(loader, settings_X, settings_y)
            p_values, c025, c975 = self.fit_linearmodel(X, y)
            row = f"{sur}"
            for predictor in predictors:
                c1 = self.format_num(c025[predictor])
                c2 = self.format_num(c975[predictor])
                row += (
                    f"&$({c1},{c2})"
                    + self.p2stars(p_values[predictor], bonferoni_correction)
                    + "$"
                )
            print(row + "\\\\")
        print("\\hline")

        # All
        settings_X = {"bo": X_bo, "metric": predictors_}
        settings_y = {"bo": y_bo, "metric": target}
        X, y, predictors = self.extract_pandasframes(loader, settings_X, settings_y)
        p_values, c025, c975 = self.fit_linearmodel(X, y)
        row = "All"
        for predictor in predictors:
            c1 = self.format_num(c025[predictor])
            c2 = self.format_num(c975[predictor])
            row += (
                f"&$({c1},{c2})"
                + self.p2stars(p_values[predictor], bonferoni_correction)
                + "$"
            )
        print(row + "\\\\")

    def table_linear_model(
        self,
        target: str = "f_regret",
        predictors: list = ["y_calibration_mse", "mean_sharpness", "elpd",],
        X_bo: bool = False,
        y_bo: bool = True,
    ):
        loader = self.loader
        predictors_ = list(predictors)

        bonferoni_correction = (
            loader.loader_summary["surrogate"]["d"] + loader.loader_summary["d"]["d"]
        )
        # Surrogates
        for sur in ["BNN", "DE", "GP", "RF"]:
            settings_X = {"bo": X_bo, "surrogate": sur, "metric": predictors_}
            settings_y = {"bo": y_bo, "surrogate": sur, "metric": target}
            X, y, predictors = self.extract_pandasframes(loader, settings_X, settings_y)
            p_values, c025, c975 = self.fit_linearmodel(X, y)
            row = f"{sur}"
            for predictor in predictors:
                c1 = self.format_num(c025[predictor])
                c2 = self.format_num(c975[predictor])
                row += (
                    f"&$({c1},{c2})"
                    + self.p2stars(p_values[predictor], bonferoni_correction)
                    + "$"
                )
            print(row + "\\\\")
        print("\\hline")

        # All
        settings_X = {"bo": X_bo, "metric": predictors_}
        settings_y = {"bo": y_bo, "metric": target}
        X, y, predictors = self.extract_pandasframes(loader, settings_X, settings_y)
        p_values, c025, c975 = self.fit_linearmodel(X, y)
        row = "All"
        for predictor in predictors:
            c1 = self.format_num(c025[predictor])
            c2 = self.format_num(c975[predictor])
            row += (
                f"&$({c1},{c2})"
                + self.p2stars(p_values[predictor], bonferoni_correction)
                + "$"
            )
        print(row + "\\\\")

    def table_real_data(self, table_name: str = "results_mnist"):
        cnx = sqlite3.connect("./results.db")
        groups = "surrogate"
        query = self.dict2query(FROM=table_name, SELECT=[groups])
        df = pd.read_sql(query, cnx)
        groups_ = sorted(df[groups].unique())
        for recal in [False, True]:
            for group in groups_:

                #################################################################################
                #################################################################################
                subfilt = (
                    {"bo": True, "recalibrate": recal, "epoch": 90}
                    if group not in ["RS", "DS"]
                    else {"bo": False, "recalibrate": False, "epoch": 90}
                )
                row = f"&{group}&"
                data = pd.read_sql(
                    self.dict2query(
                        FROM=table_name,
                        SELECT=[
                            "y_regret",
                            "y_regret_total",
                            "y_calibration_mse",
                            "mean_sharpness",
                        ],
                        WHERE=self.merge_two_dicts({groups: group}, subfilt,),
                    ),
                    cnx,
                )

                y_regret = data[["y_regret"]].to_numpy().astype(float)
                y_regret_total = data[["y_regret_total"]].to_numpy().astype(float)
                y_r_mu, y_rt_mu = np.mean(y_regret), np.mean(y_regret_total)
                if group not in ["RS", "DS"]:
                    y_calibration_mse = (
                        data[["y_calibration_mse"]].to_numpy().astype(float)
                    )
                    sharpness = data[["mean_sharpness"]].to_numpy().astype(float)
                    y_c_mu, y_s_mu = np.mean(y_calibration_mse), np.mean(sharpness)

                    rho, p_val = pearsonr(
                        y_regret.squeeze(), y_calibration_mse.squeeze()
                    )
                    # print(np.round(rho,2),np.round(p_val,2))
                    rho, p_val = pearsonr(
                        y_regret_total.squeeze(), y_calibration_mse.squeeze()
                    )
                    # print(np.round(rho,2),np.round(p_val,2))
                else:
                    y_c_mu, y_s_mu = "-", "-"
                row += f"${self.format_num(y_r_mu)}$ & ${self.format_num(y_rt_mu)}$ & ${self.format_num(y_c_mu)}$ & ${self.format_num(y_s_mu)}$"

                print(row + "\\\\")

    # def table_linear_correlation(
    #     self,
    #     settings: Dict = {"data_name": "benchmark", "epoch": 90, "snr": 100},
    #     surrogates: list = None,
    # ):
    #     loader = self.loader
    #     surrogates = (
    #         loader.loader_summary["surrogate"]["vals"]
    #         if surrogates is None
    #         else surrogates
    #     )
    #     bonferoni_correction = (
    #         loader.loader_summary["surrogate"]["d"] + loader.loader_summary["d"]["d"]
    #     )
    #     # Across dimensions
    #     for sur in surrogates:
    #         r_f = loader.extract(
    #             settings=self.merge_two_dicts(
    #                 settings, {"bo": True, "surrogate": sur, "metric": "f_regret",},
    #             )
    #         )
    #         c_r = loader.extract(
    #             settings=self.merge_two_dicts(
    #                 settings,
    #                 {"bo": False, "surrogate": sur, "metric": "y_calibration_mse",},
    #             )
    #         )
    #         c_bo = loader.extract(
    #             settings=self.merge_two_dicts(
    #                 settings,
    #                 {"bo": True, "surrogate": sur, "metric": "y_calibration_mse",},
    #             )
    #         )
    #         row = (
    #             f"{sur}&$"
    #             + self.format_num(np.nanmean(r_f))
    #             + "\,\,(\pm "
    #             + self.format_num(np.nanstd(r_f))
    #             + ")$"
    #         )

    #         if sur not in ["RS", "DS"]:
    #             x, y = loader.remove_nans(r_f.flatten(), c_r.flatten())
    #             rho_reg, p_reg = pearsonr(x, y)
    #             x, y = loader.remove_nans(r_f.flatten(), c_bo.flatten())
    #             rho_bo, p_bo = pearsonr(x, y)
    #             row += (
    #                 "&$"
    #                 + self.format_num(np.nanmean(c_r))
    #                 + "\,\,(\pm "
    #                 + self.format_num(np.nanstd(c_r))
    #                 + ")$"
    #             )
    #             row += (
    #                 "&$"
    #                 + self.format_num(np.nanmean(c_bo))
    #                 + "\,\,(\pm "
    #                 + self.format_num(np.nanstd(c_bo))
    #                 + ")$"
    #             )
    #             # regression rho
    #             row += (
    #                 "&$"
    #                 + self.format_num(rho_reg)
    #                 + self.p2stars(p_reg, bonferoni_correction)
    #                 + "$"
    #             )

    #             # bo rho
    #             row += (
    #                 "&$"
    #                 + self.format_num(rho_bo)
    #                 + self.p2stars(p_bo, bonferoni_correction)
    #                 + "$"
    #             )
    #         else:
    #             row += "&-&-&-&-"

    #         print(row + "\\\\")
    #     print("\\hline")

    #     row = "All&&&"
    #     r_f = loader.extract(
    #         settings=self.merge_two_dicts(settings, {"bo": True, "metric": "f_regret"})
    #     )
    #     c_r = loader.extract(
    #         settings=self.merge_two_dicts(
    #             settings, {"bo": False, "metric": "y_calibration_mse",}
    #         )
    #     )
    #     c_bo = loader.extract(
    #         settings=self.merge_two_dicts(
    #             settings, {"bo": True, "metric": "y_calibration_mse",}
    #         )
    #     )
    #     x, y = loader.remove_nans(r_f.flatten(), c_r.flatten())
    #     rho_reg, p_reg = pearsonr(x, y)
    #     x, y = loader.remove_nans(r_f.flatten(), c_bo.flatten())
    #     rho_bo, p_bo = pearsonr(x, y)
    #     # regression rho
    #     row += (
    #         "&$"
    #         + self.format_num(rho_reg)
    #         + self.p2stars(p_reg, bonferoni_correction=bonferoni_correction)
    #         + "$"
    #     )
    #     # bo rho
    #     row += (
    #         "&$"
    #         + self.format_num(rho_bo)
    #         + self.p2stars(p_bo, bonferoni_correction=bonferoni_correction)
    #         + "$"
    #     )
    #     print(row + "\\\\")

    # def table_linear_correlation_dims(
    #     self,
    #     settings: Dict = {"data_name": "benchmark", "epoch": 90, "snr": 100},
    #     ds: list = None,
    #     surrogates: list = None,
    # ):
    #     loader = self.loader
    #     surrogates = (
    #         loader.loader_summary["surrogate"]["vals"]
    #         if surrogates is None
    #         else surrogates
    #     )
    #     ds = loader.loader_summary["d"]["vals"] if surrogates is None else surrogates

    #     bonferoni_correction = (
    #         loader.loader_summary["surrogate"]["d"] + loader.loader_summary["d"]["d"]
    #     )

    #     for d in ds:
    #         for sur in surrogates:
    #             r_f = loader.extract(
    #                 settings=self.merge_two_dicts(
    #                     settings,
    #                     {"bo": True, "surrogate": sur, "d": d, "metric": "f_regret",},
    #                 )
    #             )
    #             c_r = loader.extract(
    #                 settings=self.merge_two_dicts(
    #                     settings,
    #                     {
    #                         "bo": False,
    #                         "surrogate": sur,
    #                         "d": d,
    #                         "metric": "y_calibration_mse",
    #                     },
    #                 )
    #             )
    #             c_bo = loader.extract(
    #                 settings=self.merge_two_dicts(
    #                     settings,
    #                     {
    #                         "bo": True,
    #                         "surrogate": sur,
    #                         "d": d,
    #                         "metric": "y_calibration_mse",
    #                     },
    #                 )
    #             )
    #             row = (
    #                 f"{sur}({d})&$"
    #                 + self.format_num(np.nanmean(r_f))
    #                 + "\,\,(\pm "
    #                 + self.format_num(np.nanstd(r_f))
    #                 + ")$"
    #             )

    #             if sur not in ["RS", "DS"]:
    #                 x, y = loader.remove_nans(r_f.flatten(), c_r.flatten())
    #                 rho_reg, p_reg = pearsonr(x, y)
    #                 x, y = loader.remove_nans(r_f.flatten(), c_bo.flatten())
    #                 rho_bo, p_bo = pearsonr(x, y)
    #                 row += (
    #                     "&$"
    #                     + self.format_num(np.nanmean(c_r))
    #                     + "\,\,(\pm "
    #                     + self.format_num(np.nanstd(c_r))
    #                     + ")$"
    #                 )
    #                 row += (
    #                     "&$"
    #                     + self.format_num(np.nanmean(c_bo))
    #                     + "\,\,(\pm "
    #                     + self.format_num(np.nanstd(c_bo))
    #                     + ")$"
    #                 )
    #                 # regression rho
    #                 row += (
    #                     "&$"
    #                     + self.format_num(rho_reg)
    #                     + self.p2stars(p_reg, bonferoni_correction)
    #                     + "$"
    #                 )

    #                 # bo rho
    #                 row += (
    #                     "&$"
    #                     + self.format_num(rho_bo)
    #                     + self.p2stars(p_bo, bonferoni_correction)
    #                     + "$"
    #                 )
    #             else:
    #                 row += "&-&-&-&-"

    #             print(row + "\\\\")
    #         print("\\hline")

    #     self.table_linear_correlation(settings=settings, surrogates=surrogates)
