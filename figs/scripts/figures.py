from sklearn.utils import deprecated
from imports.general import *
from imports.ml import *
from src.dataset import Dataset
from src.parameters import Parameters
from figs.scripts.loader import Loader


class Figures(Loader):
    def __init__(self, scientific_notation: bool = False):
        super().__init__()
        self.scientific_notation = scientific_notation
        self.savepth = os.getcwd() + "/figs/pdfs/"
        self.markers = ["o", "v", "s", "x", "d"]
        self.colors = plt.cm.plasma(np.linspace(0, 1, len(self.markers)))

    def calibration(self):
        fig = plt.figure()
        if hasattr(self, "p_axis"):
            plt.plot(
                self.p_axis,
                self.p_axis,
                "--",
                color="blue",
                marker="_",
                label="Perfectly calibrated",
            )
            for i, calibrations in enumerate(self.calibrations):
                if calibrations.shape[0] < 2:
                    continue

                mean_calibration = np.nanmean(calibrations, axis=0)
                std_calibration = np.nanstd(calibrations, axis=0) / np.sqrt(
                    calibrations.shape[0]
                )
                p = plt.plot(
                    self.p_axis,
                    mean_calibration,
                    label=r"$\mathcal{" + self.names[i] + "}$ ",
                )
                col = p[0].get_color()
                mak = p[0].get_marker()
                eb = plt.errorbar(
                    self.p_axis,
                    mean_calibration,
                    yerr=std_calibration,
                    color=col,
                    marker=mak,
                    capsize=4,
                    alpha=0.5,
                )
                eb[-1][0].set_linestyle("--")
            plt.legend()
            plt.xlabel(r"$p$")
            plt.ylabel(r"$\mathcal{C}_{\mathbf{y}}$")
            plt.tight_layout()
            settings = str.join(
                "--", [str(key) + "-" + str(val) for key, val in self.settings.items()]
            ).replace(".", "-")
            settings = "default" if settings == "" else settings
            fig.savefig(f"{self.savepth_figs}calibration---{settings}.pdf")
            plt.close()

    def metrics_vs_epochs(
        self,
        avg_names: list[str] = ["seed", "problem", "d", "acquisition"],
        settings: Dict = {"bo": True, "change_std": True},
        metrics: list[str] = [
            # "nmse",
            # "elpd",
            # "mean_sharpness",
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
        data = self.extract(settings=settings)

        avg_dims = tuple([self.loader_summary[name]["axis"] for name in avg_names])
        sem_multi = 1.96 / np.sqrt(10 * 5 * 9)
        epochs = self.loader_summary["epoch"]["vals"]
        surrogates = self.loader_summary["surrogate"]["vals"]

        fig = plt.figure()
        for i_m, metric in enumerate(metrics):
            ax = plt.subplot(len(metrics), 1, i_m + 1)
            for surrogate in surrogates:
                data_ = self.extract(
                    data, settings={"surrogate": surrogate, "metric": metric}
                )
                means = np.nanmean(data_, axis=avg_dims).squeeze()
                stds = np.nanstd(data_, axis=avg_dims).squeeze()
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

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc=(0.1, -0.01),
            ncol=len(surrogates),
            fancybox=True,
            shadow=True,
        )
        plt.xlabel("Iterations")
        plt.tight_layout()
        plt.show()
        fig.savefig(
            f"{self.savepth_figs}metrics-vs-epochs---{settings}---{save_str}.pdf"
        )
        plt.close()

    def bo_2d_contour(self, n_epochs: int = 10, seed: int = 1):
        for i_e, experiment in enumerate(p for p in self.loadpths):
            if (
                os.path.isdir(experiment)
                and os.path.isfile(experiment + "parameters.json")
                and os.path.isfile(experiment + "dataset.json")
            ):
                with open(experiment + "parameters.json") as json_file:
                    parameters = json.load(json_file)

                if (
                    not self.settings.items() <= parameters.items()
                    or seed != parameters["seed"]
                    or not parameters["bo"]
                    or not parameters["d"] == 2
                ):
                    continue

                with open(experiment + "dataset.json") as json_file:
                    dataset = json.load(json_file)

                parameters["noisify"] = False
                parameters = Parameters(parameters)
                module = importlib.import_module(parameters.data_location)
                data_object = getattr(module, parameters.data_object)
                data = data_object(parameters)
                x_min_loc = data.problem.min_loc
                x_1 = np.linspace(dataset["x_lbs"][0], dataset["x_ubs"][0], 100)
                x_2 = np.linspace(dataset["x_lbs"][1], dataset["x_ubs"][1], 100)
                X1, X2 = np.meshgrid(x_1, x_2)
                y = np.full((100, 100), np.nan)
                for i1, x1 in enumerate(x_1):
                    for i2, x2 in enumerate(x_2):
                        y[i1, i2] = data.get_y(np.array([[x1, x2]])).squeeze()

                fig = plt.figure()
                ax = fig.add_subplot(111)
                pc = ax.pcolormesh(X1, X2, y.T, linewidth=0, rasterized=True)
                ax.grid(False)
                plt.plot(
                    x_min_loc[0],
                    x_min_loc[1],
                    color="green",
                    marker="o",
                    markersize=10,
                )
                fig.colorbar(pc)

                X = np.array(dataset["X"])
                n_initial = parameters.n_initial
                x_1_init = X[:n_initial, 0]
                x_2_init = X[:n_initial, 1]
                plt.scatter(
                    x_1_init, x_2_init, marker=".", color="black",
                )
                x_1_bo = X[n_initial : n_initial + n_epochs, 0]
                x_2_bo = X[n_initial : n_initial + n_epochs, 1]
                for i in range(len(x_1_bo)):
                    plt.text(x_1_bo[i], x_2_bo[i], str(i + 1))

                plt.xlabel(r"$x_1$")
                plt.ylabel(r"$x_2$")

                fig.savefig(
                    f"{self.savepth_figs}bo-iters--{parameters.problem}-{parameters.surrogate}"
                    + f"--n-epochs-{n_epochs}--seed-{seed}--d-{parameters.d}.pdf"
                )
                plt.close()

    def exp_improv_vs_act_improv(self):
        expected_improvements = {
            x: [] for x in self.loader_summary["surrogate"]["vals"] if not x == "RS"
        }
        actual_improvements = {x: [] for x in self.loader_summary["surrogate"]["vals"]}
        for pth, data in self.data_settings.items():
            with open(f"{pth}parameters.json") as json_file:
                parameters = json.load(json_file)
            surrogate = parameters["surrogate"]
            if surrogate == "RS" or not parameters["bo"]:
                continue
            files_in_path = [
                f for f in os.listdir(pth) if "metrics---epoch" in f and ".json" in f
            ]
            for file in files_in_path:
                with open(f"{pth}{file}") as json_file:
                    scores_epoch_i = json.load(json_file)

                lst = expected_improvements[surrogate]
                lst.append(
                    np.array(scores_epoch_i["expected_improvement"])
                    .squeeze()
                    .astype(float)
                )
                expected_improvements[surrogate] = lst

                lst = actual_improvements[surrogate]
                lst.append(
                    np.array(scores_epoch_i["actual_improvement"])
                    .squeeze()
                    .astype(float)
                )
                actual_improvements[surrogate] = lst
        fig = plt.figure()
        for surrogate in expected_improvements.keys():
            x = np.array(expected_improvements[surrogate]).squeeze()
            y = np.array(actual_improvements[surrogate]).squeeze()
            x, y = self.remove_nans(x, y)
            rho, pval = pearsonr(x, y)
            plt.scatter(
                x,
                y,
                label=rf"{surrogate} ($\rho = ${rho:.2E}, $p=${pval:.2E})",
                color=ps[surrogate]["c"],
                alpha=0.4,
            )

        plt.xlabel("Expected Improvement")
        plt.ylabel("Actual Improvement")
        plt.xlim([0, 1])
        plt.legend()
        fig.savefig(f"{self.savepth_figs}exp-improv-vs-act-improv.pdf")

    def plot_xy(self, dataset: Dataset):
        assert self.d == 1
        plt.figure()
        plt.plot(dataset.data.X_test, dataset.data.y_test, "*", label="Test", alpha=0.1)
        plt.plot(dataset.data.X_train, dataset.data.y_train, "*", label="Train")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.legend()
        plt.show()

    def plot_predictive(
        self,
        dataset: Dataset,
        X_test,
        y_test,
        mu,
        sigma_predictive,
        n_stds: float = 3.0,
        name: str = "",
    ):
        assert self.d == 1
        idx = np.argsort(X_test.squeeze())
        X_test = X_test[idx].squeeze()
        y_test = y_test[idx].squeeze()
        mu = mu[idx].squeeze()
        sigma_predictive = sigma_predictive[idx].squeeze()

        fig = plt.figure()
        plt.plot(X_test, y_test, "*", label="Test", alpha=0.2)
        plt.plot(dataset.data.X, dataset.data.y, "*", label="Train")
        plt.plot(
            X_test, mu, "--", color="black", label=r"$\mathcal{M}_{\mu}$", linewidth=1,
        )
        plt.fill_between(
            X_test,
            mu + n_stds * sigma_predictive,
            mu - n_stds * sigma_predictive,
            color="blue",
            alpha=0.1,
            label=r"$\mathcal{M}_{" + str(n_stds) + "\sigma}$",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        fig.savefig(self.savepth + f"predictive{name}.pdf")
        plt.show()

    def plot_mse_sigma(self, mu, y, sigma):
        plt.figure()
        abs_err = np.abs(mu - y)
        plt.plot(abs_err, sigma, "*")
        plt.xlim([0, np.max(abs_err)])
        plt.ylim([0, np.max(abs_err)])
        plt.xlabel(r"Absolute error")
        plt.ylabel(r"$\sigma$")
        plt.show()

    def plot_histogram(self, mu_test, y_test, sigma_test, n_bins=20, title=""):
        abs_err = np.abs(mu_test - y_test).squeeze()
        counts, limits = np.histogram(abs_err, n_bins)
        err_hist = []
        for i_l in range(len(limits)):
            vals = sigma_test[
                np.logical_and(limits[i_l - 1] < abs_err, abs_err < limits[i_l])
            ]
            err_hist.append(np.mean(vals))
        err_hist = np.array(err_hist)
        plt.figure()
        plt.plot(limits, limits, label=r"$x=y$", linewidth=0.4)
        plt.plot(limits, err_hist, "*")
        plt.xlabel("Mean absolute error (binned)")
        plt.ylabel(r"$\sigma$")
        plt.title(title)
        plt.legend()

    def plot_y_calibration(self, name: str = ""):
        # Target (y) calibration
        fig = plt.figure()
        plt.plot(
            self.summary["y_p_array"], self.summary["y_p_array"], "-", label="Optimal"
        )
        plt.plot(
            self.summary["y_p_array"],
            self.summary["y_calibration"],
            "*",
            label=r"$\mathcal{M}$ | MSE = "
            + "{:.2e}".format(self.summary["y_calibration_mse"]),
        )
        plt.xlabel(r"$p$")
        plt.legend()
        plt.ylabel(r"$\mathbb{E}[ \mathbb{I} \{ \mathbf{y} \leq F^{-1}(p) \} ]$")
        fig.savefig(self.savepth + f"calibration-y{name}.pdf")
        plt.close()

    def plot_f_calibration(self, name: str = ""):
        # Mean (f) calibration
        fig = plt.figure()
        plt.plot(self.summary["f_p_array"], self.summary["f_p_array"], "--")
        plt.plot(
            self.summary["f_p_array"],
            self.summary["f_calibration"],
            "*",
            label=r"$\mathcal{M}$ | MSE = "
            + "{:.2e}".format(self.summary["f_calibration_mse"]),
        )
        plt.legend()
        plt.xlabel(r"$p$")
        plt.ylabel(r"$\mathbb{E}[ \mathbb{I} \{ \mathbf{f} \leq F^{-1}(p) \} ]$")
        plt.show()
        fig.savefig(self.savepth + f"calibration-f{name}.pdf")
        plt.close()

    def plot_sharpness_histogram(
        self, sharpness: np.ndarray = None, n_bins: int = 50, name: str = ""
    ):
        if sharpness is None and "sharpness" in self.summary:
            fig = plt.figure()
            sharpness = self.summary["sharpness"]
        if sharpness is not None:
            plt.hist(
                sharpness,
                bins=n_bins,
                label=r"$\mathcal{M}$ | mean: "
                + "{:.2e}".format(self.summary["mean_sharpness"]),
                alpha=0.6,
            )

            if "hist_sharpness" in self.summary:
                plt.hist(
                    self.summary["hist_sharpness"],
                    bins=n_bins,
                    label=r"$\mathcal{M}$  hist | mean: "
                    + "{:.2e}".format(self.summary["mean_hist_sharpness"]),
                    alpha=0.6,
                )
            if self.ne_true is not np.nan:
                plt.annotate(
                    "True",
                    xy=(self.ne_true, 0),
                    xytext=(self.ne_true, -(self.n_test / 100)),
                    arrowprops=dict(facecolor="black", shrink=0.05),
                )
            plt.legend()
            plt.xlabel("Negative Entropy")
            plt.ylabel("Count")
            fig.savefig(self.savepth + f"sharpness-histogram{name}.pdf")
            plt.close()

    def figure_regret_calibration(
        self,
        settings: Dict = {"data_name": "benchmark", "epoch": 90, "snr": 100},
        settings_x: Dict = {"bo": True, "metric": "f_regret"},
        settings_y: Dict = {"bo": False, "metric": "y_calibration_mse"},
        x_figsettings: Dict = {"label": r"$\mathcal{R}_I(f)$", "log": True},
        y_figsettings: Dict = {"label": r"$\mathcal{C}_{R}(y)$", "log": True},
        surrogates: list = ["BNN", "DE", "GP", "RF"],
    ):
        colors = plt.cm.plasma(np.linspace(0, 1, len(self.markers)))
        loader = self.loader
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i_s, sur in enumerate(surrogates):
            x = loader.extract(
                settings=loader.merge_two_dicts(
                    loader.merge_two_dicts(settings, settings_x), {"surrogate": sur}
                )
            ).flatten()

            y = loader.extract(
                settings=loader.merge_two_dicts(
                    loader.merge_two_dicts(settings, settings_y), {"surrogate": sur}
                )
            ).flatten()
            ax.plot(x, y, self.markers[i_s], color=colors[i_s], label=sur)

        if x_figsettings["log"]:
            ax.set_xscale("log")
        if y_figsettings["log"]:
            ax.set_yscale("log")
        ax.legend()
        ax.set_xlabel(x_figsettings["label"])
        ax.set_ylabel(y_figsettings["label"])
        if settings_y["bo"]:
            fig.savefig("./figs/pdfs/r-c-bo.pdf")
        else:
            fig.savefig("./figs/pdfs/r-c.pdf")
        plt.show()

    def figure_std_vs_metric(
        self,
        x="std_change",
        y="y_calibration_mse",
        groups="surrogate",
        settings: Dict = {
            "data_name": "benchmark",
            "epoch": 90,
            "snr": 100,
            "bo": True,
        },
        table_name="results_change_std",
    ):
        cnx = sqlite3.connect("./results.db")
        query = self.dict2query(table_name=table_name, columns=[groups, x],)
        df = pd.read_sql(query, cnx)
        groups_ = sorted(df[groups].unique())
        xs = sorted(df[x].unique())

        fig = plt.figure()
        for i_g, group in enumerate(groups_):
            data_mu = []
            data_std = []
            for x_ in xs:
                query = self.dict2query(
                    self.merge_two_dicts(settings, {groups: group, x: x_}),
                    table_name,
                    [y],
                )
                data = pd.read_sql(query, cnx).to_numpy()
                data_mu.append(np.nanmean(data))
                data_std.append(np.nanstd(data) / np.sqrt(np.sum(np.isfinite(data))))
            plt.errorbar(
                xs,
                data_mu,
                yerr=data_std,
                marker=self.markers[i_g],
                color=self.colors[i_g],
                label=group,
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.xlabel(r"$c$")
        plt.ylabel(rf"{self.metric_dict[y][-1]}")
        plt.xscale("log")
        plt.yscale("log")
        if y == "y_calibration_mse":
            plt.ylim([1e-3, 0.15])
        if settings["bo"]:
            fig.savefig(f"./figs/pdfs/c-{y}-bo.pdf")
        else:
            fig.savefig(f"./figs/pdfs/c-{y}.pdf")
        plt.show()

    def scatter_regret_calibration_std_change(
        self,
        x="y_calibration_mse",
        y="f_regret",
        groups="surrogate",
        settings: Dict = {
            "data_name": "benchmark",
            "epoch": 90,
            "snr": 100,
            "bo": True,
        },
        table_name="results_change_std",
        average: bool = False,
    ):
        cnx = sqlite3.connect("./results.db")
        fig = plt.figure()

        if average:
            query = self.dict2query(FROM=table_name, SELECT=[groups, "std_change"],)
            df = pd.read_sql(query, cnx)
            groups_ = sorted(df[groups].unique())
            xs = sorted(df["std_change"].unique())

            data_x = []
            data_y = []
            data_z = []
            for i_g, group in enumerate(groups_):
                data_x_ = []
                data_y_ = []
                data_z_ = []
                for x_ in xs:
                    xaxis = self.dict2query(
                        FROM=table_name,
                        SELECT=[x],
                        WHERE=self.merge_two_dicts(
                            settings, {groups: group, "std_change": x_}
                        ),
                    )
                    yaxis = self.dict2query(
                        FROM=table_name,
                        SELECT=[y],
                        WHERE=self.merge_two_dicts(
                            settings, {groups: group, "std_change": x_}
                        ),
                    )
                    zaxis = self.dict2query(
                        FROM=table_name,
                        SELECT=["mean_sharpness"],
                        WHERE=self.merge_two_dicts(
                            settings, {groups: group, "std_change": x_}
                        ),
                    )
                    data_x_.append(np.nanmean(pd.read_sql(xaxis, cnx).to_numpy()))
                    data_y_.append(np.nanmean(pd.read_sql(yaxis, cnx).to_numpy()))
                    data_z_.append(np.nanmean(pd.read_sql(zaxis, cnx).to_numpy()))
                data_x.append(data_x_)
                data_y.append(data_y_)
                data_z.append(data_z_)
            z = np.array(data_z) + np.abs(np.min(data_z))
            data_z = np.exp(-10 * ((z) / np.max(z)) + 3)
            for i_g, group in enumerate(groups_):
                plt.scatter(
                    data_x[i_g],
                    data_y[i_g],
                    marker=self.markers[i_g],
                    s=(data_z[i_g] * 100) + 5,
                    color=self.colors[i_g],
                    label=group,
                    alpha=0.6,
                )
                # circle around c = 1
                plt.plot(
                    data_x[i_g][10],
                    data_y[i_g][10],
                    marker="o",
                    markersize=30,
                    color=self.colors[i_g],
                    alpha=0.1,
                )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlim([1e-2, 1.3e-1])

        else:
            query = self.dict2query(FROM=table_name, SELECT=[groups],)
            df = pd.read_sql(query, cnx)
            groups_ = sorted(df[groups].unique())

            for i_g, group in enumerate(groups_):
                query = self.dict2query(
                    WHERE=self.merge_two_dicts(settings, {groups: group}),
                    FROM=table_name,
                    SELECT=[x, y],
                )
                df = pd.read_sql(query, cnx)
                x_ = df[[x]].to_numpy() + 1e-10
                y_ = df[[y]].to_numpy() + 1e-10
                plt.scatter(
                    x_,
                    y_,
                    marker=self.markers[i_g],
                    color=self.colors[i_g],
                    label=group,
                    alpha=0.6,
                )
            plt.xscale("log")
            plt.yscale("log")

        # plt.legend()
        plt.xlabel(rf"{self.metric_dict[x][-1]}")
        plt.ylabel(rf"{self.metric_dict[y][-1]}")
        if settings["bo"]:
            fig.savefig(f"./figs/pdfs/change-std-cal-reg-bo.pdf")
        else:
            fig.savefig(f"./figs/pdfs/change-std-cal-reg.pdf")
        plt.show()
