from sklearn.utils import deprecated
from imports.general import *
from imports.ml import *
from src.parameters import Parameters
from visualizations.scripts.loader import Loader


class Figures(Loader):
    def __init__(
        self,
        loadpths: list[str] = [],
        settings: Dict[str, str] = {},
        update_data: bool = True,
    ):
        super(Figures, self).__init__(loadpths, settings, update=update_data)
        if not self.data_was_loaded:
            raise ValueError("No data could be loaded")

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
                data_class = getattr(module, parameters.data_class)
                data = data_class(parameters)
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
                f for f in os.listdir(pth) if "scores---epoch" in f and ".json" in f
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

