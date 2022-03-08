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
        self.savepth = (
            os.getcwd()
            + "/visualizations/figures/"
            + str.join("-", [f"{key}-{val}-" for key, val in settings.items()])
        )

    def find_extreme_idx(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        extreme_high_idx = x > np.quantile(x, 0.95)
        extreme_low_idx = x < np.quantile(x, 0.05)
        return extreme_high_idx, extreme_low_idx

    def find_extreme_vals(self, x: np.ndarray, q: float = 0.05) -> Tuple[float, float]:
        return np.quantile(x, 1 - q), np.quantile(x, q)

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
            fig.savefig(f"{self.savepth}calibration---{settings}.pdf")
            plt.close()

    def metrics_vs_epochs(
        self,
        avg_names: list[str] = ["seed"],
        only_surrogates: list[str] = [],  # "GP", "BNN", "DS"
    ):
        epochs = self.loader_summary["epoch"]["vals"]
        avg_dims = tuple([self.loader_summary[name]["axis"] for name in avg_names])
        n_avgs = np.prod([len(self.loader_summary[name]["vals"]) for name in avg_names])
        for problem in self.loader_summary["problem"]["vals"]:
            for d in self.loader_summary["d"]["vals"]:
                fig = plt.figure()
                for i_m, metric in enumerate(self.loader_summary["metric"]["vals"]):
                    ax = plt.subplot(
                        len(self.loader_summary["metric"]["vals"]), 1, i_m + 1
                    )
                    for i_s, surrogate in enumerate(
                        self.loader_summary["surrogate"]["vals"]
                    ):
                        if (
                            len(only_surrogates) > 0
                            and surrogate not in only_surrogates
                        ):
                            continue

                        data = self.extract(
                            settings={
                                "problem": problem,
                                "d": d,
                                "surrogate": surrogate,
                                "metric": metric,
                                "bo": True,
                            }
                        ).squeeze()
                        if (np.sum(np.isnan(data)) / data.size) > 0.5:
                            continue
                        if len(avg_names) > 0:
                            means = np.nanmean(data, axis=avg_dims)[:, np.newaxis].T
                            stds = np.nanstd(data, axis=avg_dims)[:, np.newaxis].T
                        else:
                            means = data
                            stds = np.zeros(means.shape)

                        for i_mean, mean in enumerate(means):
                            if i_mean == 0:
                                plt.plot(
                                    epochs,
                                    mean,
                                    color=ps[surrogate]["c"],
                                    marker=ps[surrogate]["m"],
                                    label=f"${surrogate}$",
                                )
                            else:
                                plt.plot(
                                    epochs,
                                    mean,
                                    color=ps[surrogate]["c"],
                                    marker=ps[surrogate]["m"],
                                )
                            plt.fill_between(
                                epochs,
                                mean + 1 * stds[i_mean, :] / np.sqrt(n_avgs),
                                mean - 1 * stds[i_mean, :] / np.sqrt(n_avgs),
                                color=ps[surrogate]["c"],
                                alpha=0.1,
                            )

                    if (np.sum(np.isnan(data)) / data.size) > 0.5:
                        continue
                    if i_m < len(self.loader_summary["metric"]["vals"]) - 1:
                        ax.set_xticklabels([])
                    plt.ylabel(self.metric_dict[metric][-1])
                    if len(self.metric_dict[metric][-2]) == 2:
                        plt.ylim(
                            [
                                self.metric_dict[metric][-2][0],
                                self.metric_dict[metric][-2][1],
                            ]
                        )
                    plt.xlim([epochs[0] - 0.1, epochs[-1] + 0.1])
                if (np.sum(np.isnan(data)) / data.size) > 0.5:
                    plt.close()
                    continue
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    ncol=len(self.loader_summary["surrogate"]["vals"]),
                )
                plt.xlabel("Epochs")
                plt.tight_layout()
                fig.savefig(f"{self.savepth}metrics-vs-epochs---{problem}({d}).pdf")
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
                    f"{self.savepth}bo-iters--{parameters.problem}-{parameters.surrogate}"
                    + f"--n-epochs-{n_epochs}--seed-{seed}--d-{parameters.d}.pdf"
                )
                plt.close()

    def bo_regret_vs_no_bo_calibration(
        self, epoch: int = 89, avg_names: list[str] = []
    ):
        avg_dims = [self.loader_summary[name]["axis"] for name in avg_names]
        for problem in self.loader_summary["problem"]["vals"]:
            for dim in self.loader_summary["d"]["vals"]:
                fig = plt.figure()
                skip = False
                for surrogate in self.loader_summary["surrogate"]["vals"]:
                    data = self.extract(
                        settings={
                            "metric": "regret",
                            "bo": True,
                            "surrogate": surrogate,
                            "problem": problem,
                            "d": dim,
                        },
                    )
                    x = (
                        np.nanmean(data, axis=tuple(avg_dims)).squeeze()
                        if len(avg_dims) >= 0
                        else data.flatten()
                    )

                    data = self.extract(
                        settings={
                            "metric": "y_calibration_mse",
                            "bo": False,
                            "surrogate": surrogate,
                            "problem": problem,
                            "d": dim,
                        },
                    )
                    y = (
                        np.nanmean(data, axis=tuple(avg_dims)).squeeze()
                        if len(avg_dims) >= 0
                        else data.flatten()
                    )
                    # if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                    #     skip = True
                    #     continue

                    plt.scatter(
                        x,
                        y,
                        color=ps[surrogate]["c"],
                        marker=ps[surrogate]["m"],
                        label=f"{surrogate}",
                    )
                if skip:
                    continue
                plt.xlabel("Regret")
                plt.ylabel("Calibration MSE")
                # plt.xscale("log")
                plt.legend()
                fig.savefig(
                    f"{self.savepth}regret-vs-no-bo-calibration|{problem}({dim})|epoch={epoch}|avg={avg_names}.pdf"
                )

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
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.legend()
        fig.savefig(f"{self.savepth}exp-improv-vs-act-improv.pdf")

