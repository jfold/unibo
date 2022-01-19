from typing import Dict

from scipy.stats.stats import energy_distance
from imports.general import *
from imports.ml import *
from src.parameters import Parameters
from visualizations.scripts.loader import Loader


class Figures(Loader):
    def __init__(self, loadpths: list[str] = [], settings: Dict[str, str] = {}):
        super(Figures, self).__init__(loadpths, settings)
        self.savepth = (
            os.getcwd()
            + "/visualizations/figures/"
            + str.join("-", [f"{key}-{val}-" for key, val in settings.items()])
        )

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

    def plot_bo():
        pass

    def calibration_vs_epochs(self):
        n_seeds = 10  # len(self.loadpths)
        n_epochs = 50  # len(self.loadpths)
        epochs = list(range(1, n_epochs + 1))
        results = np.full(
            (
                len(self.problems),
                len(self.surrogates),
                len(self.metrics),
                n_epochs,
                n_seeds,
            ),
            np.nan,
        )
        for i_p, problem in enumerate(self.problems):
            for i_s, surrogate in enumerate(self.surrogates):
                for i_e, experiment in enumerate(
                    [p for p in self.loadpths if surrogate in p and problem in p]
                ):
                    if os.path.isdir(experiment) and os.path.isfile(
                        experiment + "parameters.json"
                    ):
                        with open(experiment + "parameters.json") as json_file:
                            parameters = json.load(json_file)
                        # Running over epochs
                        files_in_path = [
                            f for f in os.listdir(experiment) if "scores" in f
                        ]
                        for file in files_in_path:
                            if "---epoch-" in file:
                                epoch_idx = (
                                    int(file.split("---epoch-")[-1].split(".json")[0])
                                    - 1
                                )
                            else:
                                epoch_idx = n_epochs - 1

                            with open(experiment + file) as json_file:
                                scores = json.load(json_file)

                            if self.settings.items() <= parameters.items():
                                for i_m, metric in enumerate(self.metrics):
                                    results[i_p, i_s, i_m, epoch_idx, i_e] = scores[
                                        metric
                                    ]
                    else:
                        print(f"No such file: {experiment}scores.json")

            for i_s, surrogate in enumerate(self.surrogates):
                fig = plt.figure()
                for i_m, metric in enumerate(self.metric_labels):
                    plt.subplot(len(self.metrics), 1, i_m + 1)
                    means = np.nanmean(results[i_p, i_s, i_m], axis=-1)
                    stds = np.nanstd(results[i_p, i_s, i_m], axis=-1)
                    plt.plot(epochs, means)
                    plt.fill_between(
                        epochs,
                        means + 1 * stds,
                        means - 1 * stds,
                        color="blue",
                        alpha=0.1,
                        # label=r"$\mathcal{M}_{" + str(n_stds) + "\sigma}$",
                    )
                    plt.ylabel(metric)
                # plt.legend()
                plt.xlabel("Epochs")
                plt.tight_layout()
                settings = (
                    str.join(
                        "--",
                        [
                            str(key) + "-" + str(val)
                            for key, val in self.settings.items()
                        ],
                    ).replace(".", "-")
                    + problem
                    + "-"
                    + surrogate
                )
                settings = "all" if settings == "" else settings
                fig.savefig(f"{self.savepth}calibration-vs-epochs---{settings}.pdf")
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
                ):
                    continue

                with open(experiment + "dataset.json") as json_file:
                    dataset = json.load(json_file)

                parameters = Parameters(parameters)
                module = importlib.import_module(parameters.data_location)
                data_class = getattr(module, parameters.data_class)
                data = data_class(parameters)
                x_min_loc = data.problem.min_loc
                x_lbs = dataset["x_lbs"]
                x_ubs = dataset["x_ubs"]
                x_1 = np.linspace(x_lbs[0], x_ubs[0], 100)
                x_2 = np.linspace(x_lbs[1], x_ubs[1], 100)
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
                    x_min_loc[0], x_min_loc[1], color="green", marker="o", markersize=10
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
                # plt.title(f"{parameters.problem}-{parameters.surrogate}")

                fig.savefig(
                    f"{self.savepth}bo-iters--{parameters.problem}-{parameters.surrogate}"
                    + f"--n-epochs-{n_epochs}--seed-{seed}--d-{parameters.d}.pdf"
                )
                plt.close()

    def bo_4x4_contour(self, n_epoch: int = 10, seed: int = 0):
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
                ):
                    continue

                with open(experiment + "dataset.json") as json_file:
                    dataset = json.load(json_file)

                parameters = Parameters(parameters)
                module = importlib.import_module(parameters.data_location)
                data_class = getattr(module, parameters.data_class)
                data = data_class(parameters)
                x_lbs = dataset["x_lbs"]
                x_ubs = dataset["x_ubs"]
                x_1 = np.linspace(x_lbs[0], x_ubs[0], 100)
                x_2 = np.linspace(x_lbs[1], x_ubs[1], 100)
                X1, X2 = np.meshgrid(x_1, x_2)
                y = np.full((100, 100), np.nan)
                for i1, x1 in enumerate(x_1):
                    for i2, x2 in enumerate(x_2):
                        y[i1, i2] = data.get_y(np.array([[x1, x2]])).squeeze()

                fig = plt.figure()
                ax = fig.add_subplot(111)
                pc = ax.pcolormesh(X1, X2, y)
                fig.colorbar(pc)

                X = np.array(dataset["X"])
                n_initial = parameters.n_initial
                x_1_init = X[:n_initial, 0]
                x_2_init = X[:n_initial, 1]
                plt.scatter(
                    x_1_init, x_2_init, marker=".", color="black",
                )
                x_1_bo = X[n_initial : n_initial + n_epoch, 0]
                x_2_bo = X[n_initial : n_initial + n_epoch, 1]
                for i in range(len(x_1_bo)):
                    plt.text(x_1_bo[i], x_2_bo[i], str(i + 1))

                plt.xlabel(r"$x_1$")
                plt.ylabel(r"$x_2$")
                # plt.title(f"{parameters.problem}-{parameters.surrogate}")

                fig.savefig(
                    f"{self.savepth}bo-iters--{parameters.problem}-{parameters.surrogate}"
                    + f"--n-epochs-{n_epoch}--seed-{seed}--d-{parameters.d}.pdf"
                )
                plt.close()

    def bo_regret_vs_no_bo_calibration(self, epoch: int = 89, avg: bool = False):
        self.results = np.full(
            (
                len(self.problems),
                len(self.surrogates),
                len(self.acquisitions),
                len(self.ds),
                len(self.seeds),
                len(self.bos),
            ),
            np.nan,
        )
        for experiment in self.loadpths:
            if (
                os.path.isdir(experiment)
                and os.path.isfile(f"{experiment}parameters.json")
                and os.path.isfile(f"{experiment}scores.json")
            ):
                with open(f"{experiment}parameters.json") as json_file:
                    parameters = json.load(json_file)

                i_pro = self.problems.index(parameters["problem"])
                i_sur = self.surrogates.index(parameters["surrogate"])
                i_acq = self.acquisitions.index(parameters["acquisition"])
                i_dim = self.ds.index(parameters["d"])
                i_see = self.seeds.index(parameters["seed"])

                if parameters["bo"]:
                    with open(f"{experiment}scores---epoch-{epoch}.json") as json_file:
                        scores = json.load(json_file)
                    self.results[i_pro, i_sur, i_acq, i_dim, i_see, 0] = scores[
                        "regret"
                    ]
                else:
                    with open(f"{experiment}scores.json") as json_file:
                        scores = json.load(json_file)
                    self.results[i_pro, i_sur, i_acq, i_dim, i_see, 1] = scores[
                        "y_calibration_mse"
                    ]

        for i_dim, dim in enumerate(self.ds):
            for i_pro, problem in enumerate(self.problems):
                if not np.any(  # demanding all surrogates have carried out all epochs
                    np.isnan(self.results[i_pro, :, :, i_dim, :, :])
                ):
                    fig = plt.figure()
                    for i_sur, surrogate in enumerate(self.surrogates):
                        x = (
                            np.mean(
                                self.results[i_pro, i_sur, :, i_dim, :, 0], axis=-1
                            ).squeeze()
                            if avg
                            else self.results[i_pro, i_sur, :, i_dim, :, 0].flatten()
                        )
                        y = (
                            np.mean(
                                self.results[i_pro, i_sur, :, i_dim, :, 1], axis=-1
                            ).squeeze()
                            if avg
                            else self.results[i_pro, i_sur, :, i_dim, :, 1].flatten()
                        )
                        plt.scatter(
                            x,
                            y,
                            color=ps[surrogate]["c"],
                            marker=ps[surrogate]["m"],
                            label=f"{surrogate}",
                        )
                    plt.xlabel("Regret")
                    plt.ylabel("Calibration MSE")
                    # plt.xlim([0, 2])
                    plt.xscale("log")
                    # plt.yscale("log")
                    plt.legend()
                    fig.savefig(
                        f"{self.savepth}regret-vs-no-bo-calibration|{problem}({dim})|epoch={epoch}|avg={avg}.pdf"
                    )

