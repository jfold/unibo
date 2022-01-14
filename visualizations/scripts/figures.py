from typing import Dict
from imports.general import *
from imports.ml import *


class Figures(object):
    def __init__(self, loadpths: list[str] = [], settings: Dict[str, str] = {}):
        self.loadpths = loadpths
        self.surrogates = list(set([pth.split("|")[1] for pth in self.loadpths]))
        self.problems = list(set([pth.split("|")[2] for pth in self.loadpths]))
        self.settings = settings
        self.savepth = (
            os.getcwd()
            + "/visualizations/figures/"
            + str.join("-", [f"{key}-{val}-" for key, val in settings.items()])
        )
        self.metrics = [
            "nmse",
            "elpd",
            "y_calibration_mse",
            "y_calibration_nmse",
            "mean_sharpness",
            "x_opt_mean_dist",
            "x_opt_dist",
            "regret",
        ]
        self.metric_labels = [
            "nMSE",
            "ELPD",
            "Calibration MSE",
            "Calibration nMSE",
            "Sharpness",
            "Solution mean distance",
            "Solution distance",
            "Regret",
        ]

    def load_raw(self):
        self.calibrations = []
        self.names = []
        for surrogate in self.surrogates:
            calibrations = []
            sharpnesses = []
            for experiment in [
                p for p in self.loadpths if p.split("|")[1] == surrogate
            ]:
                if os.path.isfile(experiment + "scores.json") and os.path.isfile(
                    experiment + "parameters.json"
                ):
                    with open(experiment + "scores.json") as json_file:
                        scores = json.load(json_file)
                    with open(experiment + "parameters.json") as json_file:
                        parameters = json.load(json_file)
                    if self.settings.items() <= parameters.items():
                        self.p_axis = np.array(scores["y_p_array"])
                        name = f"{parameters['surrogate']}-{parameters['acquisition']}"
                        calibrations.append(np.array(scores["y_calibration"]))
            self.names.append(name)
            self.calibrations.append(np.array(calibrations))

    def generate(self,):
        self.load_raw()
        self.calibration()

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

    def bo_2d_contour(self):
        pass

    # fig = plt.figure()
    # for i_m, metric in enumerate(self.metric_labels):
    #     plt.subplot(len(self.metrics), 1, i_m + 1)
    #     means = np.nanmean(results[i_p, i_s, i_m], axis=-1)
    #     stds = np.nanstd(results[i_p, i_s, i_m], axis=-1)
    #     plt.plot(epochs, means)
    #     plt.fill_between(
    #         epochs,
    #         means + 1 * stds,
    #         means - 1 * stds,
    #         color="blue",
    #         alpha=0.1,
    #         # label=r"$\mathcal{M}_{" + str(n_stds) + "\sigma}$",
    #     )
    #     plt.ylabel(metric)
    # # plt.legend()
    # plt.xlabel("Epochs")
    # plt.tight_layout()
    # settings = (
    #     str.join(
    #         "--",
    #         [
    #             str(key) + "-" + str(val)
    #             for key, val in self.settings.items()
    #         ],
    #     ).replace(".", "-")
    #     + problem
    #     + "-"
    #     + surrogate
    # )
    # settings = "all" if settings == "" else settings
    # fig.savefig(f"{self.savepth}calibration-vs-epochs---{settings}.pdf")
    # plt.close()


if __name__ == "__main__":
    figures = Figures()
    figures.calibration_vs_epochs()
