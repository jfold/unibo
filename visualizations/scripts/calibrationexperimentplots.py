from typing import Dict
from imports.general import *
from imports.ml import *


class Figures(object):
    def __init__(self, loadpths: list[str] = [], settings: Dict[str, str] = {}):
        self.loadpths = loadpths
        self.surrogates = list(set([pth.split("|")[-1] for pth in self.loadpths]))
        self.settings = settings
        self.savepth = os.getcwd() + "/visualizations/figures/"

    def load_raw(self):
        self.calibrations = []
        self.sharpnesses = []
        self.names = []
        for surrogate in self.surrogates:
            calibrations = []
            sharpnesses = []
            for experiment in [
                p for p in self.loadpths if p.split("|")[-1] == surrogate
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
                        sharpnesses.append(np.array(scores["sharpness"]))
            self.names.append(name)
            self.calibrations.append(np.array(calibrations))
            self.sharpnesses.append(np.array(sharpnesses))

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
