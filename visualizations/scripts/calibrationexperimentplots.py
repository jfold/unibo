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
        plt.plot(
            self.p_axis, self.p_axis, "--", label="Perfectly calibrated",
        )
        for i, calibrations in enumerate(self.calibrations):
            if calibrations.shape[0] < 2:
                continue

            mean_calibration = np.nanmean(calibrations, axis=0)
            std_calibration = np.nanstd(calibrations, axis=0)
            plt.plot(
                self.p_axis,
                mean_calibration,
                "o",
                label=r"$\mathcal{" + self.names[i] + "}$ ",
            )
            eb = plt.errorbar(
                self.p_axis,
                mean_calibration,
                yerr=std_calibration,
                # color="green",
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
        pth = "./calibration_results/"
        files = [f for f in os.listdir(pth) if f.startswith("bo---")]
        res_names = ["y_dist", "X_dist", "regret", "mean_regret"]
        problems = ["Alpine01", "Adjiman", "Ackley"]
        surrogates = ["GP"]
        n_initials = [10]
        n_evals = [5, 10, 20, 50]
        Ds = [2, 4, 6, 8, 10]
        seeds = list(range(0, 50 + 1))
        results = np.full((len(res_names), len(n_evals), len(Ds), len(seeds)), np.nan)
        distances = np.full((1, len(n_evals), len(Ds), len(seeds)), np.nan)

        for surrogate in surrogates:
            for problem in problems:
                for n_initial in n_initials:
                    for i_n, n_eval in enumerate(n_evals):
                        for i_d, D in enumerate(Ds):
                            for i_s, seed in enumerate(seeds):
                                summary = {
                                    "problem": problem + f"({D})",
                                    "seed": seed,
                                    "surrogate": surrogate,
                                    "n_initial": n_initial,
                                    "n_evals": n_eval,
                                }
                                settings = "bo---" + str.join(
                                    "--",
                                    [
                                        str(key) + "-" + str(val)
                                        for key, val in summary.items()
                                    ],
                                ).replace(".", "p")
                                file_name = settings + ".npy"
                                if file_name in files:
                                    file = np.load(
                                        pth + file_name, allow_pickle=True
                                    ).item()
                                    for i_r, res_name in enumerate(res_names):
                                        results[i_r, i_n, i_d, i_s] = file[res_name]
                                    distances[0, i_n, i_d, i_s] = np.sqrt(
                                        np.mean(
                                            (file["fmin_loc"] - file["bo_opt_X"]) ** 2
                                        )
                                    )

                    for i_r, res_name in enumerate(res_names):
                        fig = plt.figure()
                        for i_n, n_eval in enumerate(n_evals):
                            mean = pd.DataFrame(
                                data=np.nanmean(results[:, i_n, :, :], axis=-1),
                                index=res_names,
                                columns=Ds,
                            )
                            std = pd.DataFrame(
                                data=np.nanstd(results[:, i_n, :, :], axis=-1),
                                index=res_names,
                                columns=Ds,
                            )
                            mean_ = mean.loc[res_name]
                            std_ = std.loc[res_name]
                            if not np.all(np.isnan(mean_)):
                                plt.errorbar(
                                    Ds,
                                    mean_,
                                    yerr=std_,
                                    elinewidth=3,
                                    capsize=4,
                                    label=f"n_evals={str(n_eval)}",
                                )

                        plt.xlabel("Dimensions")
                        plt.ylabel(res_name)
                        plt.legend()
                        fig.savefig("figs/bo---" + problem + "--" + res_name + ".pdf")
                        print("Saved", "bo---" + problem + "--" + res_name)

                    # mean     	= np.nanmean(distances[:,i_n,:,:],axis=-1).squeeze()
                    # std     	= np.nanstd(distances[:,i_n,:,:],axis=-1).squeeze()
                    # if not np.all(np.isnan(mean)):
                    # 	fig = plt.figure()
                    # 	name = "X_norm_dist"
                    # 	plt.errorbar(Ds,mean,yerr=std, elinewidth=3, capsize=4,color="blue")
                    # 	plt.xlabel("Dimensions"); plt.ylabel(name);
                    # 	fig.savefig("figs/bo---"+problem+"--"+name+".pdf")
                    # 	print("Saved","bo---"+problem+"--"+name)

