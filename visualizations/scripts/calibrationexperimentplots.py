from typing import Dict
from imports.general import *
from imports.ml import *


class Figures(object):
    def __init__(self, experiments: list[str], settings: Dict[str, str]):
        self.experiments = experiments
        self.settings = settings

    def plot_calibration():
        pth = "./calibration_results/"
        files = os.listdir(pth)
        res_names = [
            "GP_y_calibration_mse",
            "RF_y_calibration_mse",
            "mean_GP_nentropy",
            "mean_RF_nentropy",
            "mean_RF_hist_nentropy",
            "true_nentropy",
        ]

        Ks = [1, 5, 10, 100]
        snrs = [0.1, 1.0, 10.0]
        train_sizes = [5, 50, 500]
        noise_dists = ["Normal"]
        Ds = list(range(1, 10 + 1))
        seeds = list(range(1, 30 + 1))
        results = np.full(
            (
                len(res_names),
                len(Ks),
                len(snrs),
                len(train_sizes),
                len(noise_dists),
                len(Ds),
                len(seeds),
            ),
            np.nan,
        )

        for i_K, K in enumerate(Ks):
            for i_snr, snr in enumerate(snrs):
                for i_t, train_size in enumerate(train_sizes):
                    for i_n, noise_dist in enumerate(noise_dists):
                        for i_d, D in enumerate(Ds):
                            n_bins = 50
                            GP_y_curve = np.full((len(seeds), n_bins), np.nan)
                            RF_y_curve = np.full((len(seeds), n_bins), np.nan)
                            GP_f_curve = np.full((len(seeds), 19), np.nan)
                            RF_f_curve = np.full((len(seeds), 19), np.nan)
                            for i_s, seed in enumerate(seeds):
                                summary = {
                                    "N": int(3000 + train_size),
                                    "K": K,
                                    "D": D,
                                    "SNR": snr,
                                    "train_size": train_size,
                                    "noise_dist": noise_dist,
                                    "seed": seed,
                                }
                                settings = "summary---" + str.join(
                                    "--",
                                    [
                                        str(key) + "-" + str(val)
                                        for key, val in summary.items()
                                    ],
                                ).replace(".", "p")
                                if settings + ".npy" in files:
                                    file = np.load(
                                        pth + settings + ".npy", allow_pickle=True
                                    ).item()
                                    for i_r, res_name in enumerate(res_names):
                                        results[
                                            i_r, i_K, i_snr, i_t, i_n, i_d, i_s
                                        ] = file[res_name]
                                    y_p_array = file["y_p_array"]
                                    f_p_array = file["f_p_array"]
                                    GP_y_curve[i_s, :] = file["GP_y_calibration"]
                                    RF_y_curve[i_s, :] = file["RF_y_calibration"]
                                    GP_f_curve[i_s, :] = file["GP_f_calibration"]
                                    RF_f_curve[i_s, :] = file["RF_f_calibration"]
                            if not np.all(np.isnan(GP_y_curve)):
                                # y calibration
                                GP_y_curve_mean = np.nanmean(GP_y_curve, axis=0)
                                RF_y_curve_mean = np.nanmean(RF_y_curve, axis=0)
                                GP_y_curve_std = np.nanstd(
                                    GP_y_curve, axis=0
                                ) / np.sqrt(len(seeds))
                                RF_y_curve_std = np.nanstd(
                                    RF_y_curve, axis=0
                                ) / np.sqrt(len(seeds))
                                fig = plt.figure()
                                plt.plot(
                                    y_p_array,
                                    y_p_array,
                                    "--",
                                    label="Perfectly calibrated",
                                )
                                plt.plot(
                                    y_p_array,
                                    GP_y_curve_mean,
                                    "o",
                                    color="green",
                                    label=r"$\mathcal{GP}$ ",
                                )
                                eb = plt.errorbar(
                                    y_p_array,
                                    GP_y_curve_mean,
                                    yerr=GP_y_curve_std,
                                    color="green",
                                    capsize=4,
                                    alpha=0.5,
                                )  # ,label=r"$\mathcal{GP}$ s.e.m."
                                eb[-1][0].set_linestyle("--")
                                plt.plot(
                                    y_p_array,
                                    RF_y_curve_mean,
                                    "o",
                                    color="orange",
                                    label=r"$\mathcal{RF}$ ",
                                )
                                eb = plt.errorbar(
                                    y_p_array,
                                    RF_y_curve_mean,
                                    yerr=RF_y_curve_std,
                                    color="orange",
                                    capsize=4,
                                    alpha=0.5,
                                )  # ,label=r"$\mathcal{RF}$ s.e.m."
                                eb[-1][0].set_linestyle("--")
                                plt.legend()
                                plt.xlabel(r"$p$")
                                plt.ylabel(r"$\mathcal{C}_{\mathbf{y}}$")
                                name = "y-calibration|N_train={}|K={}|D={}|SNR={}".format(
                                    train_size, K, D, snr
                                )
                                plt.tight_layout()
                                fig.savefig("figs/" + name + ".pdf")

                                # f calibration
                                GP_f_curve_mean = np.nanmean(GP_f_curve, axis=0)
                                RF_f_curve_mean = np.nanmean(RF_f_curve, axis=0)
                                GP_f_curve_std = np.nanstd(
                                    GP_f_curve, axis=0
                                ) / np.sqrt(len(seeds))
                                RF_f_curve_std = np.nanstd(
                                    RF_f_curve, axis=0
                                ) / np.sqrt(len(seeds))
                                fig = plt.figure()
                                plt.plot(
                                    f_p_array,
                                    f_p_array,
                                    "--",
                                    label="Perfectly calibrated",
                                )
                                plt.plot(
                                    f_p_array,
                                    GP_f_curve_mean,
                                    "o",
                                    color="green",
                                    label=r"$\mathcal{GP}$ ",
                                )
                                eb = plt.errorbar(
                                    f_p_array,
                                    GP_f_curve_mean,
                                    yerr=GP_f_curve_std,
                                    color="green",
                                    capsize=4,
                                    alpha=0.5,
                                )  # ,label=r"$\mathcal{GP}$ s.e.m."
                                eb[-1][0].set_linestyle("--")
                                plt.plot(
                                    f_p_array,
                                    RF_f_curve_mean,
                                    "o",
                                    color="orange",
                                    label=r"$\mathcal{RF}$ ",
                                )
                                eb = plt.errorbar(
                                    f_p_array,
                                    RF_f_curve_mean,
                                    yerr=RF_f_curve_std,
                                    color="orange",
                                    capsize=4,
                                    alpha=0.5,
                                )  # ,label=r"$\mathcal{RF}$ s.e.m."
                                eb[-1][0].set_linestyle("--")
                                plt.legend()
                                plt.xlabel(r"$p$")
                                plt.ylabel(r"$\mathcal{C}_{\mathbf{f}}$")
                                name = "f-calibration|N_train={}|K={}|D={}|SNR={}".format(
                                    train_size, K, D, snr
                                )
                                plt.tight_layout()
                                fig.savefig("figs/" + name + ".pdf")

                    result_ = results[:, i_K, i_snr, i_t, i_n, :, :]
                    result = pd.DataFrame(
                        data=np.nanmean(result_, axis=-1), index=res_names, columns=Ds
                    )
                    std = pd.DataFrame(
                        data=np.nanstd(result_, axis=-1), index=res_names, columns=Ds
                    )
                    # sem     		= pd.DataFrame(data=np.nanstd(result_,axis=-1),index=res_names,columns=Ds)
                    if not np.all(np.isnan(result_)):
                        # Y CALIBRATION MSE
                        fig = plt.figure()
                        Ds = np.array(Ds)
                        eb = plt.errorbar(
                            Ds + 0.2,
                            result.loc["GP_y_calibration_mse", :],
                            yerr=std.loc["GP_y_calibration_mse", :],
                            elinewidth=3,
                            capsize=4,
                            color="green",
                            label=r"$\mathcal{GP}$",
                            alpha=0.4,
                        )
                        eb[-1][0].set_linestyle("--")
                        plt.errorbar(
                            Ds + 0.2,
                            result.loc["GP_y_calibration_mse", :],
                            yerr=std.loc["GP_y_calibration_mse", :]
                            / np.sqrt(len(seeds)),
                            elinewidth=3,
                            capsize=4,
                            color="green",
                            ls="none",
                        )
                        eb = plt.errorbar(
                            Ds - 0.2,
                            result.loc["RF_y_calibration_mse", :],
                            yerr=std.loc["RF_y_calibration_mse", :],
                            elinewidth=3,
                            capsize=4,
                            color="orange",
                            label=r"$\mathcal{RF}$",
                            alpha=0.4,
                        )
                        eb[-1][0].set_linestyle("--")
                        plt.errorbar(
                            Ds - 0.2,
                            result.loc["RF_y_calibration_mse", :],
                            yerr=std.loc["RF_y_calibration_mse", :]
                            / np.sqrt(len(seeds)),
                            elinewidth=3,
                            capsize=4,
                            color="orange",
                            ls="none",
                        )
                        plt.xticks(ticks=Ds, labels=Ds)
                        plt.ylabel("MSE")
                        plt.xlabel("X dimensions")
                        plt.legend()
                        name = "Calibration|N_train={}|K={}|SNR={}".format(
                            train_size, K, snr
                        )
                        # plt.title(name);
                        plt.tight_layout()
                        fig.savefig("figs/" + name + ".pdf")

                        # SHARPNESS CALIBRATION MSE
                        fig = plt.figure()
                        Ds = np.array(Ds)
                        eb = plt.errorbar(
                            Ds + 0.2,
                            result.loc["mean_GP_nentropy", :],
                            yerr=std.loc["mean_GP_nentropy", :],
                            elinewidth=3,
                            capsize=4,
                            color="green",
                            label=r"$\mathcal{GP}$",
                            alpha=0.4,
                        )
                        eb[-1][0].set_linestyle("--")
                        plt.errorbar(
                            Ds + 0.2,
                            result.loc["mean_GP_nentropy", :],
                            yerr=std.loc["mean_GP_nentropy", :] / np.sqrt(len(seeds)),
                            elinewidth=3,
                            capsize=4,
                            color="green",
                            ls="none",
                        )

                        eb = plt.errorbar(
                            Ds - 0.2,
                            result.loc["mean_RF_nentropy", :],
                            yerr=std.loc["mean_RF_nentropy", :],
                            elinewidth=3,
                            color="orange",
                            capsize=4,
                            label=r"$\mathcal{RF}_N$",
                            alpha=0.4,
                        )
                        eb[-1][0].set_linestyle("--")
                        plt.errorbar(
                            Ds - 0.2,
                            result.loc["mean_RF_nentropy", :],
                            yerr=std.loc["mean_RF_nentropy", :] / np.sqrt(len(seeds)),
                            elinewidth=3,
                            capsize=4,
                            color="orange",
                            ls="none",
                        )

                        eb = plt.errorbar(
                            Ds - 0.3,
                            result.loc["mean_RF_hist_nentropy", :],
                            yerr=std.loc["mean_RF_hist_nentropy", :],
                            elinewidth=3,
                            color="red",
                            capsize=4,
                            label=r"$\mathcal{RF}_H$",
                            alpha=0.4,
                        )
                        eb[-1][0].set_linestyle("--")
                        plt.errorbar(
                            Ds - 0.3,
                            result.loc["mean_RF_hist_nentropy", :],
                            yerr=std.loc["mean_RF_hist_nentropy", :]
                            / np.sqrt(len(seeds)),
                            elinewidth=3,
                            capsize=4,
                            color="red",
                            ls="none",
                        )

                        plt.plot(
                            Ds,
                            result.loc["true_nentropy", :],
                            "*",
                            color="black",
                            label="True",
                        )
                        plt.xticks(ticks=Ds, labels=Ds)
                        plt.ylabel(r"$\mathcal{S}$")
                        plt.xlabel("X dimensions")
                        plt.legend()
                        name = "Sharpness|N_train={}|K={}|SNR={}".format(
                            train_size, K, snr
                        )
                        plt.tight_layout()
                        fig.savefig("figs/" + name + ".pdf")

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

