from dataclasses import asdict
from imports.general import *
from imports.ml import *
from src.dataset import Dataset
from src.parameters import Parameters


class CalibrationPlots(object):
    """Calibration experiment class """

    def __init__(self, parameters: Parameters):
        self.__dict__.update(asdict(parameters))
        self.color = ps[parameters.surrogate]["c"]
        self.marker = ps[parameters.surrogate]["m"]

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
        plt.close()

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


class BayesianOptimizationPlots(object):
    def plot(self):
        fig = plt.figure()
        plt.plot(
            self.x_epochs,
            self.y_train_np[self.n_initial :],
            "--*",
            label="Bayesian Optimization",
        )
        plt.plot(
            self.x_epochs,
            np.tile(self.problem.fmin, len(self.x_epochs)),
            "-",
            label="Minimum",
        )
        plt.xlabel("Evaluations")
        plt.ylabel(r"$f$")
        plt.legend()
        plt.show()

    def plotit(self):

        clear_output()
        plt.scatter(self.X_train_np[:, 0], self.X_train_np[:, 1], c=self.y_train_np)
        plt.scatter(
            self.X_train_np[-1, 0],
            self.X_train_np[-1, 1],
            s=180,
            facecolors="none",
            label="Latest",
            edgecolors="r",
        )
        plt.plot(
            self.problem.min_loc[0],
            self.problem.min_loc[1],
            "*",
            markersize=16,
            label="Solution",
        )
        plt.colorbar()
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.legend()
        plt.show()
        time.sleep(0.5)
