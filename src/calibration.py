from dataclasses import asdict
import json
from numpy.lib.npyio import save
from scipy.sparse import data
from imports.general import *
from imports.ml import *
from src.optimizer import Optimizer
from src.parameters import Parameters
from src.dataset import Dataset


class Calibration(object):
    """Calibration experiment class """

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)
        self.__dict__.update(asdict(parameters))
        self.summary = {}

    def check_gaussian_sharpness(
        self, mus: np.ndarray, sigmas: np.ndarray, name: str = ""
    ) -> None:
        """Calculates the sharpness (negative entropy) of the gaussian distributions 
        with means: mus and standard deviation: sigmas
        """
        sharpness = np.array(
            [-norm.entropy(mus[i], sigmas[i]) for i in range(mus.shape[0])]
        )
        mean_sharpness = np.mean(sharpness)
        self.summary.update(
            {"mean_sharpness": mean_sharpness}  # "sharpness": sharpness,
        )
        # if self.plot_it and self.save_it:
        #     self.plot_sharpness_histogram(name=name)

    def check_histogram_sharpness(
        self, model: Model, X: np.ndarray, n_bins: int = 50
    ) -> None:
        """Calculates the sharpness (negative entropy) of the histogram distributions 
        calculated from input X
        """
        if hasattr(model, "histogram_sharpness"):
            hist_sharpness, mean_hist_sharpness = model.histogram_sharpness(
                X, n_bins=n_bins
            )
            self.summary.update(
                {
                    # f"{model.name}_hist_sharpness": hist_sharpness,
                    f"{model.name}_mean_hist_sharpness": mean_hist_sharpness,
                }
            )

    def check_f_calibration(
        self,
        mus: np.ndarray,
        sigmas: np.ndarray,
        f: np.ndarray,
        name: str,
        n_bins: int = 50,
    ) -> None:
        """# NOT USED. Calculates the calibration of underlying mean (f), hence without noise.

        """
        p_array = np.linspace(0.01, 0.99, n_bins)
        calibrations = np.full((n_bins,), np.nan)
        for i_p, p in enumerate(p_array):
            fractiles = [norm.ppf(0.5 - p / 2, loc=0, scale=sig) for sig in sigmas]
            lb_indicators = mus + fractiles < f
            ub_indicators = f < mus - fractiles
            indicators = np.logical_and(lb_indicators, ub_indicators)
            calibrations[i_p] = np.mean(indicators)

        self.summary.update(
            {
                "f_p_array": p_array,
                "f_calibration": calibrations,
                "f_calibration_mse": np.mean((p_array - calibrations) ** 2),
            }
        )

    def calibration_global(
        self,
        mus: np.ndarray,
        sigmas: np.ndarray,
        y: np.ndarray,
        n_bins: int = 50,
        return_mse: bool = False,
    ) -> None:
        """Calculates the calibration of the target (y).
        # eq. (3) in "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
        """
        p_array = np.linspace(0, 1, n_bins)
        calibrations = np.full((n_bins,), np.nan)
        for i_p, p in enumerate(p_array):
            indicators = y <= [
                norm.ppf(p, loc=mu, scale=sig) for mu, sig in zip(mus, sigmas)
            ]
            calibrations[i_p] = np.mean(indicators)

        if return_mse:
            return np.mean((calibrations - p_array) ** 2)
        else:
            self.summary.update(
                {
                    "y_p_array": p_array,
                    "y_calibration": calibrations,
                    "y_calibration_mse": np.mean((calibrations - p_array) ** 2),
                    "y_calibration_nmse": np.mean((calibrations - p_array) ** 2)
                    / np.var(p_array),
                }
            )

    def calibration_local(
        self,
        dataset: Dataset,
        mus: np.ndarray,
        sigmas: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        n_bins: int = 50,
    ) -> None:
        """Calculates the calibration of the target (y).
        # eq. (3) in "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
        """

        pair_dists = cdist(
            dataset.data.X, X, metric="euclidean"
        )  # pairwise euclidean dist between training and test points
        pair_dists = np.min(
            pair_dists, axis=0
        )  # only take radius of nearest training point
        counts, bins = np.histogram(
            pair_dists.flatten(), bins=n_bins
        )  # get histogram with 50 bins
        calibrations_intervals = np.full((n_bins,), np.nan)
        calibrations = np.full((n_bins,), np.nan)
        for i in range(len(bins) - 1):
            cond = np.logical_and(bins[i] <= pair_dists, pair_dists <= bins[i + 1])
            mus_, sigmas_, y_ = mus[cond], sigmas[cond], y[cond]
            calibrations_intervals[i] = self.calibration_global(
                mus_, sigmas_, y_, return_mse=True
            )
            calibrations[i] = np.inner(
                counts[: i + 1] / np.sum(counts[: i + 1]),
                calibrations_intervals[: i + 1],
            )
        #### for debugging:
        # plt.scatter(bins[1:], calibrations_intervals, label="Intervals")
        # plt.scatter(bins[1:], calibrations, label="All below")
        # plt.legend()
        # plt.xlabel("Distance to nearest training sample")
        # plt.show()
        # raise ValueError()
        self.summary.update(
            {
                "calibration_local_dists_to_nearest_train_sample": bins[1:],
                "calibration_local_intervals": calibrations_intervals,
                "calibration_local": calibrations,
            }
        )

    def expected_log_predictive_density(
        self, mus: np.ndarray, sigmas: np.ndarray, y: np.ndarray,
    ) -> None:
        """Calculates expected log predictive density (elpd) using
        \mathbb{E}\left[\log p_\theta(\textbf{y}|\textbf{X})\right]  
        which essientially is "on average how likely is a new test data under the model".
        """
        log_pdfs = np.array(
            [
                norm.logpdf(y[i], loc=mus[i], scale=sigmas[i])
                for i in range(sigmas.shape[0])
            ]
        )
        elpd = np.mean(log_pdfs)
        self.summary.update({"elpd": elpd})

    def improvement(self, dataset: Dataset):
        self.summary.update(
            {
                "expected_improvement": np.array(dataset.expected_improvement),
                "actual_improvement": np.array(dataset.actual_improvement),
            }
        )

    def nmse(self, y: np.ndarray, predictions: np.ndarray) -> None:
        """Calculates normalized mean square error by 
        nmse = \ frac{1}{N\cdot\mathbb{V}[\textbf{y}]} \sum_i (\textbf{y}-\hat{\textbf{y}})^2
        where N is the length of y
        """
        mse = np.mean((y - predictions) ** 2)
        nmse = mse / np.var(y)
        self.summary.update({"mse": mse})
        self.summary.update({"nmse": nmse})

    def regret(self, dataset: Dataset) -> None:
        y_solution = dataset.data.f_max if self.maximization else dataset.data.f_min
        regret = np.abs(dataset.y_opt - y_solution)
        self.summary.update({"regret": np.sum(regret)})

    def glob_min_dist(self, dataset: Dataset) -> None:
        squared_error = (dataset.X_opt - np.array(dataset.data.problem.min_loc)) ** 2
        self.summary.update({"x_opt_dist": np.sum(squared_error)})
        self.summary.update({"x_opt_mean_dist": np.mean(squared_error)})

    def save(self, save_settings: str = "") -> None:
        final_dict = {k: v.tolist() for k, v in self.summary.items()}
        json_dump = json.dumps(final_dict)
        with open(self.savepth + f"scores{save_settings}.json", "w") as f:
            f.write(json_dump)

        if hasattr(self, "uct_metrics"):
            with open(self.savepth + f"scores-uct{save_settings}.pkl", "wb") as f:
                pickle.dump(self.uct_metrics, f)

    def analyze(
        self, surrogate: Model, dataset: Dataset, save_settings: str = "",
    ) -> None:
        name = f"{save_settings}"
        if surrogate is not None:
            X_test, y_test = dataset.sample_testset()
            self.ne_true = dataset.data.ne_true
            self.y_max = dataset.data.y_max
            mu_test, sigma_test = surrogate.predict(X_test)
            self.calibration_global(mu_test, sigma_test, y_test)
            self.calibration_local(dataset, mu_test, sigma_test, X_test, y_test)
            self.check_gaussian_sharpness(mu_test, sigma_test, name)
            self.expected_log_predictive_density(
                mu_test, sigma_test, y_test,
            )
            self.nmse(y_test, mu_test)
            self.uct_metrics = uct.metrics.get_all_metrics(
                mu_test.squeeze(), sigma_test.squeeze(), y_test.squeeze(), verbose=False
            )
            self.improvement(dataset)
            # if self.plot_it and self.save_it:
            #     if self.d == 1:
            #         self.plot_predictive(
            #             dataset, X_test, y_test, mu_test, sigma_test, name=name
            #         )
            #     self.plot_y_calibration(name=name)

        self.regret(dataset)
        self.glob_min_dist(dataset)

        # Save
        if self.save_it:
            self.save(save_settings)

