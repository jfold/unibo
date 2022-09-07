from dataclasses import asdict
import json
from netrc import netrc
from numpy.lib.npyio import save
from scipy.sparse import data
from imports.general import *
from imports.ml import *
from src.parameters import Parameters
from src.dataset import Dataset


class Metrics(object):
    """Metric class """

    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        self.summary = {}

    def sharpness_gaussian(
        self, mus: np.ndarray, sigmas: np.ndarray, ne_true: float = None, name: str = ""
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
        if ne_true is not None:
            self.summary.update(
                {
                    "sharpness_abs_error": np.abs(ne_true - mean_sharpness),
                    "posterior_variance": np.mean(sigmas) ** 2,
                }
            )

        # if self.plot_it and self.save_it:
        #     self.plot_sharpness_histogram(name=name)

    def bias(self, mus: np.ndarray, f: np.ndarray) -> None:
        mse = np.mean((mus - f) ** 2)
        nmse = mse / np.var(f)
        self.summary.update(
            {"bias_mse": mse, "bias_nmse": nmse,}
        )

    def sharpness_histogram(
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

    def calibration_f(
        self, mus: np.ndarray, sigmas: np.ndarray, f: np.ndarray, n_bins: int = 50,
    ) -> None:
        p_array = np.linspace(0.01, 0.99, n_bins)
        calibrations = np.full((n_bins,), np.nan)
        assert mus.size == sigmas.size == f.size
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

    def calibration_y(
        self,
        mus: np.ndarray,
        sigmas: np.ndarray,
        y: np.ndarray,
        n_bins: int = 50,
        return_mse: bool = False,
    ) -> None:
        """Calculates the calibration of the target (y).
        ### eq. (3) in "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
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

    def calibration_y_local(
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
            dataset.data.X_train, X, metric="euclidean"
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
            calibrations_intervals[i] = self.calibration_y(
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
                "calibration_y_local": calibrations,
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
        self.summary.update({"mse": mse, "nmse": nmse})

    def regret(self, dataset: Dataset) -> None:
        y_solution = dataset.data.f_max if self.maximization else dataset.data.f_min
        y_opt = dataset.y_opt
        regret = np.abs(y_opt - y_solution)
        self.summary.update({"regret": np.sum(regret)})

    def glob_min_dist(self, dataset: Dataset) -> None:
        squared_error = (dataset.X_opt - np.array(dataset.data.y_min_loc)) ** 2
        self.summary.update(
            {
                "x_opt_dist": np.sqrt(np.sum(squared_error)),
                "x_opt_mean_dist": np.mean(squared_error),
            }
        )

    # def calc_true_regret(self, parameters: Dict, dataset: Dict):
    #     # inferring true regret
    #     dataobj = Dataset(Parameters(parameters))
    #     X = np.array(dataset["X"])
    #     y_clean = dataobj.data.get_y(X, add_noise=False)
    #     y_clean = np.array([np.min(y_clean[:i]) for i in range(1, len(y_clean) + 1)])
    #     y_clean = y_clean[dataset["n_initial"] - 1 :]
    #     true_regrets = np.abs(dataobj.data.f_min - y_clean)
    #     return true_regrets

    # def calc_running_inner_product(self, dataset: Dataset) -> np.ndarray:
    #     X = np.array(dataset["X"])
    #     running_inner_product = np.cumsum(np.diag(X @ X.T)).squeeze()[
    #         dataset["n_initial"] - 1 :
    #     ]
    #     return running_inner_product

    def calc_mahalanobis_dist_to_current_best(self, dataset: Dataset) -> np.ndarray:
        X = np.array(dataset.X)[: dataset.n_initial, :]
        y = np.array(dataset.y)[: dataset.n_initial, :]
        Sigma = np.diag((np.array(dataset.x_ubs) - np.array(dataset.x_lbs)) / 12)
        idx_opt = np.argmin(y)
        y_opt = y[[idx_opt], :]
        X_opt = X[[idx_opt], :].T
        mahalanobis_dists = [np.nan]
        cur_y = y[[i], :]
        cur_X = X[[-1], :].T
        dist = mahalanobis(cur_X, X_opt, Sigma)
        self.summary.update({"mahalanobis_dist": dist})

    def save(self, save_settings: str = "") -> None:
        final_dict = {k: v.tolist() for k, v in self.summary.items()}
        json_dump = json.dumps(final_dict)
        with open(self.savepth + f"metrics{save_settings}.json", "w") as f:
            f.write(json_dump)

        if hasattr(self, "uct_metrics"):
            with open(self.savepth + f"metrics-uct{save_settings}.pkl", "wb") as f:
                pickle.dump(self.uct_metrics, f)

    def analyze(
        self, surrogate: Model, dataset: Dataset, save_settings: str = "",
    ) -> None:
        if surrogate is not None:
            self.ne_true = dataset.data.ne_true
            mu_test, sigma_test = surrogate.predict(dataset.data.X_test)
            self.calibration_f(mu_test, sigma_test, dataset.data.f_test)
            self.calibration_y(mu_test, sigma_test, dataset.data.y_test)
            self.calibration_y_local(
                dataset, mu_test, sigma_test, dataset.data.X_test, dataset.data.y_test
            )
            self.sharpness_gaussian(
                mu_test,
                sigma_test,
                ne_true=dataset.data.ne_true,
                name=f"{save_settings}",
            )
            self.expected_log_predictive_density(
                mu_test, sigma_test, dataset.data.y_test,
            )
            self.nmse(dataset.data.y_test, mu_test)
            self.bias(mu_test, dataset.data.f_test)
            self.uct_metrics = uct.metrics.get_all_metrics(
                mu_test.squeeze(),
                sigma_test.squeeze(),
                dataset.data.y_test.squeeze(),
                verbose=False,
            )
            self.improvement(dataset)

        self.regret(dataset)
        self.glob_min_dist(dataset)

        # Save
        if self.save_it:
            self.save(save_settings)

