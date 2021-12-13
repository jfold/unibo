from dataclasses import asdict
import json
from numpy.lib.npyio import save
from imports.general import *
from imports.ml import *
from src.parameters import Parameters
from visualizations.scripts.calibrationplots import CalibrationPlots
from base.dataset import Dataset


class Calibration(CalibrationPlots):
    """Calibration experiment class """

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)
        self.__dict__.update(asdict(parameters))
        self.summary = {}

    def check_gaussian_sharpness(self, mus: np.ndarray, sigmas: np.ndarray):
        """Calculates the sharpness (negative entropy) of the gaussian distributions 
        with means: mus and standard deviation: sigmas
        """
        sharpness = np.array(
            [-norm.entropy(mus[i], sigmas[i]) for i in range(mus.shape[0])]
        )
        mean_sharpness = np.mean(sharpness)
        self.summary.update({"sharpness": sharpness, "mean_sharpness": mean_sharpness})

    def check_histogram_sharpness(self, model: Model, X: np.ndarray, n_bins: int = 50):
        """Calculates the sharpness (negative entropy) of the histogram distributions 
        calculated from input X
        """
        if hasattr(model, "histogram_sharpness"):
            hist_sharpness, mean_hist_sharpness = model.histogram_sharpness(
                X, n_bins=n_bins
            )
            self.summary.update(
                {
                    f"{model.name}_hist_sharpness": hist_sharpness,
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
    ):
        """Calculates the calibration of underlying mean (f), hence without noise.

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

    def check_y_calibration(
        self, mus: np.ndarray, sigmas: np.ndarray, y: np.ndarray, n_bins: int = 50,
    ):
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

        self.summary.update(
            {
                "y_p_array": p_array,
                "y_calibration": calibrations,
                "y_calibration_mse": np.mean((calibrations - p_array) ** 2),
            }
        )

    def expected_log_predictive_density(
        self, mus: np.ndarray, sigmas: np.ndarray, y: np.ndarray,
    ):
        """Calculates expected log predictive density (elpd) using
        \mathbb{E}\left[\log p_\theta(\textbf{y}|\textbf{X})\right]  
        which essientially is "on average how likely is a new test data under the model".
        """
        log_cdfs = np.array(
            [
                norm.logcdf(y[i], loc=mus[i], scale=sigmas[i])
                for i in range(sigmas.shape[0])
            ]
        )
        elpd = np.mean(log_cdfs)
        self.summary.update({"elpd": elpd})

    def nmse(self, y: np.ndarray, predictions: np.ndarray):
        """Calculates normalized mean square error by 
        nmse = \ frac{1}{N\cdot\mathbb{V}[\textbf{y}]} \sum_i (\textbf{y}-\hat{\textbf{y}})^2
        where N is the length of y
        """
        mse = np.mean((y - predictions) ** 2)
        nmse = mse / np.var(y)
        self.summary.update({"mse": mse})
        self.summary.update({"nmse": nmse})

    def regret(self):
        raise NotImplementedError()
        self.summary.update({"regret": None})
        self.summary.update({"total_regret": None})
        self.summary.update({"mean_regret": None})

    def glob_min_dist(self, surrogate: Model, dataset: Dataset):
        raise NotImplementedError()
        self.summary.update({"x_opt_dist": None})
        self.summary.update({"x_opt_mean_dist": None})

    def save(self, save_settings: str = ""):
        final_dict = {k: v.tolist() for k, v in self.summary.items()}
        json_dump = json.dumps(final_dict)
        with open(self.savepth + f"scores{save_settings}.json", "w") as f:
            f.write(json_dump)

    def analyze(
        self, surrogate: Model, dataset: Dataset, save_settings: str = "",
    ):
        """Calculates calibration, sharpness, expected log predictive density and 
        normalized mean square error functions for the "surrogate" on a testset
        drawn from "dataset".
        """
        X_test, y_test = dataset.sample_testset()
        self.ne_true = dataset.data.ne_true
        mu_test, sigma_test = surrogate.predict(X_test)
        self.check_y_calibration(mu_test, sigma_test, y_test)
        self.check_gaussian_sharpness(
            mu_test, sigma_test,
        )
        self.expected_log_predictive_density(
            mu_test, sigma_test, y_test,
        )
        self.nmse(y_test, mu_test)

        # Throw out?
        # self.check_histogram_sharpness(surrogate, X_test)
        # self.check_f_calibration(
        #     mu_test, sigma_test, dataset.data.f_test, name=name,
        # )

        if self.plot_it and self.save_it:
            name = f"{save_settings}"
            if self.d == 1:
                self.plot_predictive(
                    dataset, X_test, y_test, mu_test, sigma_test,
                )
            self.plot_y_calibration(name=name)
            self.plot_sharpness_histogram(name=name)

        # Save
        if self.save_it:
            self.save(save_settings)
            print(f"Successfully saved with settings: {self.surrogate}")

