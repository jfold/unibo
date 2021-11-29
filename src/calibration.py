from dataclasses import asdict
import json

from numpy.lib.npyio import save
from imports.general import *
from imports.ml import *
from src.parameters import Parameters
from surrogates.random_forest import RandomForest
from visualizations.scripts.calibrationplots import CalibrationPlots
from base.surrogate import Surrogate
from base.dataset import Dataset


class Calibration(CalibrationPlots):
    """Calibration experiment class """

    def __init__(self, parameters: Parameters) -> None:
        super().__init__()
        self.__dict__.update(asdict(parameters))
        self.summary_init()

    def summary_init(self):
        self.summary = {}
        self.settings = self.surrogate
        # str.join(
        #     "--", [str(key) + "-" + str(val) for key, val in self.summary.items()]
        # ).replace(".", "-")

    def check_gaussian_sharpness(self, mus: np.ndarray, sigmas: np.ndarray):
        """Calculates the sharpness (negative entropy) of the gaussian distributions 
        with means: mus and standard deviation: sigmas
        Args:
            mus (np.ndarray): predictive mean
            sigmas (np.ndarray): predictive standard deviation
            name (str): model name
        """
        sharpness = -tfp.distributions.Normal(mus, sigmas).entropy()
        mean_sharpness = tf.reduce_mean(sharpness)
        self.summary.update(
            {"sharpness": sharpness.numpy(), "mean_sharpness": mean_sharpness.numpy(),}
        )

    def check_histogram_sharpness(
        self, model: Surrogate, X: np.ndarray, n_bins: int = 50
    ):
        """Calculates the sharpness (negative entropy) of the histogram distributions 
        calculated from input X
        Args:
            mus (np.ndarray): predictive mean
            sigmas (np.ndarray): predictive standard deviation
            name (str): model name
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

        Args:
            mus (np.ndarray): predictive mean
            sigmas (np.ndarray): predictive std 
            f (np.ndarray): underlying mean function
            name (str): model name
            n_bins (int, optional): number of bins dividing the calibration curve. 
            Defaults to 50.
        """
        p = np.linspace(0.01, 0.99, n_bins)
        norm_dists = tfp.distributions.Normal(loc=0, scale=sigmas)
        fractiles = norm_dists.quantile(0.5 - p / 2)
        lb_indicators = mus + fractiles < f
        ub_indicators = f < mus - fractiles
        indicators = tf.logical_and(lb_indicators, ub_indicators)
        calibrations = tf.reduce_mean(tf.cast(indicators, dtype=tf.float32), axis=0)
        self.summary.update(
            {
                "f_p_array": p,
                "f_calibration": calibrations,
                "f_calibration_mse": np.mean((p - calibrations) ** 2),
            }
        )

    def check_y_calibration(
        self, mus: np.ndarray, sigmas: np.ndarray, y: np.ndarray, n_bins: int = 50,
    ):
        """Calculates the calibration of the target (y).

        Args:
            mus (np.ndarray): predictive mean
            sigmas (np.ndarray): predictive std 
            f (np.ndarray): underlying mean function
            name (str): model name
            n_bins (int, optional): number of bins dividing the calibration curve. 
            Defaults to 50.
        """
        p = np.linspace(0, 1, n_bins)
        norm_dists = tfp.distributions.Normal(loc=mus, scale=sigmas)
        # eq. (3) in "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
        indicators = y <= norm_dists.quantile(p)
        calibrations = np.mean(indicators, axis=0)
        self.summary.update(
            {
                "y_p_array": p,
                "y_calibration": calibrations,
                "y_calibration_mse": np.mean((calibrations - p) ** 2),
            }
        )

    def expected_log_predictive_density(
        self, mus: np.ndarray, sigmas: np.ndarray, y: np.ndarray,
    ):
        """Calculates expected log predictive density (elpd) using
        \mathbb{E}\left[\log p_\theta(\textbf{y}|\textbf{X})\right]  
        which essientially is "on average how likely is a new test data under the model".
        Args:
            mus (np.ndarray): predictions mean
            sigmas (np.ndarray): predictions uncertainties
            y (np.ndarray): [description]
            name (str): [description]
        """
        norm_dists = tfp.distributions.Normal(loc=mus, scale=sigmas)
        log_cdfs = norm_dists.log_cdf(y)
        elpd = tf.reduce_mean(log_cdfs).numpy()
        self.summary.update({"elpd": elpd})

    def nmse(self, y: np.ndarray, predictions: np.ndarray):
        """Calculates normalized mean square error by 
        nmse = \ frac{1}{N\cdot\mathbb{V}[\textbf{y}]} \sum_i (\textbf{y}-\hat{\textbf{y}})^2
        where N is the length of y
        Args:
            y (np.ndarray): true outputs
            predictions (np.ndarray): model prediction (\hat{y})
            name (string): model name
        """
        mse = np.mean((y - predictions) ** 2)
        nmse = mse / np.var(y)
        self.summary.update({"mse": mse})
        self.summary.update({"nmse": nmse})

    def save(self, save_settings: str = ""):
        final_dict = {k: v.tolist() for k, v in self.summary.items()}
        json_dump = json.dumps(final_dict)
        with open(self.savepth + f"scores{save_settings}.json", "w") as f:
            f.write(json_dump)

    def analyze(
        self, surrogate: Surrogate, dataset: Dataset, save_settings: str = "",
    ):
        """Calculates calibration, sharpness, expected log predictive density and 
        normalized mean square error functions for the "surrogate" on a testset
        drawn from "dataset".

        Args:
            surrogate (Surrogate): Surrogate object
            dataset (Dataset): Dataset object
            save_settings (str, optional): Extra string suffix for naming. Defaults to "".
        """
        X_test, y_test = dataset.sample_testset()
        self.ne_true = dataset.data.ne_true
        mu_test, sigma_test = surrogate.predict(X_test)
        name = f"{surrogate.name}{save_settings}"
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
            if self.d == 1:
                self.plot_predictive(
                    dataset, X_test, y_test, mu_test, sigma_test,
                )
            self.plot_y_calibration()
            self.plot_sharpness_histogram()

        # Save
        if self.save_it:
            self.save(save_settings)
            print("Successfully saved with settings:", self.settings)

