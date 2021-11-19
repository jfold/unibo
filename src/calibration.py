from imports.general import *
from imports.ml import *
from src.parameters import Parameters
from surrogates.random_forest import RandomForest
from .plots import CalibrationPlots
from base.surrogate import Surrogate
from base.dataset import Dataset


class Calibration(CalibrationPlots):
    """Calibration experiment class """

    def __init__(self, parameters: Parameters) -> None:
        super().__init__()
        self.__dict__.update(parameters.__dict__)
        self.summary_init()
        if self.plot_data and self.d == 1:
            self.plot_xy()

    def summary_init(self):
        self.summary = {}
        for attr in []:
            self.summary.update({attr: getattr(self, attr)})
        self.settings = str.join(
            "--", [str(key) + "-" + str(val) for key, val in self.summary.items()]
        ).replace(".", "-")

    def check_gaussian_sharpness(self, mus: np.ndarray, sigmas: np.ndarray, name: str):
        sharpness = -tfp.distributions.Normal(mus, sigmas).entropy()
        mean_sharpness = tf.reduce_mean(sharpness)
        self.summary.update(
            {
                f"{name}_sharpness": sharpness.numpy(),
                f"{name}_mean_sharpness": mean_sharpness.numpy(),
            }
        )

    def check_histogram_sharpness(self, model, X: np.ndarray, n_bins: int = 50):
        if hasattr(model, "histogram_sharpness"):
            hist_sharpness, mean_hist_sharpness = model.histogram_sharpness(X)
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
                f"{name}_f_calibration": calibrations,
                f"{name}_f_calibration_mse": np.mean((p - calibrations) ** 2),
            }
        )

    def check_y_calibration(
        self,
        mus: np.ndarray,
        sigmas: np.ndarray,
        y: np.ndarray,
        name: str,
        n_bins: int = 50,
    ):
        p = np.linspace(0, 1, n_bins)
        norm_dists = tfp.distributions.Normal(loc=mus, scale=sigmas)
        # eq. (3) in "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
        indicators = y <= norm_dists.quantile(p)
        calibrations = np.mean(indicators, axis=0)
        self.summary.update(
            {
                "y_p_array": p,
                f"{name}_y_calibration": calibrations,
                f"{name}_y_calibration_mse": np.mean((calibrations - p) ** 2),
            }
        )

    def expected_log_predictive_density(
        self, mus: np.ndarray, sigmas: np.ndarray, y: np.ndarray, name: str,
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
        self.summary.update({f"{name}_elpd": elpd})

    def nmse(self, y: np.ndarray, predictions: np.ndarray, name: str):
        """Calculates normalized mean square error by 
        \ frac{1}{N\cdot\mathbb{V}[\textbf{y}]} \sum_i (\textbf{y}-\hat{\textbf{y}})^2
        where N is the length of y
        Args:
            y (np.ndarray): true outputs
            predictions (np.ndarray): model prediction (\hat{y})
            name (string): model name
        """
        mse = np.mean((y - predictions) ** 2)
        nmse = mse / np.var(y)
        self.summary.update({name + f"{name}_mse": mse})
        self.summary.update({name + f"{name}_nmse": nmse})

    def analyze(
        self,
        surrogate: Surrogate,
        dataset: Dataset,
        plot_it: bool = False,
        save_it: bool = True,
    ):
        X_test, y_test = dataset.sample_testset()
        mu_test, sigma_test = surrogate.predict(X_test)

        if self.d == 1 and plot_it:
            self.plot_predictive(
                self.X_test, mu_test, sigma_test, reg_name=surrogate.name, n_stds=3,
            )
        self.check_y_calibration(
            mu_test, sigma_test, y_test, name=surrogate.name,
        )
        # self.check_f_calibration(
        #     mu_test, sigma_test, dataset.data.f_test, name=surrogate.name,
        # )
        self.check_gaussian_sharpness(
            mu_test, sigma_test, name=surrogate.name,
        )
        self.check_histogram_sharpness(surrogate, X_test)
        self.expected_log_predictive_density(
            mu_test, sigma_test, y_test, name=surrogate.name,
        )
        self.nmse(y_test, mu_test, name=surrogate.name)

        if plot_it:
            self.plot_calibration_results()

        # Save
        if save_it:
            np.save(self.savepth + "summary---" + self.settings + ".npy", self.summary)
            print("Successfully saved with settings:", self.settings)

