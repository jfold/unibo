from imports.general import *
from imports.ml import *
from src.dataset import Dataset
from src.parameters import Parameters


class Loader(object):
    def __init__(
        self,
        loadpth: str = "./results/",
        settings: Dict[str, str] = {},
        update: bool = True,
    ):
        self.loadpths = [f"{loadpth}{x}/" for x in os.listdir(loadpth)]
        self.settings = settings
        if update:
            self.load_data()
        else:
            try:
                self.load_from_file()
                self.load_metric_dict()
            except:
                print("No existing data file. Creates from scratch!")
                self.load_data()
        self.data_was_loaded = True if np.sum(np.isfinite(self.data)) > 0 else False

        self.savepth_figs = (
            os.getcwd()
            + "/visualizations/figures/"
            + str.join("-", [f"{key}-{val}-" for key, val in settings.items()])
        )
        self.savepth_tables = (
            os.getcwd()
            + "/visualizations/tables/"
            + str.join("-", [f"{key}-{val}-" for key, val in settings.items()])
        )

    def load_from_file(self):
        with open(os.getcwd() + "/results/metrics.pkl", "rb") as pkl:
            dict = pickle.load(pkl)
        for key, value in dict.items():
            setattr(self, key, value)

    def load_metric_dict(self):
        """These will be our final dimensionl in data array"""
        self.metric_dict = {
            "nmse": ["nMSE", -1, [0, 2], r"nMSE"],
            "elpd": ["ELPD", 1, [-5, 5], r"ELPD"],
            "mean_sharpness": ["Sharpness", 1, [-5, 5], r"$ \mathcal{S}$"],
            "sharpness_abs_error": ["Sharpness Error", 1, [], r"$ \mathcal{S}_e$"],
            "bias_mse": ["Bias", 1, [], r"$ \mathcal{S}$"],
            "x_opt_mean_dist": [
                "Solution mean distance",
                -1,
                [],
                r"$ \mathbb{E}[|| \mathbf{x}_o - \mathbf{x}_s ||_2] $",
            ],
            "y_regret": ["Regret on y", -1, [], r"$ \mathcal{R}_y$"],
            "f_regret": ["Regret on f", -1, [], r"$ \mathcal{R}_f$"],
            "true_regret": ["true_regret", -1, [], r"$ \mathcal{R}_t$"],
            "mahalanobis_dist": ["mahalanobis_dist", -1, [], r"$ D_M$"],
            "y_calibration_mse": [
                "Calibration MSE",
                -1,
                [],
                r"$ \mathbb{E}[( \mathcal{C}_{\mathbf{y}}(p) - p)^2] $",
            ],
            "uct-accuracy-corr": ["corr", 1, [0, 2], r"Corr"],
            "uct-avg_calibration-rms_cal": ["rms_cal", -1, [], r"RMSCE"],
            "uct-avg_calibration-miscal_area": ["miscal_area", -1, [], r"MA"],
            # "uct-adv_group_calibration-rms_adv_group_cal": [
            #     "rms_adv_group_cal",
            #     -1,
            #     [],
            #     r"RMSACE",
            # ],
            "uct-scoring_rule-nll": ["nll", -1, [], r"NLL"],
        }

    def define_check_params(self):
        """These will be our D-1 dimensions and constitute experimental grid.
        All data loaded will have these keys in their parameters.json file """
        self.check_params = [
            "seed",
            "d",
            "n_test",
            "snr",
            "n_initial",
            "n_evals",
            "problem",
            "data_name",
            "change_std",
            "surrogate",
            "acquisition",
            "bo",
        ]

    def load_data(self, save: bool = True):
        self.load_metric_dict()
        self.define_check_params()
        self.peak_data()
        self.init_data_object()
        for pth, parameters in self.data_settings.items():
            if not self.settings.items() <= parameters.items():
                continue

            with open(f"{pth}metrics.json") as json_file:
                metrics = json.load(json_file)
            with open(f"{pth}dataset.json") as json_file:
                dataset = json.load(json_file)

            if os.path.isfile(f"{pth}metrics-uct.pkl"):
                with open(f"{pth}metrics-uct.pkl", "rb") as pkl:
                    uct_scores = pickle.load(pkl)
            else:
                uct_scores = None

            params_idx = [
                self.values[i].index(parameters[key])
                for i, key in enumerate(self.names)
                if key in self.check_params
            ]
            data_idx = params_idx
            data_idx.extend(
                [parameters["n_evals"], None]
            )  # since we have added "epoch" and "metrics" on top of parameters

            for metric in self.metric_dict.keys():
                data_idx[-1] = self.values[-1].index(metric)
                if metric in metrics:
                    self.data[tuple(data_idx)] = metrics[metric]
                elif "uct-" in metric and uct_scores is not None:
                    entries = metric.split("-")
                    self.data[tuple(data_idx)] = uct_scores[entries[1]][entries[2]]

            # Running over epochs
            files_in_path = [
                f for f in os.listdir(pth) if "metrics---epoch" in f and ".json" in f
            ]
            for file in files_in_path:
                # epoch index
                data_idx[-2] = int(file.split("---epoch-")[-1].split(".json")[0])

                with open(f"{pth}{file}") as json_file:
                    scores_epoch_i = json.load(json_file)

                file = file.replace("metrics---", "metrics-uct---").replace(
                    ".json", ".pkl"
                )
                if os.path.isfile(f"{pth}{file}"):
                    with open(f"{pth}{file}", "rb") as pkl:
                        uct_scores_epoch_i = pickle.load(pkl)
                else:
                    uct_scores_epoch_i = None

                for metric in self.metric_dict.keys():
                    data_idx[-1] = self.values[-1].index(metric)
                    if metric in scores_epoch_i:
                        self.data[tuple(data_idx)] = scores_epoch_i[metric]
                    elif "uct-" in metric and uct_scores_epoch_i is not None:
                        entries = metric.split("-")
                        self.data[tuple(data_idx)] = uct_scores_epoch_i[entries[1]][
                            entries[2]
                        ]

        if save:
            with open(os.getcwd() + "/results/metrics.pkl", "wb") as pkl:
                pickle.dump(self.__dict__, pkl)

    def extract(
        self, data: np.ndarray = None, settings: Dict[str, list] = {}
    ) -> np.ndarray:
        """Example: 
        >>> extract(settings = {"bo": [True]})
        Returns all the data where "bo" is true
        """
        del_idx_dict = {
            k: {
                "axis": self.names.index(k),
                "idxs": self.values[self.names.index(k)],
                "removes": [],
            }
            for k in settings.keys()
        }
        data = self.data if data is None else data
        for k, vals in settings.items():
            bool_arr = np.ones(len(del_idx_dict[k]["idxs"]), dtype=bool)
            vals = vals if type(vals) is list else [vals]
            for val in vals:
                del_idx = np.array(np.array(del_idx_dict[k]["idxs"]) != val, dtype=bool)
                bool_arr = np.logical_and(bool_arr, del_idx)

            idxs = list(reversed(sorted(np.argwhere(bool_arr.squeeze()).tolist())))
            axis = del_idx_dict[k]["axis"]
            for idx in idxs:
                data = np.delete(data, idx, axis=axis)
        return data

    def remove_nans(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert x.shape == y.shape
        is_nan_idx = np.isnan(x)
        y = y[np.logical_not(is_nan_idx)]
        x = x[np.logical_not(is_nan_idx)]
        is_nan_idx = np.isnan(y)
        x = x[np.logical_not(is_nan_idx)]
        y = y[np.logical_not(is_nan_idx)]
        return x, y

    def remove_extremes(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert x.shape == y.shape
        remove_idx = y > np.quantile(y, 0.95)
        y = y[np.logical_not(remove_idx)]
        x = x[np.logical_not(remove_idx)]
        remove_idx = y < np.quantile(y, 0.05)
        x = x[np.logical_not(remove_idx)]
        y = y[np.logical_not(remove_idx)]
        return x, y

    def find_extreme_idx(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        extreme_high_idx = x > np.quantile(x, 0.95)
        extreme_low_idx = x < np.quantile(x, 0.05)
        return extreme_high_idx, extreme_low_idx

    def find_extreme_vals(self, x: np.ndarray, q: float = 0.05) -> Tuple[float, float]:
        return np.quantile(x, 1 - q), np.quantile(x, q)

    def save_to_tex(self, df: pd.DataFrame, name: str):
        with open(f"{self.savepth_tables}/{name}.tex", "w") as file:
            file.write(df.to_latex(escape=False))
        file.close()

    def peak_data(self):
        self.data_settings = {}
        self.experiments_summary = {k: [] for k in self.check_params}
        for loadpth in self.loadpths:
            if (
                os.path.isdir(loadpth)
                and os.path.isfile(f"{loadpth}parameters.json")
                and os.path.isfile(f"{loadpth}metrics.json")
                and os.path.isfile(f"{loadpth}dataset.json")
            ):
                with open(f"{loadpth}parameters.json") as json_file:
                    parameters = json.load(json_file)
                if not self.settings.items() <= parameters.items() or not all(
                    param in parameters for param in self.check_params
                ):
                    continue
                self.data_settings.update(
                    {loadpth: {k: parameters[k] for k in self.check_params}}
                )

                for k in self.check_params:
                    if k in parameters.keys():
                        lst = self.experiments_summary[k]
                        if parameters[k] not in lst:
                            lst.append(parameters[k])
                            self.experiments_summary.update({k: lst})

                # sort
                self.values = []
                self.dims = []
                self.names = []
                for key, val in self.experiments_summary.items():
                    if len(val) > 0:
                        lst = sorted(val)
                        self.experiments_summary.update({key: lst})
                        self.values.append(sorted(set(val)))
                        self.dims.append(len(self.values[-1]))
                        self.names.append(key)
                    else:
                        raise ValueError(f"Key {key} empty!")

    def init_data_object(self):
        self.values.append(
            list(range(1 + int(np.max(self.values[self.names.index("n_evals")]))))
        )
        self.dims.append(len(self.values[-1]))
        self.names.append("epoch")

        self.values.append(list(self.metric_dict.keys()))
        self.dims.append(len(self.values[-1]))
        self.names.append("metric")
        self.loader_summary = {
            self.names[i]: {"d": self.dims[i], "axis": i, "vals": self.values[i]}
            for i in range(len(self.values))
        }
        self.data = np.full(tuple(self.dims), np.nan)

