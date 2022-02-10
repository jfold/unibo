from imports.general import *
from imports.ml import *
from src.dataset import Dataset
from src.parameters import Parameters


class Loader(object):
    def __init__(
        self,
        loadpths: list[str] = [],
        settings: Dict[str, str] = {},
        update: bool = True,
    ):
        self.loadpths = loadpths
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

    def load_from_file(self):
        with open(os.getcwd() + "/metrics.pkl", "rb") as pkl:
            dict = pickle.load(pkl)
        for key, value in dict.items():
            setattr(self, key, value)

    def load_metric_dict(self):
        """These will be our final dimensionl in data array"""
        self.metric_dict = {
            "nmse": ["nMSE", -1, [0, 2], r"nMSE"],
            "elpd": ["ELPD", 1, [-5, 5], r"ELPD"],
            "mean_sharpness": ["Sharpness", 1, [-5, 5], r"$ \mathcal{S}$"],
            "x_opt_mean_dist": [
                "Solution mean distance",
                -1,
                [],
                r"$ \mathbb{E}[|| \mathbf{x}_o - \mathbf{x}_s ||_2] $",
            ],
            "regret": ["Regret", -1, [], r"$ \mathcal{R}$"],
            "true_regret": ["true_regret", -1, [], r"$ \mathcal{R}_t$"],
            "mahalanobis_dist": ["mahalanobis_dist", -1, [], r"$ D_M$"],
            "running_inner_product": ["running_inner_product", -1, [], r"$ RC$"],
            "y_calibration_mse": [
                "Calibration MSE",
                -1,
                [],
                r"$ \mathbb{E}[( \mathcal{C}_{\mathbf{y}}(p) - p)^2] $",
            ],
            # uct module:
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

    def load_params_tocheck(self):
        """These will be our D-1 dimensions and constitute experimental grid"""
        self.check_params = [
            "seed",
            "d",
            "n_test",
            "n_initial",
            "n_evals",
            "problem",
            "change_std",
            "surrogate",
            "acquisition",
            "bo",
        ]

    def peak_data(self):
        self.data_settings = {}
        self.data_summary = {k: [] for k in self.check_params}
        for i_e, experiment in enumerate(p for p in self.loadpths):
            if (
                os.path.isdir(experiment)
                and os.path.isfile(f"{experiment}parameters.json")
                and os.path.isfile(f"{experiment}scores.json")
                and os.path.isfile(f"{experiment}dataset.json")
            ):
                with open(f"{experiment}parameters.json") as json_file:
                    parameters = json.load(json_file)

                if not self.settings.items() <= parameters.items() or not all(
                    param in parameters for param in self.check_params
                ):
                    continue

                self.data_settings.update(
                    {experiment: {k: parameters[k] for k in self.check_params}}
                )

                for k in self.check_params:
                    if k in parameters.keys():
                        lst = self.data_summary[k]
                        lst.append(parameters[k])
                        self.data_summary.update({k: lst})

                # if i_e > 20: # for debugging
                #     break

    def init_data_object(self):
        self.values = []
        self.dims = []
        self.names = []
        for key, val in self.data_summary.items():
            self.values.append(sorted(set(val)))
            self.dims.append(len(self.values[-1]))
            self.names.append(key)

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

    def calc_true_regret(self, parameters: Dict, dataset: Dict):
        # inferring true regret
        dataobj = Dataset(Parameters(parameters))
        X = np.array(dataset["X"])
        y_clean = dataobj.data.get_y(X, add_noise=False)
        y_clean = np.array([np.min(y_clean[:i]) for i in range(1, len(y_clean) + 1)])
        y_clean = y_clean[dataset["n_initial"] - 1 :]
        true_regrets = np.abs(dataobj.data.f_min - y_clean)
        return true_regrets

    def calc_running_inner_product(self, dataset: Dict) -> np.ndarray:
        X = np.array(dataset["X"])
        running_inner_product = np.cumsum(np.diag(X @ X.T)).squeeze()[
            dataset["n_initial"] - 1 :
        ]
        return running_inner_product

    def calc_mahalanobis_dist_to_current_best(self, dataset: Dict) -> np.ndarray:
        X = np.array(dataset["X"])
        y = np.array(dataset["y"])
        n_initial = dataset["n_initial"]
        Sigma = np.cov(X, rowvar=False)
        idx_opt = np.argmin(y[:n_initial, :])
        y_opt = y[[idx_opt], :]
        X_opt = X[[idx_opt], :].T
        mahalanobis_dists = [np.nan]
        for i in range(n_initial, X.shape[0]):
            cur_y = y[[i], :]
            cur_X = X[[i], :].T
            mahalanobis_dists.append(mahalanobis(cur_X, X_opt, Sigma))
            if cur_y < y_opt:
                y_opt = cur_y
                X_opt = cur_X
        return np.array(mahalanobis_dists)

    def load_data(self, save: bool = True):
        self.load_metric_dict()
        self.load_params_tocheck()
        self.peak_data()
        self.init_data_object()
        for pth, parameters in self.data_settings.items():
            if not self.settings.items() <= parameters.items():
                continue

            with open(f"{pth}scores.json") as json_file:
                scores = json.load(json_file)
            with open(f"{pth}dataset.json") as json_file:
                dataset = json.load(json_file)

            true_regrets = self.calc_true_regret(parameters, dataset)
            running_inner_product = self.calc_running_inner_product(dataset)
            mahalanobis_dists = self.calc_mahalanobis_dist_to_current_best(dataset)

            if os.path.isfile(f"{pth}scores-uct.pkl"):
                with open(f"{pth}scores-uct.pkl", "rb") as pkl:
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
                if metric in scores:
                    self.data[tuple(data_idx)] = scores[metric]
                elif "uct-" in metric and uct_scores is not None:
                    entries = metric.split("-")
                    self.data[tuple(data_idx)] = uct_scores[entries[1]][entries[2]]
                elif metric == "true_regret":
                    self.data[tuple(data_idx)] = true_regrets[-1]
                elif metric == "mahalanobis_dist":
                    self.data[tuple(data_idx)] = mahalanobis_dists[-1]
                elif metric == "running_inner_product":
                    self.data[tuple(data_idx)] = running_inner_product[-1]

            # Running over epochs
            files_in_path = [
                f for f in os.listdir(pth) if "scores---epoch" in f and ".json" in f
            ]
            for file in files_in_path:
                # epoch index
                data_idx[-2] = int(file.split("---epoch-")[-1].split(".json")[0])

                with open(f"{pth}{file}") as json_file:
                    scores_epoch_i = json.load(json_file)

                file = file.replace("scores---", "scores-uct---").replace(
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
                    elif metric == "true_regret":
                        self.data[tuple(data_idx)] = true_regrets[data_idx[-2]]
                    elif metric == "mahalanobis_dist":
                        self.data[tuple(data_idx)] = mahalanobis_dists[data_idx[-2]]
                    elif metric == "running_inner_product":
                        self.data[tuple(data_idx)] = running_inner_product[data_idx[-2]]

        if save:
            with open(os.getcwd() + "/metrics.pkl", "wb") as pkl:
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
