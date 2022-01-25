from imports.general import *
from imports.ml import *


class Loader(object):
    def __init__(self, loadpths: list[str] = [], settings: Dict[str, str] = {}):
        self.loadpths = loadpths
        self.settings = settings
        self.load_metric_dict()
        self.load_params_tocheck()
        self.peak_data()
        self.load_data()
        self.extract()
        self.data_was_loaded = True if np.sum(np.isfinite(self.data)) > 0 else False

    def load_metric_dict(self):
        self.metric_dict = {
            "nmse": ["nMSE", -1, r"nMSE"],
            "elpd": ["ELPD", 1, r"ELPD"],
            "y_calibration_mse": [
                "Calibration MSE",
                -1,
                r"$ \mathbb{E}[( \mathcal{C}_{\mathbf{y}}(p) - p)^2] $",
            ],
            # "y_calibration_nmse": ["Calibration nMSE", -1,],
            "mean_sharpness": ["Sharpness", 1, r"$ \mathcal{S}$"],
            "x_opt_mean_dist": [
                "Solution mean distance",
                -1,
                r"$ \mathbb{E}[|| \mathbf{x}_o - \mathbf{x}_s ||_2] $",
            ],
            "x_opt_dist": [
                "Solution distance",
                -1,
                r"$ || \mathbf{x}_o - \mathbf{x}_s ||_2 $",
            ],
            "regret": ["Regret", -1, r"$ \mathcal{R}$"],
        }

    def load_params_tocheck(self):
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
                # parameters["n_evals"] = 90
                # json_dump = json.dumps(parameters)
                # with open(f"{experiment}parameters.json", "w") as f:
                #     f.write(json_dump)
                with open(f"{experiment}scores.json") as json_file:
                    scores = json.load(json_file)
                with open(f"{experiment}dataset.json") as json_file:
                    dataset = json.load(json_file)

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

    def load_data(self):
        self.init_data_object()
        for pth, parameters in self.data_settings.items():
            with open(f"{pth}scores.json") as json_file:
                scores = json.load(json_file)
            with open(f"{pth}dataset.json") as json_file:
                dataset = json.load(json_file)

            if not self.settings.items() <= parameters.items():
                continue

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
                self.data[tuple(data_idx)] = scores[metric]

            # Running over epochs
            files_in_path = [f for f in os.listdir(pth) if "scores---epoch" in f]
            for file in files_in_path:
                data_idx[-2] = int(
                    file.split("---epoch-")[-1].split(".json")[0]
                )  # epoch index
                with open(f"{pth}{file}") as json_file:
                    scores_epoch_i = json.load(json_file)
                for metric in self.metric_dict.keys():
                    data_idx[-1] = self.values[-1].index(metric)
                    self.data[tuple(data_idx)] = scores_epoch_i[metric]
        self.loader_summary.update(
            {"missing": np.sum(np.isnan(self.data)) / self.data.size}
        )
        self.loader_summary.update(
            {"missing": np.sum(np.isnan(self.data)) / self.data.size}
        )

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
