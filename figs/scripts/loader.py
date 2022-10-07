from sqlite3 import Connection
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
        self.load_attributes()

    def load_attributes(self):
        self.metric_dict = {
            "nmse": ["nMSE", -1, [0, 2], "nMSE"],
            "elpd": ["ELPD", 1, [-5, 5], "$\mathcal{L}$"],
            "mean_sharpness": ["Sharpness", 1, [-5, 5], "$\mathcal{S}$"],
            "sharpness_error_true_minus_model": [
                "Sharpness Error",
                1,
                [],
                "$\mathcal{S}_E$",
            ],
            "bias_mse": ["Bias", 1, [], "$\mathcal{E}$"],
            "y_regret": ["Instant regret on y", -1, [], "$\mathcal{R}_y^I$"],
            "f_regret": ["Instant regret on f", -1, [], "$\mathcal{R}_f^I$"],
            "y_regret_total": ["Total regret on y", -1, [], "$\mathcal{R}_y^T$"],
            "f_regret_total": ["Total regret on f", -1, [], "$\mathcal{R}_f^T$"],
            "std_change": ["Std. change", -1, [], "$c$"],
            "mahalanobis_dist": ["mahalanobis_dist", -1, [], "$D_M$"],
            "y_calibration_mse": ["Calibration MSE", -1, [], "$E_{\mathcal{C}_y}$",],
            "y_calibration_over": [
                "Over calibration",
                -1,
                [],
                "$E_{\mathcal{C}^+_y}$",
            ],
            "y_calibration_under": [
                "Under calibration",
                -1,
                [],
                "$E_{\mathcal{C}^-_y}$",
            ],
            "uct_calibration": ["UCT-C", 1, [0, 2], "UCT-C"],
            "uct_sharpness": ["UCT-S", -1, [], "UCT-S"],
        }
        self.check_params = [
            "seed",
            "d",
            "n_test",
            "snr",
            "n_initial",
            "n_evals",
            "problem",
            "data_name",
            "recalibrate",
            "recal_mode",
            "n_validation",
            "std_change",
            "surrogate",
            "acquisition",
            "bo",
        ]

    def delete_sql_table(self, cnx: Connection, table_name: str):
        cur = cnx.cursor()
        cur.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        )
        table_exists = len(cur.fetchone()) == 1
        if table_exists:
            cur.execute(f"DELETE FROM {table_name};",)
            cnx.commit()

    def path2sql(
        self,
        folders: list = ["results_regret_vs_calibration",],
        db: str = "./results.db",
        delete_existing: bool = False,
    ):
        columns = np.unique(np.append(self.check_params, list(self.metric_dict.keys())))
        cnx = sqlite3.connect(db)

        df = pd.DataFrame(columns=columns)
        for loaddir in folders:
            loadpths = [
                f"./{loaddir}/{x}/"
                for x in os.listdir(f"./{loaddir}/")
                if os.path.isdir(f"./{loaddir}/{x}/")
            ]
            if delete_existing:
                self.delete_sql_table(cnx, loaddir)

            for i_p, pth in tqdm(enumerate(loadpths), total=len(loadpths)):
                if not os.path.isfile(f"{pth}parameters.json") or not os.path.isfile(
                    f"{pth}metrics.json"
                ):
                    print(f"No file(s) in {pth}")
                    continue

                with open(f"{pth}parameters.json") as json_file:
                    parameters = json.load(json_file)
                with open(f"{pth}metrics.json") as json_file:
                    metrics = json.load(json_file)

                data_params = {}
                for x in self.check_params:
                    if x in parameters.keys():
                        data_params.update({x: parameters[x]})
                    else:
                        data_params.update({x: "NULL"})

                data = dict(data_params)
                data.update({"epoch": parameters["n_evals"]})
                for x in self.metric_dict:
                    if x in metrics.keys():
                        m = (
                            [metrics[x]]
                            if not isinstance(metrics[x], list)
                            else metrics[x]
                        )
                    if x in metrics.keys() and len(m) > 0 and np.isfinite(m[-1]):
                        data.update({x: m[-1]})
                        if x == "y_regret" or x == "f_regret":
                            data.update({f"{x}_total": np.sum(m)})
                    # elif x == "y_calibration_over":
                    #     data.update({x: self.calibration(metrics, "over")})
                    # elif x == "y_calibration_under":
                    #     data.update({x: self.calibration(metrics, "under")})
                    elif x not in data and x not in parameters.keys():
                        data.update({x: "NULL"})

                df = df.append(data, ignore_index=True)

                if (i_p != 0 and i_p % 4000 == 0) or i_p == len(loadpths) - 1:
                    df.to_sql(loaddir, cnx, if_exists="append")
                    df = pd.DataFrame(columns=columns)

        cnx.close()

    def dict2query(
        self,
        FROM: str = "",
        SELECT: list = None,
        WHERE: Dict = {},
        ORDERBY: list = None,
    ) -> str:
        cols = ",".join(SELECT) if SELECT is not None else "*"
        query = f"SELECT {cols} FROM {FROM}"
        c = 1
        dict_len = len(WHERE.keys())
        if dict_len > 0:
            query += " WHERE "
        for key, val in WHERE.items():
            if isinstance(val, bool):
                val = 1 if val else 0
            query += f"{key}='{val}'"
            if c < dict_len:
                query += " and "
            c += 1
        if ORDERBY is not None and len(ORDERBY) > 0:
            query += " ORDER BY " + ",".join(ORDERBY)
        return query + ";"

    def p2stars(self, p: float, bonfferoni: float = 1.0):
        assert 0.0 <= p <= 1.0
        stars = ""
        p *= bonfferoni
        if p < 0.05:
            stars = "^{*}"
        if p < 1e-10:
            stars = "^{**}"
        return stars

    def merge_two_dicts(self, x, y):
        z = x.copy()  # start with keys and values of x
        z.update(y)  # modifies z with keys and values of y
        return z

    def format_num(self, x: float):
        if np.abs(x) < 10 ** (-3) and self.scientific_notation:
            return (
                f"{x:.1E}".replace("E-0", "E-")
                .replace("E+0", "E+")
                .replace("E-", "\cdot 10^{-")
                .replace("E+", "10^{")
                + "}"
            )
        else:
            return f"{x:.3f}"

    def calibration(self, metrics, over_under: str):
        if over_under == "under":
            E_cal_minus = np.nanmean(
                np.maximum(0, metrics["y_calibration"] - metrics["p_array"]) ** 2
            )
            return E_cal_minus
        elif over_under == "over":
            E_cal_plus = np.nanmean(
                np.maximum(0, metrics["p_array"] - metrics["y_calibration"]) ** 2
            )
            return E_cal_plus

    # def load_from_file(self):
    #     with open(os.getcwd() + f"/{self.loadpth}/metrics.pkl", "rb") as pkl:
    #         dict = pickle.load(pkl)
    #     for key, value in dict.items():
    #         setattr(self, key, value)
    # def load_data(self, save: bool = True):
    #     self.load_metric_dict()
    #     self.define_check_params()
    #     self.peak_data()
    #     self.init_data_object()
    #     for pth, parameters in self.data_settings.items():
    #         if not self.settings.items() <= parameters.items():
    #             continue

    #         with open(f"{pth}metrics.json") as json_file:
    #             metrics = json.load(json_file)
    #         with open(f"{pth}dataset.json") as json_file:
    #             dataset = json.load(json_file)

    #         if os.path.isfile(f"{pth}metrics-uct.pkl"):
    #             with open(f"{pth}metrics-uct.pkl", "rb") as pkl:
    #                 uct_scores = pickle.load(pkl)
    #         else:
    #             uct_scores = None

    #         params_idx = [
    #             self.values[i].index(parameters[key])
    #             for i, key in enumerate(self.names)
    #             if key in self.check_params
    #         ]
    #         data_idx = params_idx
    #         data_idx.extend(
    #             [parameters["n_evals"], None]
    #         )  # since we have added "epoch" and "metrics" on top of parameters

    #         for metric in self.metric_dict.keys():
    #             data_idx[-1] = self.values[-1].index(metric)
    #             if metric in metrics:
    #                 self.data[tuple(data_idx)] = metrics[metric]
    #             elif "uct-" in metric and uct_scores is not None:
    #                 entries = metric.split("-")
    #                 self.data[tuple(data_idx)] = uct_scores[entries[1]][entries[2]]

    #         # Running over epochs
    #         files_in_path = [
    #             f for f in os.listdir(pth) if "metrics---epoch" in f and ".json" in f
    #         ]
    #         for file in files_in_path:
    #             # epoch index
    #             data_idx[-2] = int(file.split("---epoch-")[-1].split(".json")[0])

    #             with open(f"{pth}{file}") as json_file:
    #                 scores_epoch_i = json.load(json_file)

    #             file = file.replace("metrics---", "metrics-uct---").replace(
    #                 ".json", ".pkl"
    #             )
    #             if os.path.isfile(f"{pth}{file}"):
    #                 with open(f"{pth}{file}", "rb") as pkl:
    #                     uct_scores_epoch_i = pickle.load(pkl)
    #             else:
    #                 uct_scores_epoch_i = None

    #             for metric in self.metric_dict.keys():
    #                 data_idx[-1] = self.values[-1].index(metric)
    #                 if metric in scores_epoch_i:
    #                     self.data[tuple(data_idx)] = scores_epoch_i[metric]
    #                 elif "uct-" in metric and uct_scores_epoch_i is not None:
    #                     entries = metric.split("-")
    #                     self.data[tuple(data_idx)] = uct_scores_epoch_i[entries[1]][
    #                         entries[2]
    #                     ]

    #     if save:
    #         with open(os.getcwd() + f"/{self.loadpth}/metrics.pkl", "wb") as pkl:
    #             pickle.dump(self.__dict__, pkl)

    # def extract(
    #     self,
    #     data: np.ndarray = None,
    #     settings: Dict[str, list] = {},
    #     return_values: bool = False,
    # ) -> np.ndarray:
    #     """Example:
    #     >>> extract(settings = {"bo": [True]})
    #     Returns all the data where "bo" is true
    #     """
    #     del_idx_dict = {
    #         k: {
    #             "axis": self.names.index(k),
    #             "idxs": self.values[self.names.index(k)],
    #             "removes": [],
    #         }
    #         for k in settings.keys()
    #     }
    #     data = self.data if data is None else data
    #     values = []
    #     for k, vals in settings.items():
    #         bool_arr = np.ones(len(del_idx_dict[k]["idxs"]), dtype=bool)
    #         vals = vals if type(vals) is list else [vals]
    #         for val in vals:
    #             del_idx = np.array(np.array(del_idx_dict[k]["idxs"]) != val, dtype=bool)
    #             bool_arr = np.logical_and(bool_arr, del_idx)

    #         idxs = list(reversed(sorted(np.argwhere(bool_arr.squeeze()).tolist())))
    #         axis = del_idx_dict[k]["axis"]
    #         values_ = self.loader_summary[k]["vals"]
    #         for idx in idxs:
    #             data = np.delete(data, idx, axis=axis)
    #             values_ = np.delete(values_, idx)
    #         values.append(values_)
    #     if return_values:
    #         return data, values[-1]
    #     return data

    # def remove_nans(
    #     self, x: np.ndarray, y: np.ndarray
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     assert x.shape == y.shape
    #     is_nan_idx = np.isnan(x)
    #     y = y[np.logical_not(is_nan_idx)]
    #     x = x[np.logical_not(is_nan_idx)]
    #     is_nan_idx = np.isnan(y)
    #     x = x[np.logical_not(is_nan_idx)]
    #     y = y[np.logical_not(is_nan_idx)]
    #     return x, y

    # def remove_extremes(
    #     self, x: np.ndarray, y: np.ndarray
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     assert x.shape == y.shape
    #     remove_idx = y > np.quantile(y, 0.95)
    #     y = y[np.logical_not(remove_idx)]
    #     x = x[np.logical_not(remove_idx)]
    #     remove_idx = y < np.quantile(y, 0.05)
    #     x = x[np.logical_not(remove_idx)]
    #     y = y[np.logical_not(remove_idx)]
    #     return x, y

    # def find_extreme_idx(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     extreme_high_idx = x > np.quantile(x, 0.95)
    #     extreme_low_idx = x < np.quantile(x, 0.05)
    #     return extreme_high_idx, extreme_low_idx

    # def find_extreme_vals(self, x: np.ndarray, q: float = 0.05) -> Tuple[float, float]:
    #     return np.quantile(x, 1 - q), np.quantile(x, q)

    # # def save_to_tex(self, df: pd.DataFrame, name: str):
    # #     with open(f"{self.savepth_tables}/{name}.tex", "w") as file:
    # #         file.write(df.to_latex(escape=False))
    # #     file.close()

    # def peak_data(self):
    #     self.data_settings = {}
    #     self.experiments_summary = {k: [] for k in self.check_params}
    #     self.values = []
    #     self.dims = []
    #     self.names = []
    #     for loadpth in self.loadpths:
    #         if (
    #             os.path.isdir(loadpth)
    #             and os.path.isfile(f"{loadpth}parameters.json")
    #             and os.path.isfile(f"{loadpth}metrics.json")
    #             and os.path.isfile(f"{loadpth}dataset.json")
    #         ):
    #             with open(f"{loadpth}parameters.json") as json_file:
    #                 parameters = json.load(json_file)
    #             if not self.settings.items() <= parameters.items() or not all(
    #                 param in parameters for param in self.check_params
    #             ):
    #                 continue
    #             self.data_settings.update(
    #                 {loadpth: {k: parameters[k] for k in self.check_params}}
    #             )

    #             for k in self.check_params:
    #                 if k in parameters.keys():
    #                     lst = self.experiments_summary[k]
    #                     if parameters[k] not in lst:
    #                         lst.append(parameters[k])
    #                         self.experiments_summary.update({k: lst})

    #             for key, val in self.experiments_summary.items():
    #                 print(key, val)
    #                 raise ValueError()
    #                 if len(val) > 0:
    #                     lst = sorted(val)
    #                     self.experiments_summary.update({key: lst})
    #                     self.values.append(sorted(set(val)))
    #                     self.dims.append(len(self.values[-1]))
    #                     self.names.append(key)
    #                 else:
    #                     raise ValueError(f"Key {key} empty!")

    # def init_data_object(self):
    #     self.values.append(
    #         list(range(1 + int(np.max(self.values[self.names.index("n_evals")]))))
    #     )
    #     self.dims.append(len(self.values[-1]))
    #     self.names.append("epoch")

    #     self.values.append(list(self.metric_dict.keys()))
    #     self.dims.append(len(self.values[-1]))
    #     self.names.append("metric")
    #     self.loader_summary = {
    #         self.names[i]: {"d": self.dims[i], "axis": i, "vals": self.values[i]}
    #         for i in range(len(self.values))
    #     }
    #     self.data = np.full(tuple(self.dims), np.nan)

