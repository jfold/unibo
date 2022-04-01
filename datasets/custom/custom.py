from datasets.RBF.rbf_sampler import RBFSampler
from imports.general import *
from imports.ml import *
from src.parameters import Parameters


class CustomData(object):
    """Linear sum activation data generation class."""

    def __init__(self, parameters: Parameters, dataset: Dict = None):
        if dataset is None:
            dataset = RBFSampler(parameters).__dict__
        self.X = dataset["X_train"]
        self.y = dataset["y_train"]
        self.X_test = dataset["X_test"]
        self.y_test = dataset["y_test"]
        if "f_train" in dataset.keys() and "f_test" in dataset.keys():
            self.f_train = dataset["f_train"]
            self.f_test = dataset["f_test"]
            self.f_min = np.min(self.f_test)
            self.f_max = np.max(self.f_test)
        self.min_loc = self.X_test[[np.argmin(self.y_test)], :]
        self.y_min = np.min(self.y_test)
        self.y_max = np.max(self.y_test)
        self.ne_true = dataset["ne_true"]

    def sample_X(self, n_samples):
        return self.X_test

    def get_y(self, x_idx, parse_idx: bool = False):
        if parse_idx:
            return self.y_test[[x_idx], :]
        else:
            return self.y_test
