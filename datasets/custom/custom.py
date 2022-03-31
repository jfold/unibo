from imports.general import *
from imports.ml import *
from src.parameters import Parameters


class CustomData(object):
    """Linear sum activation data generation class."""

    def __init__(self, dataset: Dict):
        self.X = dataset["X_train"]
        self.y = dataset["y_train"]
        self.X_test = dataset["X_test"]
        self.y_test = dataset["y_test"]
        if "f_train" in dataset.keys() and "f_test" in dataset.keys():
            self.f_train = dataset["f_train"]
            self.f_test = dataset["f_test"]
            self.f_min = np.min(self.f_test)
            self.f_max = np.max(self.f_test)

        self.y_min = np.min(self.y_test)
        self.y_max = np.max(self.y_test)

    def sample_X(self, n_samples):
        return self.X_test

    def get_y(self, X):
        return self.y_test
