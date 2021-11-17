import unittest
from sklearn.datasets import make_regression
from surrogates.random_forest import *


class RandomForestTest(unittest.TestCase):
    def test(self) -> None:
        parameters = Parameters(n_train=9, n_test=1, D=2)
        X, y = make_regression(n_samples=10, n_features=2)
        X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
        y = (y - np.nanmean(y, axis=0)) / np.nanstd(y, axis=0)
        rf = RandomForest(parameters)
        rf.fit(X, y)
        mu, var = rf.predict(X)
        nentropies, mean_nentropy = rf.histogram_sharpness(X)
        assert isinstance(rf.model, RandomForestRegressor)
        assert isinstance(mu, np.ndarray)  # and mu.shape == y.shape
        assert isinstance(var, np.ndarray)  # and var.shape == y.shape
        assert isinstance(nentropies, np.ndarray)
        assert isinstance(mean_nentropy, float)


if __name__ == "__main__":
    unittest.main()
