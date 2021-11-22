import unittest
from sklearn.datasets import make_regression
from surrogates.random_forest import *


class ModelsTest(unittest.TestCase):
    def test_RandomForest(self) -> None:
        parameters = Parameters({"n_train": 90, "n_test": 10, "D": 2})
        X, y = make_regression(n_samples=100, n_features=2)
        X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
        y = (y - np.nanmean(y, axis=0)) / np.nanstd(y, axis=0)
        y = y[:, np.newaxis]
        rf = RandomForest(parameters)
        rf.fit(X, y)
        mu, std = rf.predict(X)
        nentropies, mean_nentropy = rf.histogram_sharpness(X)
        assert isinstance(rf.model, RandomForestRegressor)
        assert isinstance(mu, np.ndarray) and mu.shape == y.shape
        assert isinstance(std, np.ndarray) and std.shape == y.shape
        assert isinstance(nentropies, np.ndarray)
        assert isinstance(mean_nentropy, float) and np.isfinite(mean_nentropy)


if __name__ == "__main__":
    unittest.main()
