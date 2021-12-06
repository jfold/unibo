import unittest
from sklearn.datasets import make_regression
from surrogates.random_forest import *


class ModelsTest(unittest.TestCase):
    def test_RandomForest(self) -> None:
        parameters = Parameters({"n_evals": 90, "n_test": 10, "d": 2, "vanilla": True})
        dataset = Dataset(parameters)
        rf = RandomForest(parameters, dataset)
        X_test, y_test = dataset.sample_testset(n_samples=100)
        mu, std = rf.predict(X_test)
        nentropies, mean_nentropy = rf.histogram_sharpness(X_test)
        assert isinstance(rf.model, RandomForestRegressor)
        assert isinstance(mu, np.ndarray) and mu.shape == y_test.shape
        assert isinstance(std, np.ndarray) and std.shape == y_test.shape
        assert isinstance(nentropies, np.ndarray)
        assert isinstance(mean_nentropy, float) and np.isfinite(mean_nentropy)


if __name__ == "__main__":
    unittest.main()
