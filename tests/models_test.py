import unittest
from surrogates.gaussian_process import GaussianProcess
from surrogates.random_forest import *


class ModelsTest(unittest.TestCase):
    def test_RandomForest(self) -> None:
        parameters = Parameters({"n_evals": 90, "n_test": 10, "d": 2, "vanilla": True})
        dataset = Dataset(parameters)
        model = RandomForest(parameters, dataset)
        X_test, y_test = dataset.sample_testset(n_samples=100)
        mu, std = model.predict(X_test)
        nentropies, mean_nentropy = model.histogram_sharpness(X_test)
        assert isinstance(model.model, RandomForestRegressor)
        assert isinstance(mu, np.ndarray) and mu.shape == y_test.shape
        assert isinstance(std, np.ndarray) and std.shape == y_test.shape
        assert isinstance(nentropies, np.ndarray)
        assert isinstance(mean_nentropy, float) and np.isfinite(mean_nentropy)

    def test_GaussianProcess(self) -> None:
        parameters = Parameters({"d": 2, "vanilla": True})
        dataset = Dataset(parameters)
        model = GaussianProcess(parameters, dataset)
        print(model)

    def _test_BayesiaNeuralNetwork(self) -> None:
        raise NotImplementedError()


if __name__ == "__main__":
    unittest.main()
