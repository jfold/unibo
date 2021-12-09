from os import mkdir
import unittest
from imports.ml import *
from surrogates.gaussian_process import GaussianProcess
from surrogates.random_forest import *
from surrogates.bayesian_neural_network import BayesianNeuralNetwork
from visualizations.scripts.calibrationplots import CalibrationPlots

kwargs = {"savepth": os.getcwd() + "/results/tests/", "d": 1, "vanilla": True}


class ModelsTest(unittest.TestCase):
    def test_RandomForest(self) -> None:
        kwargs.update({"surrogate": "RF"})
        parameters = Parameters(kwargs, mkdir=True)
        dataset = Dataset(parameters)
        model = RandomForest(parameters, dataset)
        X_test, y_test = dataset.sample_testset(
            n_samples=parameters.n_evals + parameters.n_evals
        )
        mu, std = model.predict(X_test)
        nentropies, mean_nentropy = model.histogram_sharpness(X_test)
        assert isinstance(model.model, RandomForestRegressor)
        assert isinstance(mu, np.ndarray) and mu.shape == y_test.shape
        assert isinstance(std, np.ndarray) and std.shape == y_test.shape
        assert isinstance(nentropies, np.ndarray)
        assert isinstance(mean_nentropy, float) and np.isfinite(mean_nentropy)

        plots = CalibrationPlots(parameters)
        plots.plot_predictive(dataset, X_test, y_test, mu, std)

    def test_RandomForest_bo_validation(self):
        pass

    def test_GaussianProcess(self) -> None:
        kwargs.update({"surrogate": "GP"})
        parameters = Parameters(kwargs, mkdir=True)
        dataset = Dataset(parameters)
        model = GaussianProcess(parameters, dataset)
        X_test, y_test = dataset.sample_testset(
            n_samples=parameters.n_evals + parameters.n_evals
        )
        mu, std = model.predict(X_test)
        assert isinstance(model.model, botorch.models.SingleTaskGP)
        assert isinstance(mu, np.ndarray) and mu.shape == y_test.shape
        assert isinstance(std, np.ndarray) and std.shape == y_test.shape

        plots = CalibrationPlots(parameters)
        plots.plot_predictive(dataset, X_test, y_test, mu, std)

    def test_BayesiaNeuralNetwork(self) -> None:
        kwargs.update({"surrogate": "BNN"})
        parameters = Parameters(kwargs, mkdir=True)
        dataset = Dataset(parameters)
        model = BayesianNeuralNetwork(parameters, dataset)
        X_test, y_test = dataset.sample_testset(
            n_samples=parameters.n_evals + parameters.n_evals
        )
        mu, std = model.predict(X_test)
        assert isinstance(model.model, nn.Sequential)
        assert isinstance(mu, np.ndarray) and mu.shape == y_test.shape
        assert isinstance(std, np.ndarray) and std.shape == y_test.shape

        plots = CalibrationPlots(parameters)
        plots.plot_predictive(dataset, X_test, y_test, mu, std)


if __name__ == "__main__":
    unittest.main()
