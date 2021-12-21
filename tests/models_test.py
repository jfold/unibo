from os import mkdir
import unittest
from imports.ml import *
from surrogates.gaussian_process import GaussianProcess
from surrogates.random_forest import *
from surrogates.bayesian_neural_network import BayesianNeuralNetwork
from src.optimizer import Optimizer
from src.calibration import Calibration
from visualizations.scripts.calibrationplots import CalibrationPlots

kwargs = {"savepth": os.getcwd().replace("\\", "/") + "/results/tests/", "d": 1, "vanilla": True}


class ModelsTest(unittest.TestCase):
    def test_RandomForest(self) -> None:
        kwargs.update({"surrogate": "RF"})
        parameters = Parameters(kwargs, mkdir=True)
        dataset = Dataset(parameters)
        model = RandomForest(parameters, dataset)
        X_test, y_test = dataset.sample_testset(
            n_samples=parameters.n_evals + parameters.n_initial
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

    def test_visual_validate_bo_iter(self):
        for surrogate in ["GP", "RF", "BNN"]:
            kwargs.update({"surrogate": surrogate, "acquisition": "EI", "d": 1})
            parameters = Parameters(kwargs, mkdir=True)
            dataset = Dataset(parameters)
            optimizer = Optimizer(parameters)
            calibration = Calibration(parameters)
            X_test, y_test = dataset.sample_testset(n_samples=500)

            idx = np.argsort(X_test.squeeze())
            X_test = X_test[idx].squeeze()
            y_test = y_test[idx].squeeze()
            X_test_torch = torch.tensor(np.expand_dims(X_test[:, np.newaxis], 1))

            for e in range(10):
                optimizer.fit_surrogate(dataset)
                mus, sigmas = optimizer.surrogate_object.predict(X_test_torch)
                mus = mus.squeeze()
                sigmas = sigmas.squeeze()

                optimizer.construct_acquisition_function(dataset)
                ei = (
                    optimizer.acquisition_function(X_test_torch)
                    .detach()
                    .numpy()
                    .squeeze()
                )

                fig, ax1 = plt.subplots()
                ax2 = ax1.twinx()
                # acquisition
                ax1.plot(
                    X_test, ei, "-", color="blue", label="Acquisition",
                )
                # predictive
                ax2.plot(
                    X_test,
                    mus,
                    "--",
                    color="black",
                    label=r"$\mathcal{M}_{\mu}$",
                    linewidth=1,
                )
                ax2.fill_between(
                    X_test,
                    mus + 3 * sigmas,
                    mus - 3 * sigmas,
                    color="blue",
                    alpha=0.1,
                    label=r"$\mathcal{M}_{" + str(3) + "\sigma}$",
                )
                # data
                ax2.plot(X_test, y_test, "*", color="red", label="Test", alpha=0.2)
                ax2.plot(
                    dataset.data.X, dataset.data.y, "*", color="blue", label="Train"
                )
                ax1.set_ylabel("Acquistion value", color="blue")
                ax2.set_ylabel("y")
                x_new = X_test[[np.argmax(ei)], np.newaxis]
                y_new = dataset.data.get_y(x_new)
                ax2.plot(x_new, y_new, "*", color="black", label="New")
                dataset.add_X_get_y(x_new)
                plt.legend()
                fig.savefig(parameters.savepth + f"epoch---{e}.pdf")
                plt.close()

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
