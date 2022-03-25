from os import mkdir
import unittest
from imports.ml import *
from surrogates.deep_ensemble import DeepEnsemble
from surrogates.dummy_surrogate import DummySurrogate
from surrogates.gaussian_process import GaussianProcess
from surrogates.random_forest import *
from surrogates.bayesian_neural_network import BayesianNeuralNetwork
from src.optimizer import Optimizer
from src.calibration import Calibration
from src.experiment import Experiment
from visualizations.scripts.calibrationplots import CalibrationPlots

kwargs = {"savepth": os.getcwd() + "/results/tests/", "d": 1, "vanilla": True}


class ModelsTest(unittest.TestCase):
    def test_visual_validate_random_search(self):
        for surrogate in ["GP", "RF", "BNN"]:
            kwargs.update(
                {
                    "surrogate": surrogate,
                    "acquisition": "RS",
                    "d": 1,
                    "problem": "Csendes",
                }
            )
            parameters = Parameters(kwargs, mkdir=True)
            dataset = Dataset(parameters)
            optimizer = Optimizer(parameters)
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
                ax1.plot(X_test, ei, "-", color="blue", label="Acquisition", alpha=0.3)
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

    def plot_1d(self, save_settings: str = ""):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        # acquisition
        ax1.plot(
            self.X_test, self.ei, "-", color="blue", label="Acquisition",
        )
        # predictive
        ax2.plot(
            self.X_test,
            self.mus,
            "--",
            color="black",
            label=r"$\mathcal{M}_{\mu}$",
            linewidth=1,
        )
        ax2.fill_between(
            self.X_test,
            self.mus + 3 * self.sigmas,
            self.mus - 3 * self.sigmas,
            color="blue",
            alpha=0.1,
            label=r"$\mathcal{M}_{" + str(3) + "\sigma}$",
        )
        # data
        ax2.plot(self.X_test, self.y_test, "*", color="red", label="Test", alpha=0.2)
        ax2.plot(
            self.dataset.data.X, self.dataset.data.y, "*", color="blue", label="Train"
        )
        ax1.set_ylabel("Acquistion value", color="blue")
        ax2.set_ylabel("y")
        x_new = self.X_test[[np.argmax(self.ei)], np.newaxis]
        y_new = self.dataset.data.get_y(x_new)
        ax2.plot(x_new, y_new, "*", color="black", label="New")
        self.dataset.add_X_get_y(x_new)
        plt.legend()
        fig.savefig(f"{self.parameters.savepth}predictive{save_settings}.pdf")
        plt.close()

    def test_visual_validate_bo_iter(self):
        for surrogate in ["DS", "GP", "RF", "BNN"]:
            kwargs.update(
                {
                    "surrogate": surrogate,
                    "acquisition": "EI",
                    "d": 1,
                    "problem": "Csendes",
                }
            )
            self.parameters = Parameters(kwargs, mkdir=True)
            self.dataset = Dataset(self.parameters)
            self.optimizer = Optimizer(self.parameters)
            X_test, y_test = self.dataset.sample_testset(n_samples=500)

            idx = np.argsort(X_test.squeeze())
            self.X_test = X_test[idx].squeeze()
            self.y_test = y_test[idx].squeeze()
            self.X_test_torch = torch.tensor(
                np.expand_dims(self.X_test[:, np.newaxis], 1)
            )

            self.optimizer.fit_surrogate(self.dataset)
            self.dataset.save(save_settings="---epoch-0")
            n_epocs = 10
            for e in range(n_epocs):
                save_settings = f"---epoch-{e+1}" if e < n_epocs - 1 else ""
                x_new, acq_val = self.optimizer.bo_iter(self.dataset)
                self.dataset.add_X_get_y(x_new, acq_val)
                self.optimizer.fit_surrogate(self.dataset)
                self.ei = (
                    self.optimizer.acquisition_function(self.X_test_torch)
                    .detach()
                    .numpy()
                    .squeeze()
                )
                mus, sigmas = self.optimizer.surrogate_object.predict(self.X_test_torch)
                self.mus = mus.squeeze()
                self.sigmas = sigmas.squeeze()
                if self.parameters.d == 1:
                    self.plot_1d(save_settings)
            self.dataset.save()

    def test_GaussianProcess(self) -> None:
        kwargs.update({"surrogate": "GP", "snr": 5, "n_initial": 100})
        parameters = Parameters(kwargs, mkdir=True)
        dataset = Dataset(parameters)
        model = GaussianProcess(parameters, dataset)
        print(dataset.data.noise_var)
        print(model.model.likelihood.noise.cpu().detach().numpy().squeeze())
        print(model.model.covar_module.lengthscale.cpu().detach().numpy().squeeze())
        X_test, y_test = dataset.sample_testset(
            n_samples=parameters.n_evals + parameters.n_evals
        )
        mu, std = model.predict(X_test)
        assert isinstance(model.model, botorch.models.SingleTaskGP)
        assert isinstance(mu, np.ndarray) and mu.shape == y_test.shape
        assert isinstance(std, np.ndarray) and std.shape == y_test.shape

        plots = CalibrationPlots(parameters)
        plots.plot_predictive(dataset, X_test, y_test, mu, std)

    def test_GaussianProcess_bo(self) -> None:
        seeds = range(10)
        for seed in seeds:
            kwargs.update(
                {
                    "surrogate": "GP",
                    "d": 2,
                    "seed": seed,
                    "problem": "Adjiman",
                    "n_evals": 10,
                    "bo": True,
                    "savepth": os.getcwd() + "/results/",
                }
            )
            parameters = Parameters(kwargs, mkdir=True)
            experiment = Experiment(parameters)
            experiment.run()

    def test_DummySurrogate(self) -> None:
        kwargs.update({"surrogate": "DS", "d": 2})
        parameters = Parameters(kwargs, mkdir=True)
        dataset = Dataset(parameters)
        model = DummySurrogate(parameters, dataset)
        X_test, y_test = dataset.sample_testset(
            n_samples=parameters.n_evals + parameters.n_evals
        )
        mu, std = model.predict(X_test)
        assert isinstance(mu, np.ndarray) and mu.shape == y_test.shape
        assert isinstance(std, np.ndarray) and std.shape == y_test.shape

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

    def test_DeepEnsemble(self) -> None:
        kwargs.update({"surrogate": "DE"})
        parameters = Parameters(kwargs, mkdir=True)
        dataset = Dataset(parameters)
        model = DeepEnsemble(parameters, dataset)
        X_test, y_test = dataset.sample_testset(
            n_samples=parameters.n_evals + parameters.n_evals
        )
        mu, std = model.predict(X_test)
        plots = CalibrationPlots(parameters)
        plots.plot_predictive(dataset, X_test, y_test, mu, std)

        assert isinstance(mu, np.ndarray) and mu.shape == y_test.shape
        assert isinstance(std, np.ndarray) and std.shape == y_test.shape

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


if __name__ == "__main__":
    unittest.main()
