import unittest
from main import *


class MainTest(unittest.TestCase):
    def test_run(self) -> None:
        kwargs = {"savepth": os.getcwd() + "/results/", "n_evals": 2, "d": 2}
        parameters = Parameters(**kwargs)
        experiment = Experiment(parameters)
        experiment.run_bo()

    def test_demo1d(self) -> None:
        kwargs = {
            "savepth": os.getcwd() + "/results/",
            "d": 1,
            "plot_it": True,
        }
        parameters = Parameters(**kwargs)
        experiment = Experiment(parameters)
        experiment.run_calibraion_demo()
        assert experiment.dataset.data.X.shape == (parameters.n_train, parameters.d)
        assert experiment.dataset.data.y.shape == (parameters.n_train, 1)

    def test_demo2d(self) -> None:
        kwargs = {
            "savepth": os.getcwd() + "/results/",
            "d": 2,
            "plot_it": True,
        }
        parameters = Parameters(**kwargs)
        experiment = Experiment(parameters)
        experiment.run_calibraion_demo()
        assert experiment.dataset.data.X.shape == (parameters.n_train, parameters.d)
        assert experiment.dataset.data.y.shape == (parameters.n_train, 1)


if __name__ == "__main__":
    unittest.main()
