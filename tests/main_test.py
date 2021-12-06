import unittest
from main import *


class MainTest(unittest.TestCase):
    def test_run(self) -> None:
        kwargs = {
            "savepth": os.getcwd() + "/results/",
            "n_evals": 5,
            "d": 2,
            "plot_it": True,
            "vanilla": True,
        }
        parameters = Parameters(kwargs, mkdir=True)
        experiment = Experiment(parameters)
        experiment.run()

    def test_demo1d(self) -> None:
        kwargs = {
            "savepth": os.getcwd() + "/results/",
            "d": 1,
            "plot_it": True,
            "vanilla": True,
        }
        parameters = Parameters(kwargs, mkdir=True)
        experiment = Experiment(parameters)
        experiment.run_calibraion_demo()
        assert experiment.dataset.data.X.shape == (parameters.n_evals, parameters.d)
        assert experiment.dataset.data.y.shape == (parameters.n_evals, 1)

    def test_demo2d(self) -> None:
        kwargs = {
            "savepth": os.getcwd() + "/results/",
            "d": 2,
            "plot_it": True,
            "vanilla": True,
        }
        parameters = Parameters(kwargs, mkdir=True)
        experiment = Experiment(parameters)
        experiment.run_calibraion_demo()
        assert experiment.dataset.data.X.shape == (parameters.n_evals, parameters.d)
        assert experiment.dataset.data.y.shape == (parameters.n_evals, 1)

    def test_benchmark_2d(self) -> None:
        kwargs = {
            "savepth": os.getcwd() + "/results/",
            "d": 2,
            "plot_it": True,
            "vanilla": True,
            "data_location": "datasets.benchmarks.benchmark",
            "data_class": "Benchmark",
            "problem": "Alpine01",
        }
        parameters = Parameters(kwargs, mkdir=True)
        experiment = Experiment(parameters)
        experiment.run_calibraion_demo()


if __name__ == "__main__":
    unittest.main()
