from os import mkdir
import unittest
from main import *
from visualizations.scripts.calibrationexperimentplots import Figures


class ScalingTest(unittest.TestCase):
    def test_calibration(self) -> None:
        savepths = []
        for seed in range(3):
            kwargs = {
                "savepth": os.getcwd() + "/results/",
                "d": 2,
                "seed": seed,
                "plot_it": True,
                "vanilla": True,
            }
            parameters = Parameters(kwargs, mkdir=True)
            experiment = Experiment(parameters)
            experiment.run_calibraion_demo()
            savepths.append(parameters.savepth)

    def test_bayesian_opt(self) -> None:
        savepths = []
        print("HELLO")
        for seed in range(3):
            kwargs = {
                "savepth": os.getcwd() + "/results/",
                "d": 2,
                "seed": seed,
                "plot_it": True,
                "vanilla": True,
                "n_evals": 10,
            }
            parameters = Parameters(kwargs, mkdir=True)
            experiment = Experiment(parameters)
            experiment.run_bo()
            savepths.append(parameters.savepth)


if __name__ == "__main__":
    unittest.main()
