from os import mkdir
import unittest
from main import *


class ScalingTest(unittest.TestCase):
    def test_calibration(self) -> None:
        savepths = []
        for seed in range(3):
            kwargs = {
                "savepth": os.getcwd() + "/results/",
                "d": 2,
                "seed": seed,
                "plot_it": True,
                "data_location": "datasets.verifications.verification",
                "data_class": "VerificationData",
                "vanilla": True,
            }
            parameters = Parameters(kwargs, mkdir=True)
            experiment = Experiment(parameters)
            experiment.run_calibraion_demo()
            savepths.append(parameters.savepth)

    def test_bayesian_opt(self) -> None:
        savepths = []
        for seed in range(3):
            kwargs = {
                "savepth": os.getcwd() + "/results/",
                "d": 2,
                "seed": seed,
                "data_location": "datasets.verifications.verification",
                "data_class": "VerificationData",
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
