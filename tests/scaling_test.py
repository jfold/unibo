from os import mkdir
import unittest
from main import *
from visualizations.scripts.calibrationexperimentplots import Figures


class ScalingTest(unittest.TestCase):
    def test(self) -> None:
        for seed in range(3):
            kwargs = {
                "savepth": os.getcwd() + "/results/",
                "d": 2,
                "seed": seed,
                "plot_it": True,
            }
            parameters = Parameters(kwargs, mkdir=True)
            experiment = Experiment(parameters)
            experiment.run_calibraion_demo()

        figures = Figures()
        figures.generate()


if __name__ == "__main__":
    unittest.main()
