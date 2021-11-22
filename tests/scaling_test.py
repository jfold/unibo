from os import mkdir
import unittest
from main import *
from visualizations.scripts.calibrationexperimentplots import Figures


class ScalingTest(unittest.TestCase):
    def test(self) -> None:
        savepths = []
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
            savepths.append(parameters.savepth)

        figures = Figures(savepths)
        figures.generate()


if __name__ == "__main__":
    unittest.main()
