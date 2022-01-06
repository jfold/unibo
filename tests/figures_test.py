from os import mkdir
import unittest
from main import *
from visualizations.scripts.tables import Tables
from visualizations.scripts.figures import Figures


class ResultsTest(unittest.TestCase):
    def test_plots_default(self) -> None:
        loadpths = os.listdir(os.getcwd() + "/results/tests/")
        loadpths = [os.getcwd() + "/results/tests/" + f + "/" for f in loadpths]
        loadpths = [f for f in loadpths if os.path.isdir(f)]
        Figures(loadpths).generate()

    def test_tables_default(self) -> None:
        loadpths = os.listdir(os.getcwd() + "/results/tests/")
        loadpths = [os.getcwd() + "/results/tests/" + f + "/" for f in loadpths]
        loadpths = [f for f in loadpths if os.path.isdir(f)]
        Tables(loadpths, settings={"bo": True}).generate()
        Tables(loadpths, settings={"bo": False}).generate()

    def test_plots_epochs(self) -> None:
        loadpths = os.listdir(os.getcwd() + "/results/")
        loadpths = [
            os.getcwd() + "/results/" + f + "/" for f in loadpths if "tests" not in f
        ]
        loadpths = [f for f in loadpths if os.path.isdir(f)]
        Figures(loadpths).calibration_vs_epochs()


if __name__ == "__main__":
    unittest.main()
