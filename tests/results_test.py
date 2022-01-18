from os import mkdir
import unittest
from main import *
from visualizations.scripts.loader import Loader
from visualizations.scripts.ranking import Ranking
from visualizations.scripts.tables import Tables
from visualizations.scripts.figures import Figures


class ResultsTest(unittest.TestCase):
    def test_loader(self) -> None:
        loadpths = os.listdir(os.getcwd() + "/results/")
        loadpths = [os.getcwd() + "/results/" + f + "/" for f in loadpths]
        loadpths = [f for f in loadpths if os.path.isdir(f)]
        loader = Loader(loadpths)
        print(loader.data_summary)

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

    def test_ranking(self) -> None:
        loadpths = os.listdir(os.getcwd() + "/results/")
        loadpths = [
            os.getcwd() + "/results/" + f + "/" for f in loadpths if "tests" not in f
        ]
        loadpths = [f for f in loadpths if os.path.isdir(f)]
        Ranking(loadpths).run()

    def test_plots_bo_2d_contour(self) -> None:
        loadpths = os.listdir(os.getcwd() + "/results/")
        loadpths = [
            os.getcwd() + "/results/" + f + "/" for f in loadpths if "tests" not in f
        ]
        loadpths = [f for f in loadpths if os.path.isdir(f)]
        figures = Figures(loadpths)  # , {"surrogate": "GP", "problem": "Adjiman"})
        for seed in range(1, 11):
            figures.bo_2d_contour(n_epochs=50, seed=seed)

    def test_bo_regret_vs_no_bo_calibration(self) -> None:
        loadpths = os.listdir(os.getcwd() + "/results/")
        loadpths = [
            os.getcwd() + "/results/" + f + "/" for f in loadpths if "tests" not in f
        ]
        loadpths = [f for f in loadpths if os.path.isdir(f)]
        figures = Figures(loadpths)
        figures.bo_regret_vs_no_bo_calibration(epoch=50, avg=False)
        figures.bo_regret_vs_no_bo_calibration(epoch=50, avg=True)


if __name__ == "__main__":
    unittest.main()
