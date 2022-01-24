from os import mkdir
import unittest
from main import *
from visualizations.scripts.loader import Loader
from visualizations.scripts.ranking import Ranking
from visualizations.scripts.tables import Tables
from visualizations.scripts.figures import Figures
import time


def get_loadpths(pth: str) -> list[str]:
    loadpths = os.listdir(pth)
    loadpths = [f"{pth}{f}/" for f in loadpths if "tests" not in f]
    return loadpths


class ResultsTest(unittest.TestCase):
    def test_loader(self) -> None:
        start_time = time.time()
        loadpths = get_loadpths(os.getcwd() + "/results/")
        loader = Loader(loadpths)
        for k in loader.loader_summary.keys():
            print(k, loader.loader_summary[k])
        print("data.shape:", loader.data.shape)
        print(
            f"LOADING ALL {len(loadpths)} FILES TOOK: --- %s seconds ---"
            % (time.time() - start_time)
        )

    def test_metrics_vs_epochs(self) -> None:
        loadpths = get_loadpths(os.getcwd() + "/results/")
        Figures(loadpths).metrics_vs_epochs()

    def test_plots_bo_2d_contour(self) -> None:
        loadpths = get_loadpths(os.getcwd() + "/results/")
        figures = Figures(loadpths)
        for seed in range(1, 11):
            figures.bo_2d_contour(n_epochs=50, seed=seed)

    def test_bo_regret_vs_no_bo_calibration(self) -> None:
        loadpths = get_loadpths(os.getcwd() + "/results/")
        figures = Figures(loadpths)
        figures.bo_regret_vs_no_bo_calibration(epoch=50, avg_names=[])
        figures.bo_regret_vs_no_bo_calibration(epoch=50, avg_names=["seed"])

    def test_ranking(self) -> None:
        loadpths = get_loadpths(os.getcwd() + "/results/")
        Ranking(loadpths).run()


if __name__ == "__main__":
    unittest.main()
