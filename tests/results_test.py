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
        loader = Loader(loadpths, update=True)  # change update
        for k in loader.loader_summary.keys():
            print(k, loader.loader_summary[k])
        print("data.shape:", loader.data.shape, "data.nbytes:", loader.data.nbytes)
        print(
            f"LOADING ALL {len(loadpths)} FILES TOOK: --- %s seconds ---"
            % (time.time() - start_time)
        )

    def test_plot_metrics_vs_epochs(self) -> None:
        loadpths = get_loadpths(os.getcwd() + "/results/")
        figures = Figures(loadpths)

        figures.metrics_vs_epochs(
            metrics=["y_calibration_mse", "true_regret"], save_str="C-R",
        )

        figures.metrics_vs_epochs(
            metrics=["y_calibration_mse", "mahalanobis_dist"], save_str="C-M",
        )

    def test_plot_bo_2d_contour(self) -> None:
        loadpths = get_loadpths(os.getcwd() + "/results/")
        figures = Figures(loadpths)
        for seed in range(1, 11):
            figures.bo_2d_contour(n_epochs=50, seed=seed)

    def test_table_correlation(self) -> None:
        loadpths = get_loadpths(os.getcwd() + "/results/")
        ranking = Tables(loadpths, update_data=False)
        ranking.correlation_table()

    def test_table_ranking_no_bo(self) -> None:
        loadpths = get_loadpths(os.getcwd() + "/results/")
        ranking = Ranking(loadpths, update_data=False)
        ranking.table_ranking_no_bo()

    def test_table_ranking_with_bo(self) -> None:
        loadpths = get_loadpths(os.getcwd() + "/results/")
        ranking = Ranking(loadpths, update_data=False)
        ranking.table_ranking_with_bo()

    def test_calculate_rank(self) -> None:
        # start_time = time.time()
        loadpths = get_loadpths(os.getcwd() + "/results/")
        ranking = Ranking(loadpths, update_data=False)
        ranking.calc_surrogate_ranks(with_bo=True, save=True)
        ranking.calc_surrogate_ranks(with_bo=False, save=True)

    def test_plot_rank_vs_epochs(self) -> None:
        loadpths = get_loadpths(os.getcwd() + "/results/")
        ranking = Ranking(loadpths, update_data=False)

        ranking.rank_vs_epochs(
            metrics=["y_calibration_mse", "true_regret"], save_str="C-R",
        )

        ranking.rank_vs_epochs(
            metrics=["y_calibration_mse", "mahalanobis_dist"], save_str="C-M",
        )

    def test_plot_exp_improv_vs_act_improv(self) -> None:
        loadpths = get_loadpths(os.getcwd() + "/results/")
        figures = Figures(loadpths, update_data=False)
        figures.exp_improv_vs_act_improv()


if __name__ == "__main__":
    unittest.main()
