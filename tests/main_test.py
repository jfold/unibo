import unittest
from main import *
import time

# Test domain:
surrogates = ["GP", "RF"]  # ,"BNN"]
dims = [1, 10]
kwargs = {
    "savepth": os.getcwd() + "/results/",
    "n_evals": 3,
    "plot_it": True,
    "vanilla": True,
}


class MainTest(unittest.TestCase):
    def test_run(self) -> None:
        # t_0 = time.time() # consider timing
        # test_result = []
        for surrogate in surrogates:
            kwargs_ = kwargs
            kwargs_.update(
                {"d": 2, "surrogate": surrogate,}
            )
            parameters = Parameters(kwargs, mkdir=True)
            experiment = Experiment(parameters)
            experiment.run()
            # with open(
            #     self.savepth + f"scores{parameters.save_settings}.json", "w"
            # ) as f:
            #     f.write(json_dump)

    def test_toy_calibration_demo(self) -> None:
        for d in dims:
            for surrogate in surrogates:
                kwargs_ = kwargs
                kwargs_.update({"d": d, "surrogate": surrogate})
            parameters = Parameters(kwargs, mkdir=True)
            experiment = Experiment(parameters)
            experiment.run_calibration_demo()
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
        experiment.run_calibration_demo()


if __name__ == "__main__":
    unittest.main()
