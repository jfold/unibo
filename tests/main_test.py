import unittest
from main import *
import time

# Test domain:
surrogates = ["GP", "RF"]  # ,"BNN"]
dims = [1, 2, 10]
data = {
    "Benchmark": {
        "location": "datasets.benchmarks.benchmark",
        "problems": ["Alpine01"],
    },
    "VerificationData": {
        "location": "datasets.verifications.verification",
        "problems": ["Default"],
    },
}
kwargs = {
    "savepth": os.getcwd() + "/results/tests/",
    "n_evals": 3,
    "d": 2,
    "plot_it": True,
    "vanilla": True,
}

# t_0 = time.time() # consider timing
# test_result = []
# with open(
#     self.savepth + f"scores{parameters.save_settings}.json", "w"
# ) as f:
#     f.write(json_dump)


class MainTest(unittest.TestCase):
    def test_toy_bo_calibration(self) -> None:
        for surrogate in surrogates:
            kwargs_ = kwargs
            kwargs_.update({"surrogate": surrogate})
            parameters = Parameters(kwargs_, mkdir=True)
            experiment = Experiment(parameters)
            experiment.run()

    def test_toy_calibration(self) -> None:
        kwargs_ = kwargs
        for d in dims:
            for surrogate in surrogates:
                for data_name, info in data.items():
                    for problem in info["problems"]:
                        kwargs_ = kwargs
                        kwargs_.update(
                            {
                                "d": d,
                                "surrogate": surrogate,
                                "data_class": data_name,
                                "data_location": info["location"],
                                "problem": problem,
                            }
                        )
                        parameters = Parameters(kwargs_, mkdir=True)
                        experiment = Experiment(parameters)
                        experiment.run_calibration_demo()
                        assert experiment.dataset.data.X.shape == (
                            parameters.n_evals + parameters.n_initial,
                            parameters.d,
                        )
                        assert experiment.dataset.data.y.shape == (
                            parameters.n_evals + parameters.n_initial,
                            1,
                        )


if __name__ == "__main__":
    unittest.main()
