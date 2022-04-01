from os import mkdir
from imports.ml import *
import unittest
from datasets.benchmarks.benchmark import Benchmark
from datasets.verifications.verification import VerificationData
from main import *
from datasets.GP.gp import GPSampler
import matplotlib.pyplot as plt
import torch

kwargs = {
    "savepth": os.getcwd() + "/results/tests/",
    "data_object": "Benchmark",
    "data_location": "datasets.benchmarks.benchmark",
}


def check_dataset_attr(obj):
    assert hasattr(obj, "sample_X")
    assert hasattr(obj, "get_y")
    assert hasattr(obj, "lbs")
    assert hasattr(obj, "ubs")
    assert hasattr(obj, "d")
    assert hasattr(obj, "ne_true")


class DatasetsTest(unittest.TestCase):
    def test_benchmark(self) -> None:
        n_evals = 3000
        kwargs.update({"problem": "Adjiman", "d": 2})
        parameters = Parameters(kwargs)
        data = Benchmark(parameters)
        check_dataset_attr(data)
        X = data.sample_X(n_samples=n_evals)
        y = data.get_y(X)
        assert X.shape == (n_evals, parameters.d)
        assert y.shape == (n_evals, 1)

    def test_save_unibo_benchmarks_problems(self) -> None:
        result = {}
        random.seed(2022)
        for d in range(1, 15):
            kwargs.update({"d": d})
            parameters = Parameters(kwargs, mkdir=False)
            benchmarks = Benchmark(parameters).benchmark_tags
            keys = list(benchmarks.keys())
            random.shuffle(keys)
            benchmarks = {key: benchmarks[key] for key in keys}
            # benchmarks = sorted(benchmarks)
            problems = []
            for problem in benchmarks:
                if (
                    "unimodal" in benchmarks[problem]
                    and "boring" not in benchmarks[problem]
                ):
                    problems.append(problem)
            result.update({d: problems})
        print(result)
        json_dump = json.dumps(result)
        with open("datasets/benchmarks/unibo-problems.json", "w") as f:
            f.write(json_dump)

    def test_verification(self) -> None:
        n_evals = 2
        parameters = Parameters(kwargs)
        data = VerificationData(parameters)
        check_dataset_attr(data)
        X = data.sample_X(n_samples=n_evals)
        y = data.get_y(X)
        assert X.shape == (n_evals, parameters.d)
        assert y.shape == (n_evals, 1)

    def test_GP_sampler(self) -> None:
        n_evals = 2
        parameters = Parameters(kwargs)
        parameters.problem = "GP"
        data = GPSampler(parameters)
        check_dataset_attr(data)
        X = data.sample_X(n_samples=n_evals)
        y = data.get_y(X)
        print(y)
        assert X.shape == (n_evals, parameters.d)
        assert y.shape == (n_evals, 1)
        if parameters.d == 1:
            test_x = torch.linspace(-1, 1, 200).double()
            y = data.get_y(test_x)
            plt.plot(test_x, y)
            plt.show()


if __name__ == "__main__":
    unittest.main()
