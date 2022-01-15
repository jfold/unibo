from os import mkdir
import unittest
from datasets.benchmarks.benchmark import Benchmark
from datasets.verifications.verification import VerificationData
from main import *
from datasets.GP.gp import GPSampler
import matplotlib.pyplot as plt
import torch

kwargs = {
    "savepth": os.getcwd() + "/results/tests/",
    "data_class": "Benchmark",
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

    def test_benchmark_problem_dimensionality(self) -> None:
        n_functions = 5
        result = {}
        for d in range(1, 15):
            kwargs.update({"d": d})
            parameters = Parameters(kwargs, mkdir=False)
            benchmarks = Benchmark(parameters).benchmark_tags
            n_problems = 1
            problems = []
            for problem in sorted(benchmarks):
                if "unimodal" in benchmarks[problem]:
                    problems.append(problem)
                    n_problems += 1
                if n_problems > n_functions:
                    break
            result.update({d: problems})
        print(result)

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
