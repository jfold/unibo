from os import mkdir
import unittest
from datasets.benchmarks.benchmark import Benchmark
from datasets.verifications.verification import VerificationData
from main import *

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
        n_evals = 2
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


if __name__ == "__main__":
    unittest.main()
