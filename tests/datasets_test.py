from os import mkdir
import unittest
from datasets.benchmarks.benchmark import Benchmark
from datasets.verifications.verification import VerificationData
from main import *


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
        parameters = Parameters()
        data = Benchmark(parameters)
        check_dataset_attr(data)
        X = data.sample_X(n_samples=n_evals)
        y = data.get_y(X)
        assert X.shape == (n_evals, parameters.d)
        assert y.shape == (n_evals, 1)

    def test_verification(self) -> None:
        n_evals = 2
        parameters = Parameters()
        data = VerificationData(parameters)
        check_dataset_attr(data)
        X = data.sample_X(n_samples=n_evals)
        y = data.get_y(X)
        assert X.shape == (n_evals, parameters.d)
        assert y.shape == (n_evals, 1)


if __name__ == "__main__":
    unittest.main()
