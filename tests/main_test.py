import unittest
from main import *


class MainTest(unittest.TestCase):
    def test_run(self) -> None:
        kwargs = {"savepth": os.getcwd() + "/results/", "n_evals": 2, "d": 2}
        parameters = Parameters(**kwargs)
        experiment = Experiment(parameters)
        experiment.run()

    def test_demo(self) -> None:
        kwargs = {"savepth": os.getcwd() + "/results/", "d": 2}
        parameters = Parameters(**kwargs)
        experiment = Experiment(parameters)
        experiment.demo()


if __name__ == "__main__":
    unittest.main()
