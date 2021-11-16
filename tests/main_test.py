import unittest
from main import *


class MainTest(unittest.TestCase):
    def test_run(self) -> None:
        kwargs = {"savepth": os.getcwd() + "/results/"}
        parameters = Parameters(**kwargs)
        experiment = Experiment(parameters)
        experiment.run()


if __name__ == "__main__":
    unittest.main()
