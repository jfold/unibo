import unittest
from models.random_forest import RandomForest
from src.parameters import Defaults


class RandomForestTest(unittest.TestCase):
    def test_init(self) -> None:
        rf = RandomForest(Defaults())
        assert rf is not None


if __name__ == "__main__":
    unittest.main()
