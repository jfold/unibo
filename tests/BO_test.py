from os import mkdir
import unittest
from main import *
import numpy as np
import botorch
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
from src.optimizer import Optimizer
from src.dataset import Dataset
import matplotlib.pyplot as plt
kwargs = {
    "savepth": os.getcwd() + "/results/tests/",
}

class BoTest(unittest.TestCase):
    def test_BO_GP(self) -> None:
        train_x = np.array([0.1, 0.4, 0.6, 0.9])
        train_y = 3*train_x-3*train_x**2
        model = botorch.models.SingleTaskGP(torch.tensor(train_x).unsqueeze(-1), torch.tensor(train_y).unsqueeze(-1))
        state_dict = torch.load('objects/BO_test_GP')
        model.load_state_dict(state_dict)
        likelihood = ExactMarginalLogLikelihood(model.likelihood, model)
        botorch.fit.fit_gpytorch_model(likelihood)
        parameters = Parameters(kwargs, mkdir=True)
        parameters.maximization=True
        dataset = Dataset(parameters)
        dataset.y_opt = 0.48
        optim_test = Optimizer(parameters)
        optim_test.is_fitted = True
        optim_test.surrogate_model = model
        a_tuple = optim_test.bo_iter_test(dataset)
        assert optim_test.bo_iter_test(dataset)[0][0] > 0.49
        assert optim_test.bo_iter_test(dataset)[0][0] < 0.51

if __name__ == "__main__":
    unittest.main()