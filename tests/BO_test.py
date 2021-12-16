import numpy as np
import botorch
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from optimizer import Optimizer
from dataset import Dataset
kwargs = {
    "savepth": os.getcwd() + "/results/tests/",
}

#class BoTest(unittest.TestCase):
    #def test_BO_GP(self) -> None:
train_x = np.array([0.1, 0.4, 0.6, 0.9])
train_y = train_x-train_x**2
n=4
model = botorch.models.SingleTaskGP(torch.tensor(train_x).unsqueeze(-1), torch.tensor(train_y).unsqueeze(-1))
state_dict = torch.load('../objects/BO_test_GP')
model.load_state_dict(state_dict)
likelihood = ExactMarginalLogLikelihood(model.likelihood, model)
parameters = Parameters(kwargs, mkdir=True)
dataset = Dataset(parameters)
dataset.y_opt = 0.24
optim_test = Optimizer(parameters)
optim_test.is_fitted = True
optim_test.surrogate_model = model
