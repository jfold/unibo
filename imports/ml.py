import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, kstest, entropy
from sklearn.model_selection import train_test_split
import torch
import botorch
from botorch.models.model import Model
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["figure.figsize"] = (10, 6)
matplotlib.rcParams["savefig.bbox"] = "tight"
