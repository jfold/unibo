import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, kstest, entropy, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors as KNNsklearn
from sklearn.linear_model import LinearRegression
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
# matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
#     color=["r", "k", "c"], marker=["s", "o", "v"]
# )
plot_settings = {
    "GP": {"c": "r", "m": "s"},
    "RF": {"c": "k", "m": "o"},
    "BNN": {"c": "c", "m": "v"},
}

