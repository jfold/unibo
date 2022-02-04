import sys
import random
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm, uniform, kstest, entropy, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors as KNNsklearn
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import energy_distance
import torch
import botorch
from botorch.models.model import Model
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import uncertainty_toolbox as uct  # https://github.com/uncertainty-toolbox/uncertainty-toolbox

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["figure.figsize"] = (12, 18)
matplotlib.rcParams["savefig.bbox"] = "tight"
# plot-settings:
ps = {
    "GP": {"c": "red", "m": "x"},
    "RF": {"c": "blue", "m": "4"},
    "BNN": {"c": "orange", "m": "v"},
    "DS": {"c": "black", "m": "*"},
    "RS": {"c": "palegreen", "m": "h"},
}

