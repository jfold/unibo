from src.parameters import Parameters
from src.dataset import Dataset
from surrogates.bayesian_neural_network import BayesianNeuralNetwork
from surrogates.gaussian_process import GaussianProcess
from surrogates.random_forest import RandomForest
from surrogates.dummy_surrogate import DummySurrogate
from surrogates.deep_ensemble import DeepEnsemble
import torch
import numpy as np
import torchvision
from torchvision import datasets
from src.MNIST_utility import *
import torch.optim as optim
import matplotlib.pyplot as plt
from botorch.acquisition.analytic import ExpectedImprovement
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import random
from torch.utils.data.sampler import SubsetRandomSampler
import json
from sklearn.preprocessing import StandardScaler
import itertools
from experiments.FashionMNIST import *
from datetime import datetime
import time
import sys


def run():
    start = time.time()
    try:
        args = sys.argv[-1].split("|")
    except:
        args = []
    print("------------------------------------")
    print("Arguments:", args)
    print("RUNNING EXPERIMENT...")
    kwargs = {}
    parameters_temp = Parameters(mkdir=False)
    if args[0] != "main.py":
        for arg in args:
            var = arg.split("=")[0]
            val = arg.split("=")[1]
            par_val = getattr(parameters_temp, var)

            if isinstance(par_val, bool):
                val = val.lower() == "true"
            elif isinstance(par_val, int):
                val = int(val)
            elif isinstance(par_val, float):
                val = float(val)
            elif isinstance(par_val, str):
                pass
            else:
                var = None
                print("COULD NOT FIND VARIABLE:", var)
            kwargs.update({var: val})

    parameters = Parameters(kwargs, mkdir=True)
    print("Running with:", parameters)
    FashionMNIST(parameters)
    print("FINISHED EXPERIMENT")
    print("------------------------------------")


if __name__ == "__main__":
    run()
