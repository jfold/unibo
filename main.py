from src.parameters import Parameters
from src.dataset import Dataset
from src.experiment import Experiment
from surrogates.bayesian_neural_network import BayesianNeuralNetwork
from surrogates.gaussian_process import GaussianProcess
from surrogates.random_forest import RandomForest
from surrogates.dummy_surrogate import DummySurrogate
from surrogates.deep_ensemble import DeepEnsemble
import torch
import numpy as np
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
from theory_experiments.FashionMNIST import *
from theory_experiments.MNIST import *
from theory_experiments.FashionMNIST_CNN import *
from theory_experiments.MNIST_CNN import *
from theory_experiments.NewsClassification import *
from theory_experiments.SVM_wine import *
from datetime import datetime
import time
import sys
import warnings

#python3 -c "from main import *; run()" $args
#"seed=0|n_seeds_per_job=1|surrogate=GP|acquisition=EI|data_name=mnist|std_change=1.0|bo=True|experiment=experiment-GP--0|test=False|extensive_metrics=True|recalibrate=False"
#"seed=0|n_seeds_per_job=1|surrogate=GP|acquisition=EI|data_name=benchmark|problem_idx=11|snr=100|bo=True|d=3|experiment=experiment-TEST--seed-TEST"|test=False|recalibrate=False"
def run():
    start = time.time()
    try:
        args = sys.argv[-1].split("|")
    except:
        args = []
    print("------------------------------------")
    print("Arguments:", args)
    print("RUNNING EXPERIMENT...")
    warnings.filterwarnings("ignore", message="A not p.d.")
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

    for i in range(kwargs['n_seeds_per_job']):
        parameters = Parameters(kwargs, mkdir=True)
        print("Running with:", parameters)
        experiment = Experiment(parameters)
        experiment.run()
        print("FINISHED EXPERIMENT")
        print("------------------------------------")
        kwargs['seed'] += 1
if __name__ == "__main__":
    run()
