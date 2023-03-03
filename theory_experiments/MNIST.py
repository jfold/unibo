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
from datetime import datetime


def MNIST(params):
    datetime_time = datetime.now().strftime("%d_%b_%Y (%H_%M_%S_%f)").replace(" ", "")
    params.d = 5
    file_name = "timestamp="+datetime_time+"_MNIST_btrain="+str(params.b_train)+"_hiddensize="+str(params.hidden_size)
    random_seed = params.seed
    experiment_dict = {}
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    initial_setting_dict = {"random_seed":params.seed}

    batch_size_test = 1000
    output_size = 10
    get_new_init = True
    input_size = 28*28
    valid_size = 0.25
    batch_size_train = params.b_train
    hidden_size = params.hidden_size
    
    epochs = np.linspace(1, 10, num=10).astype(int)
    dropout_rates = np.linspace(0, 0.8, num=10).astype(float)
    #ln(0.00001)= -11.51, ln(0.1) = - 2.23, 
    learning_rates = np.linspace(-11.51, -2.23, num=10)
    initial_setting_dict['batch_size_test']=batch_size_test
    initial_setting_dict['output_size']=output_size
    initial_setting_dict['input_size']=input_size
    experiment_dict['init_settings'] = initial_setting_dict
    train_dataset = datasets.MNIST(root='./datasets/', train=True, 
                download=True, transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))
    valid_dataset = datasets.MNIST(root='./datasets/', train=True, 
                download=True, transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                    batch_size=batch_size_test, sampler=valid_sampler)
    valid_losses = []
    accuracies = []
    hyperparam_list = []
    for epoch in epochs:
        for dropout in dropout_rates:
            for learning_rate in learning_rates:
                MNIST_experiment = {}
                hyperparams = np.array([hidden_size, learning_rate, batch_size_train, epoch, dropout])
                hyperparams = hyperparams.T
                scaler_x = StandardScaler()
                scaler_y = StandardScaler()
                dataset = Dataset(params)
                dataset.data.problem = "NN"
                valid_loss, accuracy = NN_BO_iter(input_size, output_size, train_dataset, train_sampler, valid_loader, hyperparams)
                valid_losses.append(valid_loss.detach().item())
                accuracies.append(accuracy)
                hyperparam_list.append(hyperparams.T.tolist())
    MNIST_results = {}
    MNIST_results["hyperparams"] = hyperparam_list
    MNIST_results["valid_losses"] = valid_losses
    MNIST_results["accuracies"] = accuracies
    experiment_dict['MNIST_results'] = MNIST_results
    with open("/zhome/49/2/135395/PhD/unibo/results/MNIST/" + file_name+ ".json", 'w') as fp:
        json.dump(experiment_dict, fp, indent=4)
if __name__ == "__main__":
    MNIST()
