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
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from src.text_utility import *
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

#from https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

def NewsClassification(params):
    datetime_time = datetime.now().strftime("%d_%b_%Y (%H_%M_%S_%f)").replace(" ", "")
    params.d = 5
    file_name = "timestamp="+datetime_time+"_NewsClass_btrain="+str(params.b_train)+"_hiddensize="+str(params.hidden_size)
    random_seed = params.seed
    experiment_dict = {}
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    initial_setting_dict = {"random_seed":params.seed}

    batch_size_test = 1000
    output_size = 4
    get_new_init = True
    valid_size = 0.25
    batch_size_train = params.b_train
    hidden_size = params.hidden_size
    
    epochs = np.linspace(1, 10, num=10).astype(int)
    #epochs=[10]
    dropout_rates = np.linspace(0, 0.8, num=10).astype(float)
    #dropout_rates=[0.2]
    #Scale is different on learning rate due to using SGD as gradients are sparse at times with this model.
    #Please note that a scheduler is also used to step learning rate!
    learning_rates = np.linspace(0.1, 10, num=10)
    #learning_rates = [1]
    initial_setting_dict['batch_size_test']=batch_size_test
    initial_setting_dict['output_size']=output_size
    experiment_dict['init_settings'] = initial_setting_dict

    tokenizer = get_tokenizer('basic_english')
    train_iter = AG_NEWS(root='./datasets/', split='train')

    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1

    def collate_batch(batch):
        device = torch.device("cpu")
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    train_dataset = to_map_style_dataset(train_iter)
    num_train = int(len(train_dataset)*valid_size)
    split_train, split_valid = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    valid_loader = torch.utils.data.DataLoader(split_valid, 
                    batch_size=batch_size_test, collate_fn=collate_batch)
    valid_losses = []
    accuracies = []
    hyperparam_list = []
    for epoch in epochs:
        for dropout in dropout_rates:
            for learning_rate in learning_rates:
                MNIST_experiment = {}
                hyperparams = np.array([hidden_size, learning_rate, batch_size_train, epoch, dropout])
                hyperparams = hyperparams.T
                dataset = Dataset(params)
                dataset.data.problem = "text_NN"
                valid_loss, accuracy = text_BO_iter(len(vocab), output_size, split_train, valid_loader, hyperparams, collate_batch)
                valid_losses.append(valid_loss.detach().item())
                accuracies.append(accuracy)
                hyperparam_list.append(hyperparams.T.tolist())
    MNIST_results = {}
    MNIST_results["hyperparams"] = hyperparam_list
    MNIST_results["valid_losses"] = valid_losses
    MNIST_results["accuracies"] = accuracies
    experiment_dict['MNIST_results'] = MNIST_results
    with open("/zhome/49/2/135395/PhD/unibo/results/NewsClass/" + file_name+ ".json", 'w') as fp:
        json.dump(experiment_dict, fp, indent=4)
#    with open("./results/NewsClass/" + file_name+ ".json", 'w') as fp:
#        json.dump(experiment_dict, fp, indent=4)
if __name__ == "__main__":
    NewsClassification()
