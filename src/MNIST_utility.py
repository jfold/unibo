import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import json
from scipy import linalg

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        return x
    
    def train(self, x, labels, optimizer, loss_func):
        optimizer.zero_grad()
        outputs = self(x)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss


def NN_BO_iter(input_size, output_size, train_dataset, train_sampler, valid_loader, hyperparam):
    NeuralNet = Net(input_size, int(hyperparam[0]), output_size, hyperparam[4])
    optimizer = optim.Adam(NeuralNet.parameters(), lr=np.exp(hyperparam[1]))
    loss_func = nn.CrossEntropyLoss()
    train_loss = []
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=int(hyperparam[2]), sampler=train_sampler)
    for i in range(int(hyperparam[3])):
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            images = images.reshape(-1, 28*28)
            loss = NeuralNet.train(images, labels, optimizer, loss_func)
        train_loss.append(loss.item())
    with torch.no_grad():
        correct = 0
        count = 0
        for i, (images, labels) in enumerate(valid_loader):
            images = images.reshape(-1, 28*28)
            outputs = NeuralNet(images)
            _, predictions = torch.max(outputs.data, 1)
            count += labels.size(0)
            correct += (predictions == labels).sum().item()
            loss = loss_func(outputs, labels)
    return loss, correct/(count)*100