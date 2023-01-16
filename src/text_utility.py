from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
import torch.optim as optim
import numpy as np
#from https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
#Must install torchtext on cluster

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class, dropout_rate):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(self.dropout(embedded))

    def train(self, text, offsets, label, optimizer, loss_func):
        optimizer.zero_grad()
        outputs = self(text, offsets)
        loss = loss_func(outputs, label)
        loss.backward()
        optimizer.step()
        return loss

def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)

def text_BO_iter(vocab_size, output_size, split_train, valid_loader, hyperparam, collate_batch):
    NeuralNet = TextClassificationModel(vocab_size, int(hyperparam[0]), output_size, hyperparam[4])
    optimizer = optim.SGD(NeuralNet.parameters(), lr=hyperparam[1])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    loss_func = nn.CrossEntropyLoss()
    train_loss = []
    train_loader = torch.utils.data.DataLoader(split_train, 
                    batch_size=int(hyperparam[2]), collate_fn=collate_batch)
    for i in range(int(hyperparam[3])):
        for idx, (label, text, offsets) in enumerate(train_loader):
            loss = NeuralNet.train(text, offsets, label, optimizer, loss_func)
        scheduler.step()
        train_loss.append(loss.item())
    with torch.no_grad():
        correct = 0
        count = 0
        for i, (label, text, offsets) in enumerate(valid_loader):
            outputs = NeuralNet(text, offsets)
            _, predictions = torch.max(outputs.data, 1)
            count += label.size(0)
            correct += (predictions == label).sum().item()
            loss = loss_func(outputs, label)
    return loss, correct/(count)*100