import sys
from torchdyn.models import *
from torchdyn import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.functional import accuracy

import seaborn as sns 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import statistics

class FlightsDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        flights = sns.load_dataset('flights').passengers.astype(float)
        train = np.array(flights[:132])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_normalized = torch.FloatTensor(scaler.fit_transform(train.reshape(-1, 1))).view(-1)

        train_data = sequence_data(train_data_normalized, 6)
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

        return train_loader

    def test_dataloader(self):
        flights = sns.load_dataset('flights').passengers.astype(float)
        test = np.array(flights[132:])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        test_data_normalized = torch.FloatTensor(scaler.fit_transform(test.reshape(-1, 1))).view(-1)

        test_data = sequence_data(test_data_normalized, 6)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

        return test_loader

class LSTM(pl.LightningModule):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        print(input_seq.view(len(input_seq), 1, -1).shape)
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
    
    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        inputs = torch.transpose(inputs, 0, 1)
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
        
        logits = self.forward(inputs)

        criterion = nn.MSELoss()
        loss = criterion(logits, labels[0])
        return loss
    
    def test_step(self, test_batch, batch_idx):
        inputs, labels = test_batch
        inputs = torch.transpose(inputs, 0, 1)
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
        
        logits = self(inputs)
        criterion = nn.MSELoss()
        loss = criterion(logits, labels[0])
        metrics = {'loss': loss.item()}
        return metrics

    def test_epoch_end(self, outputs):
        return {'average_loss': statistics.mean([x['loss'] for x in outputs])}

    def configure_optimizers(self):
        optimiser = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        return optimiser


def sequence_data(data, window):
    sequences = []
    for i in range(len(data)-window):
        seq = data[i:i+window]
        label = data[i+window:i+window+1]
        sequences.append((seq, label))

    return sequences

if __name__ == '__main__':
    print("Libraries work")
    flights_data_module = FlightsDataModule()
    model = LSTM()

    trainer = pl.Trainer(max_epochs=50) # fast_dev_run boolean is useful

    trainer.fit(model, flights_data_module)
    trainer.test()
    # train_tensor = torch.Tensor(sequence_data(train_data_normalized, 6))
        
        