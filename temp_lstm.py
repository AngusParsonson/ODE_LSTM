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
import matplotlib.pyplot as plt

class MinTempDataModule(pl.LightningDataModule):
    def __init__(self, window=10, batch_size=1):
        super().__init__()
        self.window = window
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        df = pd.read_csv(r"C:\Users\Angus Parsonson\Documents\University\Fourth Year\Dissertation\data\daily-min-temperatures.csv")
        train = self.normalise1D(torch.FloatTensor(df['Temp'])[:2920])
        test = self.normalise1D(torch.FloatTensor(df['Temp'])[2920:])
        self.train_data = self.sequence_data(train, self.window)
        self.test_data  = self.sequence_data(test, self.window)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        return train_dataloader
    
    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        return test_dataloader

    def normalise1D(self, tensor1D):
        tensor1D -= torch.min(tensor1D)
        tensor1D /= torch.max(tensor1D)

        return tensor1D

    def sequence_data(self, data, window):
        sequence = []
        for i in range(0, len(data)-window):
            inputs_seq = data[i:i+window]
            label = data[i+window:i+window+1]
            sequence.append((inputs_seq, label))
        
        return sequence

class LSTM(pl.LightningModule):
    def __init__(self, input_size=10, hidden_size=100, num_layers=2, output_size=1, batch_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        # self.conv = nn.Conv1d(1, 1, 3, padding=1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_cell = (torch.zeros(num_layers, 1, hidden_size),
                            torch.zeros(num_layers, 1, hidden_size))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tanh = nn.Tanh()

    def forward(self, x):
        print((x.view(len(x), 1, -1)).shape)
        print(self.hidden_cell[0].shape)
        lstm_out, self.hidden = self.lstm(x.view(len(x), 1, -1), self.hidden_cell)
        # lstm_out = self.tanh(self.conv(lstm_out))
        predictions = self.fc(lstm_out.view(len(x), -1))

        return predictions

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        self.hidden_cell = (torch.zeros(self.num_layers, 1, self.hidden_size),
                            torch.zeros(self.num_layers, 1, self.hidden_size))
        
        logits = self.forward(inputs)

        criterion = nn.MSELoss()
        loss = criterion(logits, labels)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        inputs, _ = test_batch
        self.hidden_cell = (torch.zeros(self.num_layers, 1, self.hidden_size),
                            torch.zeros(self.num_layers, 1, self.hidden_size))
        
        logits = self.forward(inputs)
        items = [x.item() for x in logits]

        return {'predictions': items}
    
    def test_epoch_end(self, outputs):
        temp_data_module = MinTempDataModule()
        df = pd.read_csv(r"C:\Users\Angus Parsonson\Documents\University\Fourth Year\Dissertation\data\daily-min-temperatures.csv")
        data = temp_data_module.normalise1D(torch.FloatTensor(df['Temp'][:2920]))
        old = temp_data_module.normalise1D(torch.FloatTensor(df['Temp']))
        preds = [x['predictions'] for x in outputs]
        old_items = [x.item() for x in old]
        results = []

        for i in range(0, 2920):
            results.append(data[i].item())
        for i in range(10):
            results.append(0.5)
        for i in range (len(preds)):
            for pred in preds[i]:
                results.append(pred)

        plt.title('Temperatures')
        plt.ylabel('Temp')
        plt.xlabel('Day')
        plt.autoscale(axis='x', tight=True)
        plt.plot(results[2930:], label='prediction')
        plt.plot(old_items[2930:], label='actual')
        plt.legend()
        plt.show()

    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr=0.0001)
        return optimiser

# Dataset describes minimum daily temperaturres over 10 years (1981-1990) 
# in Melbourne, Australia
if __name__ == '__main__':
    temp_data_module = MinTempDataModule(10, 16)
    # df = pd.read_csv(r"C:\Users\Angus Parsonson\Documents\University\Fourth Year\Dissertation\data\daily-min-temperatures.csv")
    # temp_data_module.plot_data()
    model = LSTM(num_layers=2)

    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(model, temp_data_module)
    trainer.test()

    
    
    