import sys
from torchdyn.models import *
from torchdyn import *
from torchdiffeq import odeint

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
    def __init__(self, window=10, batch_size=1, with_time=False):
        super().__init__()
        self.window = window
        self.batch_size = batch_size
        self.with_time = with_time
    
    def setup(self, stage=None):
        df = pd.read_csv(r"C:\Users\Angus Parsonson\Documents\University\Fourth Year\Dissertation\data\daily-min-temperatures.csv")
        
        train = self.normalise1D(torch.FloatTensor(df['Temp'])[:2920])
        test = self.normalise1D(torch.FloatTensor(df['Temp'])[2920:])

        if (self.with_time):
            train_with_time = []
            test_with_time = []
            for i in range(len(train)):
                train_with_time.append(torch.Tensor([torch.Tensor([i]), train[i]]))
            for i in range(len(test)):
                test_with_time.append(torch.Tensor([torch.Tensor([i]), test[i]]))
            train_with_time = torch.stack(train_with_time)
            test_with_time = torch.stack(test_with_time)

            self.train_data = self.sequence_data(train_with_time, self.window)
            self.test_data  = self.sequence_data(test_with_time, self.window)
        else:
            self.train_data = self.sequence_data(train, self.window)
            self.test_data  = self.sequence_data(test, self.window)

    def train_dataloader(self):
        if self.with_time:
            train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)
        else:
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

class ODELSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ODELSTMCell, self).__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.ode = NeuralDE(self.fc, solver='dopri5', sensitivity='adjoint')
    
    def forward(self, inputs, hx, ts):
        batch_size = ts.size(0)
        
        trajectory = []
        # print("last hidden: ")
        # print(hx[0])
        for i, t in enumerate(ts):
            # print(str(i) + "th batch")
            trajectory.append(self.ode.trajectory(hx[0], t))
            # print(len(trajectory))
            # print(trajectory)
        trajectory = torch.stack(trajectory)
        
        new_h = trajectory[torch.arange(batch_size),1,torch.arange(batch_size),:]
        # print(new_h)
        # print(trajectory.shape)
        # print(trajectory)
        # print(trajectory)
        # print(trajectory[:,1][torch.arange(batch_size)][torch.arange(batch_size)][torch.arange(batch_size)])
        # print(trajectory[torch.arange(batch_size), torch.arange(batch_size)])
        new_h, new_c = self.lstm(inputs, (new_h, hx[1]))
        new_h, new_c = self.lstm(inputs, (new_h, new_c))
        # new_h, new_c = self.lstm(inputs, hx)
        # indices = torch.argsort(ts)
        # batch_size = ts.size(0)
        # s_sort = ts[indices]
        # # s_sort = s_sort + torch.linspace(0, 1e-4, batch_size)
        # # # HACK: Make sure no two points are equal
        
        # trajectory = self.ode.trajectory(new_h, ts)
        # print(trajectory)
        # new_h = trajectory[:,]
        # # # new_h = trajectory[indices, torch.arange(batch_size)]
        # new_h = trajectory[torch.arange(batch_size), torch.arange(batch_size)]
        # return (
        #     torch.zeros((batch_size, self.hidden_size)),
        #     torch.zeros((batch_size, self.hidden_size)),
        # )
        return (new_h, new_c)

class ODELSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size):
        super(ODELSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.cell = ODELSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x, timespans):
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_state = (
            torch.zeros((batch_size, self.hidden_size)),
            torch.zeros((batch_size, self.hidden_size)),
        )

        outputs = []
        last_output = torch.zeros((batch_size, 1))
        # last_ts = timespans[:,0].squeeze()
        for t in range(1, seq_len):
            inputs = torch.unsqueeze(x[:, t], 1)
            # ts = torch.stack([timespans[:, t].squeeze(), last_ts])
            ts = timespans[:,t-1:t+1]
            # ts = timespans[:, t].squeeze()
            # ts = torch.unsqueeze(timespans[:, t], 1)
            hidden_state = self.cell.forward(inputs, hidden_state, ts)
            current_output = self.fc(hidden_state[0])
            outputs.append(current_output)
            last_output = current_output 
        
        outputs = torch.stack(outputs, dim=1)

        return last_output
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs[:,:,1], inputs[:,:,0])
        
        # batch = inputs.shape[0]
        # inputs = inputs.view(batch, -1)
        criterion = nn.MSELoss()
        loss = criterion(logits, labels[:,:,1])

        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs[:,:,1], inputs[:,:,0])
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
        return torch.optim.Adam(self.parameters(), lr=0.0001)

# Dataset describes minimum daily temperaturres over 10 years (1981-1990) 
# in Melbourne, Australia
if __name__ == '__main__':
    temp_data_module = MinTempDataModule(10, 16, with_time=True)
    # temp_data_module = MinTempDataModule(10, 16, with_time=False)

    # df = pd.read_csv(r"C:\Users\Angus Parsonson\Documents\University\Fourth Year\Dissertation\data\daily-min-temperatures.csv")
    # temp_data_module.plot_data()
    # model = LSTM(num_layers=2)
    model = ODELSTM(input_size=1, hidden_size=100)

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, temp_data_module)
    trainer.test()

    
    
    