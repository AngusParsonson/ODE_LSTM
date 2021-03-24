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

class GBPUSDDataModule(pl.LightningDataModule):
    def __init__(self, window=10, batch_size=1):
        super().__init__()
        self.window = window
        self.batch_size = batch_size

    def setup(self, stage=None):
        df = pd.read_csv(r"C:\Users\Angus Parsonson\Documents\University\Fourth Year\Dissertation\data\GBPUSD_Ticks_08.03.2021-08.03.2021.csv")
        train = self.prep_data(self.convert_to_seconds(df[:3000]))
        test = self.prep_data(self.convert_to_seconds(df[3000:4000]))
        
        self.train_data = self.sequence_data(train, self.window)
        self.test_data = self.sequence_data(test, self.window)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        return train_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        return test_dataloader

    def convert_to_seconds(self, df):
        df = self.convert_to_midprice(df)
        delta_t = []
        time_in_seconds = []
        prev_time = 0.0
        for index, row in df.iterrows():
            tokens = row['LocalTime'].split()
            time = tokens[1]
            h, m, s = time.split(':')
            time = float(int(h) * 3600 + int(m) * 60 + float(s))
            delta_t.append(time - prev_time)
            time_in_seconds.append([row['Midprice'], time])
            prev_time = time
        time_in_seconds = np.array(time_in_seconds)
        print(time_in_seconds.shape)
        plt.plot(time_in_seconds[:,1], time_in_seconds[:,0])
        plt.show()
        df['LocalTime'] = delta_t
        return df

    def convert_to_midprice(self, df):
        df['Midprice'] = (df['Ask'] + df['Bid']) / 2
        return df

    def sequence_data(self, data, window):
        sequence = []
        for i in range(0, len(data)-window):
            inputs_seq = data[i:i+window]
            label = data[i+window:i+window+1]
            sequence.append((inputs_seq, label))
        
        return sequence

    def prep_data(self, data):
        data.drop('Ask', axis=1, inplace=True)
        data.drop('Bid', axis=1, inplace=True)
        data.drop('AskVolume', axis=1, inplace=True)
        data.drop('BidVolume', axis=1, inplace=True)
        data = torch.FloatTensor(data.to_numpy())

        return data

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

        self.ode = NeuralDE(self.fc, solver='rk4', sensitivity='autograd')
    
    def forward(self, inputs, hx, ts):
        new_h, new_c = self.lstm(inputs, hx)
        indices = torch.argsort(ts)
        batch_size = ts.size(0)
        s_sort = ts[indices]
        s_sort = s_sort + torch.linspace(0, 1e-4, batch_size)
        # HACK: Make sure no two points are equal
        trajectory = self.ode.trajectory(new_h, s_sort)
        new_h = trajectory[indices, torch.arange(batch_size)]
        
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

        for t in range(seq_len):
            inputs = torch.unsqueeze(x[:, t], 1)
            ts = timespans[:, t].squeeze()
            # ts = torch.unsqueeze(timespans[:, t], 1)
            hidden_state = self.cell.forward(inputs, hidden_state, ts)
            current_output = self.fc(hidden_state[0])
            outputs.append(current_output)
            last_output = current_output 
        
        outputs = torch.stack(outputs, dim=1)

        return last_output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x[:,:,1], x[:,:,0])
        labels = y[:,:,1]
        criterion = nn.MSELoss()
        loss = criterion(logits, labels)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x[:,:,1], x[:,:,0])
        labels = y[:,:,1]
        criterion = nn.MSELoss()
        loss = criterion(logits, labels)

        items = [x.item() for x in logits]

        return {'predictions': items}

    def test_epoch_end(self, outputs):
        preds = [x['predictions'] for x in outputs]
        results = []
        for i in range (len(preds)):
            for pred in preds[i]:
                results.append(pred)
        # print(np.array(preds))
        plt.plot(results)
        plt.show()    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

if __name__ == '__main__':
    data_module = GBPUSDDataModule(50, 16)
    model = ODELSTM(1, 100)
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model, data_module)
    trainer.test()

    
    
    