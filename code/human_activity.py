import sys
from torchdyn.models import *
from torchdyn import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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

class timeseries(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.long)
        print(self.x[0].shape, self.y[0].shape)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.len

class HumanActivityDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        trainX, trainy, testX, testy = self.load_dataset(r'C:\Users\Angus Parsonson\Documents\University\Fourth Year\Dissertation\data\\')
        print(min(trainy), max(trainy))
        self.train_data = timeseries(trainX, trainy)
        self.test_data = timeseries(testX, testy)

    def train_dataloader(self):
        # train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        train_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        return train_dataloader
    
    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        return test_dataloader

    # load a single file as a numpy array
    def load_file(self, filepath):
        dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
        return dataframe.values

    def load_group(self, filenames, prefix=''):
        loaded = list()
        for name in filenames:
            data = self.load_file(prefix + name)
            loaded.append(data)
        # stack group so that features are the 3rd dimension
        loaded = np.dstack(loaded)
        return loaded

    # load a dataset group, such as train or test
    def load_dataset_group(self, group, prefix=''):
        filepath = prefix + group + '/Inertial Signals/'
        # load all 9 files as a single array
        filenames = list()
        # total acceleration
        filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
        # body acceleration
        filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
        # body gyroscope
        filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
        # load input data
        X = self.load_group(filenames, filepath)
        # load class output
        y = self.load_file(prefix + group + '/y_'+group+'.txt')
        return X, y

    # load the dataset, returns train and test X and y elements
    def load_dataset(self, prefix=''):
        # load all train
        trainX, trainy = self.load_dataset_group('train', prefix + 'HARDataset/')
        print(trainX.shape, trainy.shape)
        # load all test
        testX, testy = self.load_dataset_group('test', prefix + 'HARDataset/')
        print(testX.shape, testy.shape)
        # zero-offset class values
        trainy = trainy - 1
        testy = testy - 1
        # one hot encode y
        # trainy = pd.to_categorical(trainy)
        # testy = pd.to_categorical(testy)
        # print(trainX.shape, trainy.shape, testX.shape, testy.shape)
        return trainX, trainy, testX, testy

class LSTM(pl.LightningModule):
    def __init__(self, input_size=9, hidden_size=100, seq_len=128, num_layers=1, output_size=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

        self.fc = nn.Linear(hidden_size, output_size)
        # self.hidden_cell = (torch.zeros(num_layers, batch_size, hidden_size),
        #                     torch.zeros(num_layers, batch_size, hidden_size))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:,-1,:])
        predictions = self.soft(self.fc(self.relu(lstm_out)))

        return predictions

    def training_step(self, batch, batch_idx):
        X, y = batch
        print(X.shape)
        print(X)
        logits = self.forward(X)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y.view(-1))

        return loss
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)

        total = 0
        correct = 0
        for i in range(len(logits)):
            pred = logits[i].argmax(dim=0, keepdim=True)
            # print(pred)
            if (pred[0] == y[i]):
                correct += 1
            total += 1

        metrics = {'correct': correct, 'total': total}
        return metrics

    def test_epoch_end(self, outputs):
        correct = sum([x['correct'] for x in outputs])
        total = sum([x['total'] for x in outputs])
        print(100*correct/total)
        return {'overall_accuracy': 100*correct/total}   

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

if __name__ == '__main__':
    data_module = HumanActivityDataModule(batch_size=4)
    model = LSTM(input_size=9, hidden_size=100, seq_len=128, num_layers=2)
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, data_module)
    trainer.test()