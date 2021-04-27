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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class timeseries(Dataset):
    def __init__(self,x,y,with_time=True):
        if with_time:
            self.x = torch.tensor(x,dtype=torch.float32)
        else:
            self.x = torch.tensor(x,dtype=torch.float32)[:,:,1]
        
        self.y = torch.tensor(y,dtype=torch.float32)[:,:,1]
        # print(self.x[0])
        # print(self.y[0])
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.len

class GBPUSDDataModule(pl.LightningDataModule):
    def __init__(self, window=10, batch_size=1, time_projection=10, with_time=True):
        super().__init__()
        self.window = window
        self.batch_size = batch_size
        self.time_projection = time_projection
        self.with_time = with_time

    def setup(self, stage=None):
        df = pd.read_csv(r"C:\Users\Angus Parsonson\Documents\University\Fourth Year\Dissertation\data\GBPUSD_Ticks_08.03.2021-08.03.2021.csv")
        train = self.prep_data(self.convert_to_seconds(df[:3000]))
        test = self.prep_data(self.convert_to_seconds(df[3000:4000]))
        # plt.plot(train[:,0], train[:,1])
        # plt.show()
        plt.plot(test[:,0], test[:,1])
        plt.show()
        X_train, Y_train = self.sequence_data(train, self.window)
        X_test, Y_test = self.sequence_data(test, self.window)
    
        self.train_data = timeseries(X_train, Y_train, with_time=self.with_time)
        self.test_data = timeseries(X_test, Y_test, with_time=self.with_time)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        # train_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        return train_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        return test_dataloader

    def normalise(self, data):
        mp = data[:,1]
        data[:,1] = (mp - mp.min()) / (mp.max() - mp.min())
        return data

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
        # plt.plot(time_in_seconds[:,1], time_in_seconds[:,0])
        # plt.show()
        df['LocalTime'] = time_in_seconds[:,1]
        return df

    def convert_to_midprice(self, df):
        df['Midprice'] = (df['Ask'] + df['Bid']) / float(2.0)
        return df

    def sequence_data(self, data, window):
        X = []
        Y = []
        stag = 0
        up = 0
        dwn = 0
        size = len(data)
        for i in range(0, len(data)-window):
            inputs_seq = data[i:i+window]
            label = data[i+window:i+window+1]
            j = i+window-1
            curr_midprice = data[j][1]
            curr_time = data[j][0]
            # while (j < len(data) and data[j][0] < curr_time+self.time_projection):
            #     j += 1
            # if (j >= len(data)):
            #     size = i+window-1
            #     break

            # if (data[j][1] > curr_midprice + 0.005):
            #     label = 0
            #     up += 1
            # elif (data[j][1] < curr_midprice - 0.005):
            #     label = 2
            #     dwn += 1
            # else:
            #     label = 1
            #     stag += 1
            # print(label)
            X.append(inputs_seq)
            Y.append(label)
        print("up: " + str(up) + " , down: " + str(dwn) + ", stagnant: " + str(stag))

        return np.array(X), np.array(Y)


    def prep_data(self, data):
        data.drop('Ask', axis=1, inplace=True)
        data.drop('Bid', axis=1, inplace=True)
        data.drop('AskVolume', axis=1, inplace=True)
        data.drop('BidVolume', axis=1, inplace=True)
        data = self.normalise(np.array(data))
        
        return data

class LSTM(pl.LightningModule):
    def __init__(self, input_size=1, hidden_size=100, seq_len=10, num_layers=1, batch_size=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

        self.fc = nn.Linear(hidden_size, output_size)
        # self.hidden_cell = (torch.zeros(num_layers, batch_size, hidden_size),
        #                     torch.zeros(num_layers, batch_size, hidden_size))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.tanh = nn.Tanh()

    def forward(self, x):
        # self.hidden_cell = (torch.zeros(self.num_layers, len(x), self.hidden_size),
        #                     torch.zeros(self.num_layers, len(x), self.hidden_size))
        
        lstm_out, self.hidden = self.lstm(x)
        lstm_out = lstm_out[:,-1,:]
        predictions = self.fc(self.tanh(lstm_out))

        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x.view(-1, self.seq_len, 1))
        # labels = torch.Tensor(y).type(torch.LongTensor)
        # print(logits)
        # print(y)
        criterion = nn.MSELoss()
        loss = criterion(logits, y)
        # print(logits)
        # print(labels)
        # print(loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, labels = batch
        logits = self.forward(x.view(-1, self.seq_len, 1))
        # for log in logits:
        #     print(log)
        items = [x.item() for x in logits]

        return {'predictions': items}

    def test_epoch_end(self, outputs):
        preds = [x['predictions'] for x in outputs] 
        results = []
        for i in range(len(preds)):
            for p in preds[i]:
                results.append(p)

        print(results)
        plt.plot(results)
        plt.show()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

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
        batch_size = ts.size(0)

        new_h = torch.zeros(batch_size, hx[0].size(1))
        for batch_idx, batch in enumerate(ts):
            new_h[batch_idx] = self.ode.trajectory(hx[0][batch_idx], batch)[1]

        # trajectory = []
        # for i, t in enumerate(ts):
        #     trajectory.append(self.ode.trajectory(hx[0], t)[1])
        # trajectory = torch.stack(trajectory)
        
        # new_h = trajectory[torch.arange(batch_size),torch.arange(batch_size),:]
        new_h, new_c = self.lstm(inputs, (new_h, hx[1]))
        new_h, new_c = self.lstm(inputs, (new_h, new_c))

        # new_h, new_c = self.lstm(inputs, hx)
        # new_h, new_c = self.lstm(inputs, (new_h, new_c))

        return (new_h, new_c)

class ODELSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size):
        super(ODELSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.cell = ODELSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(self.hidden_size, 3)

    def forward(self, x, timespans):
        batch_size = x.size(0)
        seq_len = x.size(1)

        hidden_state = (
            torch.zeros((batch_size, self.hidden_size)),
            torch.zeros((batch_size, self.hidden_size)),
        )

        outputs = []
        last_output = torch.zeros((batch_size, 1))
        for t in range(1, seq_len):
            inputs = x[:,t].view(-1, 1)
            # ts = torch.stack([timespans[:, t].squeeze(), last_ts])
            ts = timespans[:,t-1:t+1]
            
            # ts = timespans[:, t].squeeze()
            # ts = torch.unsqueeze(timespans[:, t], 1)
            hidden_state = self.cell.forward(inputs, hidden_state, ts)
            current_output = F.relu(self.fc(hidden_state[0]))
            outputs.append(current_output)
            last_output = current_output 
        
        outputs = torch.stack(outputs, dim=1)

        return last_output
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x[:,:,1], x[:,:,0])
        labels = torch.zeros(logits.shape)
        for i, l in enumerate(labels):
            l[int(y[i].item())] = 1

        # for log in logits:
        #     print(log.argmax(dim=0, keepdim=True))
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(logits, labels)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, labels = batch
        logits = self.forward(x[:,:,1], x[:,:,0])
        total = 0
        correct = 0
        for i in range(len(logits)):
            pred = logits[i].argmax(dim=0, keepdim=True)
            print(pred)
            if (pred[0] == labels[i]):
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
        return torch.optim.Adam(self.parameters(), lr=1e-4)
        

if __name__ == '__main__':
    # print(torch.cuda.is_available())
    data_module = GBPUSDDataModule(window=50, batch_size=4, time_projection=10, with_time=False)
    model = LSTM(input_size=1, hidden_size=100, seq_len=50)
    # model = ODELSTM(1, hidden_size=100)
    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(model, data_module)
    trainer.test()

    
    
    