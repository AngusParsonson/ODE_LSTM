import sys
from torchdyn.models import *
from torchdyn.datasets import *
from torchdyn import *

import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl

import matplotlib.pyplot as plt

d = ToyDataset()
X, yn = d.generate(n_samples=512, noise=1e-1, dataset_type='moons')

colours = ['orange', 'blue']
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
for i in range(len(X)):
    ax.scatter(X[i,0], X[i,1], s=1, color=colours[yn[i].int()])

X_train = torch.Tensor(X)
y_train = torch.LongTensor(yn.long())
train = data.TensorDataset(X_train, y_train)
trainloader = data.DataLoader(train, batch_size=len(X), shuffle=True)
plt.show()

class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        print(y.shape)
        print("logits: ")
        print(y_hat.shape)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return trainloader

f = nn.Sequential(
    DepthCat(1),
    nn.Linear(2+1,64), 
    nn.Tanh(), 
    DepthCat(1),
    nn.Linear(64+1,2))

model = NeuralDE(f, sensitivity='adjoint', solver='dopri5')
learn = Learner(model)
trainer = pl.Trainer(min_epochs=100, max_epochs=300)
trainer.fit(learn)

s_span = torch.linspace(0,1,100)
trajectory = model.trajectory(X_train, s_span).detach().cpu()

color=['orange', 'blue']

fig = plt.figure(figsize=(10,2))
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
for i in range(500):
    ax0.plot(s_span, trajectory[:,i,0], color=color[int(yn[i])], alpha=.1);
    ax1.plot(s_span, trajectory[:,i,1], color=color[int(yn[i])], alpha=.1);
ax0.set_xlabel(r"$s$ [Depth]") ; ax0.set_ylabel(r"$h_0(s)$")
ax1.set_xlabel(r"$s$ [Depth]") ; ax1.set_ylabel(r"$z_1(s)$")
ax0.set_title("Dimension 0") ; ax1.set_title("Dimension 1")
plt.show()
