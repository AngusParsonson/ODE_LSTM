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

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size=128
size=28
path_to_data="data/mnist_data"

class MNISTDataModule(pl.LightningDataModule):
    # 469 batches with batch_size=128
    # each image is 1*28*28, 1 represents the fact that there is only one
    # colour channel (greyscale)

    def train_dataloader(self):
        mnist_dataset_train = datasets.EMNIST(path_to_data,
            split='mnist', 
            train=True,
            download=True, 
            transform=transforms.ToTensor(), 
            target_transform=None)

        train_loader = DataLoader(mnist_dataset_train, batch_size=batch_size, shuffle=True)
        return train_loader
    
    def test_dataloader(self):
        mnist_dataset_test = datasets.EMNIST(path_to_data,
            split='mnist', 
            train=False,
            download=False, 
            transform=transforms.ToTensor(), 
            target_transform=None)
        
        test_loader  = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False)
        return test_loader

class ODE(pl.LightningModule):
    def __init__(self, model:nn.Module, sensitivity='adjoint', solver='rk4'):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        
        labels = F.one_hot(labels, 10)
        labels = labels.to(torch.float32)
        # batch = inputs.shape[0]
        # inputs = inputs.view(batch, -1)
        logits = self(inputs)
        
        criterion = nn.MSELoss()
        loss = criterion(logits, labels)
        return loss

    def test_step(self, test_batch, batch_idx):
        inputs, labels = test_batch
        # batch = inputs.shape[0]
        # inputs = inputs.view(batch, -1) 

        logits = self(inputs)
        total = 0
        correct = 0
        for i in range(logits[0].shape[0]):
            pred = logits[i].argmax(dim=0, keepdim=True)
            if (pred[0] == labels[i]):
                correct += 1
            total += 1

        metrics = {'correct': correct, 'total': total}
        return metrics

    def test_epoch_end(self, outputs):
        correct = sum([x['correct'] for x in outputs])
        total = sum([x['total'] for x in outputs])
        
        return {'overall_accuracy': 100*correct/total}

    def configure_optimizers(self):
        optimiser = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        return optimiser

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 392)
        self.bn1 = nn.BatchNorm1d(392)
        self.fc2 = nn.Linear(392, 28)
        self.bn2 = nn.BatchNorm1d(28)
        self.fc3 = nn.Linear(28, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(batch, -1) 
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        
        x = self.softmax(x)
        return x

if __name__ == '__main__':
    net = Net()
    f = nn.Sequential(
        nn.Conv2d(1, 1, 3, padding=1),
        nn.Tanh()
    )
    # neuralDE = NeuralSDE(f, solver='rk4', sensitivity='autograd')
    neuralDE = NeuralDE(f, solver='rk4', sensitivity='autograd')
    data_module = MNISTDataModule()
    model = nn.Sequential(neuralDE, net)

    learner = ODE(model)
    trainer = pl.Trainer(max_epochs=10) # fast_dev_run boolean is useful

    trainer.fit(learner, data_module)
    trainer.test()