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
    mnist_dataset_train = datasets.EMNIST(path_to_data,
        split='mnist', 
        train=True,
        download=True, 
        transform=transforms.ToTensor(), 
        target_transform=None)
    
    mnist_dataset_test = datasets.EMNIST(path_to_data,
        split='mnist', 
        train=True,
        download=False, 
        transform=transforms.ToTensor(), 
        target_transform=None)

    # 469 batches with batch_size=128
    # each image is 1*28*28, 1 represents the fact that there is only one
    # colour channel (greyscale)
    train_loader = DataLoader(mnist_dataset_train, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=True)
    net = Net()
    criterion = nn.MSELoss()
    optimiser = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # train
    for epoch in range(10):
        loss_log = 0
        for data in train_loader:
            inputs, labels = data
            labels = F.one_hot(labels, 10)
            labels = labels.to(torch.float32)

            optimiser.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            loss_log = loss.item()

        print("epoch " + str(epoch) + " loss: " + str(loss_log))

    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            
            output = net(inputs)
            for i in range(output[0].shape[0]):
                pred = output[i].argmax(dim=0, keepdim=True)
                if (pred[0] == labels[i]):
                    correct += 1
                total += 1
    print(correct)
    print(100*correct/total)
    # test
        
