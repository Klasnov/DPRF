import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EmnistModel(nn.Module):
    def __init__(self):
        super(EmnistModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, padding=2)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = torch.nn.Linear(1024, 47)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Cifar10Model(torch.nn.Module):
    def __init__(self):
        super(Cifar10Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, padding=2)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
