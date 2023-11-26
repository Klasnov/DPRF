import torch
import torch.nn as nn
import torch.nn.functional as F


class Minst_Model(nn.Module):
    def __init__(self):
        super(Minst_Model, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Cifar10_Model(nn.Module):
    def __init__(self):
        super(Cifar10_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.norm = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = self.norm(self.pool(x))
        x = F.relu(self.conv2(x))
        x = self.pool(self.norm(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Emnist_Model(nn.Module):
    def __init__(self):
        super(Emnist_Model, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 27)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
