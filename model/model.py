import torch.nn as nn


class Minst_Model(nn.Module):
    def __init__(self):
        super(Minst_Model, self).__init__()
        input_size = 784
        hidden_size = 128
        output_size = 10
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
