import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class cifarModel(torch.nn.Module):
    def __init__(self):
        super(cifarModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, padding=2)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 1024)
        self.fc2 = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_batch(iter_train, train_data_loader, device):
    try:
        data, label = next(iter_train)
    except StopIteration:
        iter_train = iter(train_data_loader)
        data, label = next(iter_train)
    return data.to(device), label.to(device)


def main(is_noniid=True):
    ID = 0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH = 32
    ROUND = 6000

    if is_noniid:
        train_data = torch.load(f"./data/cifar10/data/client{ID}/x.pt").to(torch.float32)
        train_label = torch.load(f"./data/cifar10/data/client{ID}/y.pt").to(torch.int64)
        test_data = torch.load(f"./data/cifar10/data/x_test.pt").to(torch.float32)
        test_label = torch.load(f"./data/cifar10/data/y_test.pt").to(torch.int64)
        train_data_loader = DataLoader(TensorDataset(train_data, train_label), batch_size=BATCH, shuffle=True)
        iter_train = iter(train_data_loader)
        train_data_loader_full = DataLoader(TensorDataset(train_data, train_label), batch_size=len(train_data), shuffle=True)
        test_data_loader_full = DataLoader(TensorDataset(test_data, test_label), batch_size=len(test_data), shuffle=True)

    else:
        cifar10_trainset = datasets.CIFAR10(root='./data/cifar10/data', train=True, download=True)
        cifar10_testset = datasets.CIFAR10(root='./data/cifar10/data', train=False, download=True)
        train_data = cifar10_trainset.data.transpose(0, 3, 1, 2)
        train_label = np.asarray(cifar10_trainset.targets)
        test_data = cifar10_testset.data.transpose(0, 3, 1, 2)
        test_label = np.asarray(cifar10_testset.targets)
        train_data_loader = DataLoader(TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_label, dtype=torch.int64)), batch_size=BATCH, shuffle=True)
        iter_train = iter(train_data_loader)
        train_data_loader_full = DataLoader(TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_label, dtype=torch.int64)), batch_size=len(train_data), shuffle=True)
        test_data_loader_full = DataLoader(TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_label, dtype=torch.int64)), batch_size=len(test_data), shuffle=True)

    model = cifarModel().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    training_accs = []
    testing_accs = []
    losses = []

    for round in range(ROUND):
        input, label = get_batch(iter_train, train_data_loader, DEVICE)
        model.train()
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        del input, label, output

        model.eval()
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for input, label in train_data_loader_full:
                input = input.to(DEVICE)
                label = label.to(DEVICE)
                output = model(input)
                _, predicted = torch.max(output.data, 1)
                total_train += label.size(0)
                correct_train += (predicted == label).sum().item()
        training_acc = correct_train / total_train * 100
        training_accs.append(training_acc)

        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for input, label in test_data_loader_full:
                input = input.to(DEVICE)
                label = label.to(DEVICE)
                output = model(input)
                _, predicted = torch.max(output.data, 1)
                total_test += label.size(0)
                correct_test += (predicted == label).sum().item()
        testing_acc = correct_test / total_test * 100
        testing_accs.append(testing_acc)

        losses.append(loss.item())

        if round % 10 == 0:
            print("Round %d: train_acc %.4f%%, test_acc %.4f%%, loss %.4f" % (round, training_acc, testing_acc, loss.item()))

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    rounds = np.arange(0, len(training_accs))
    ax1.plot(rounds, training_accs, label="Training Accuracy", color="blue")
    ax1.plot(rounds, testing_accs, label="Testing Accuracy", color="red")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax2.plot(rounds, losses, label="Loss", color="green")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.legend()
    plt.savefig(f"./img/cifar10_cnn_acc_loss.png")


if __name__ == "__main__":
    main(is_noniid=True)
