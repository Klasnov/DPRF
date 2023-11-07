import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import random

if not os.path.exists('./data/emnist/data'):
    os.makedirs('./data/emnist/data')

# Load EMNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
emnist_trainset = datasets.EMNIST(root='./data/emnist/data', split='bymerge', train=True, download=True, transform=transform)
emnist_testset = datasets.EMNIST(root='./data/emnist/data', split='bymerge', train=False, download=True, transform=transform)

X_train = emnist_trainset.data.numpy()
y_train = emnist_trainset.targets.numpy()
X_test = emnist_testset.data.numpy()
y_test = emnist_testset.targets.numpy()

# Group data by labels
label_to_indices = {label: np.where(y_train == label)[0] for label in range(47)}

# Generate random allocation vector for each label
n_clients = 50
np.random.seed(6)
p = np.random.dirichlet(np.repeat(1e2, n_clients))
allocation_vectors = {label: np.random.choice(n_clients, size=len(indices), p=p) for label, indices in label_to_indices.items()}

# Split training data among clients based on allocation vectors
client_data_idx = {i: [] for i in range(n_clients)}

for label, indices in label_to_indices.items():
    for c in range(n_clients):
        client_indices = indices[allocation_vectors[label] == c]
        client_data_idx[c].extend(client_indices)

# Shuffle data within each client
for c in range(n_clients):
    random.shuffle(client_data_idx[c])

# Save as .pt files
for c in range(n_clients):
    client_dir = './data/emnist/data/client{}'.format(c)
    if not os.path.exists(client_dir):
        os.makedirs(client_dir)
    torch.save(torch.tensor(X_train[client_data_idx[c]]), os.path.join(client_dir, 'x.pt'))
    torch.save(torch.tensor(y_train[client_data_idx[c]]), os.path.join(client_dir, 'y.pt'))
torch.save(torch.tensor(X_test), './data/emnist/data/x_test.pt')
torch.save(torch.tensor(y_test), './data/emnist/data/y_test.pt')
