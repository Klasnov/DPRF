import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import random

if not os.path.exists('./data/cifar10/data'):
    os.makedirs('./data/cifar10/data')

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar10_trainset = datasets.CIFAR10(root='./data/cifar10/data', train=True, download=True, transform=transform)
cifar10_testset = datasets.CIFAR10(root='./data/cifar10/data', train=False, download=True, transform=transform)

X_train = np.array(cifar10_trainset.data)
y_train = np.array(cifar10_trainset.targets)
X_test = np.array(cifar10_testset.data)
y_test = np.array(cifar10_testset.targets)

# Group data by labels
label_to_indices = {label: np.where(y_train == label)[0] for label in range(10)}

# Generate random allocation vector for each label
n_clients = 20
np.random.seed(5)
allocation_vectors = {label: np.random.randint(0, n_clients, size=len(indices)) for label, indices in label_to_indices.items()}

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
    client_dir = './data/cifar10/data/client{}'.format(c)
    if not os.path.exists(client_dir):
        os.makedirs(client_dir)
    torch.save(torch.tensor(X_train[client_data_idx[c]]), os.path.join(client_dir, 'x.pt'))
    torch.save(torch.tensor(y_train[client_data_idx[c]]), os.path.join(client_dir, 'y.pt'))
torch.save(torch.tensor(X_test), './data/cifar10/data/x_test.pt')
torch.save(torch.tensor(y_test), './data/cifar10/data/y_test.pt')

# Visualize label distribution
label_dist_matrix = np.zeros((10, n_clients))  # Swap the dimensions
for c in range(n_clients):
    label_counts = np.bincount(y_train[client_data_idx[c]], minlength=10)
    label_dist_matrix[:, c] = label_counts  # Swap the dimensions

# Plot the heatmap with swapped axes
plt.figure(figsize=(15, 6))
plt.imshow(label_dist_matrix, cmap='YlGnBu', aspect='auto', origin='upper')
plt.colorbar(label='Label Count')
plt.xticks(np.arange(n_clients), np.arange(n_clients))
plt.yticks(np.arange(10), np.arange(10))
plt.ylabel('Label')
plt.xlabel('Client')
plt.title('Label Distribution of CIFAR-10 Dataset across Clients')
plt.show()
