import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import random

if not os.path.exists('./data/minst/data'):
    os.makedirs('./data/minst/data')

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_trainset = datasets.MNIST(root='./data/minst/data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data/minst/data', train=False, download=True, transform=transform)

X_train = mnist_trainset.data.numpy()
y_train = mnist_trainset.targets.numpy()
X_test = mnist_testset.data.numpy()
y_test = mnist_testset.targets.numpy()

# Group data by labels
label_to_indices = {label: np.where(y_train == label)[0] for label in range(10)}

# Generate random allocation vector for each label
n_clients = 10
np.random.seed(1)
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
    client_dir = './data/minst/data/client{}'.format(c)
    if not os.path.exists(client_dir):
        os.makedirs(client_dir)
    torch.save(torch.tensor(X_train[client_data_idx[c]]), os.path.join(client_dir, 'x.pt'))
    torch.save(torch.tensor(y_train[client_data_idx[c]]), os.path.join(client_dir, 'y.pt'))

# Visualize label distribution
label_dist_matrix = np.zeros((n_clients, 10))
for c in range(n_clients):
    label_counts = np.bincount(y_train[client_data_idx[c]], minlength=10)
    label_dist_matrix[c, :] = label_counts

# Plot the heatmap
plt.figure(figsize=(10, 6))
plt.imshow(label_dist_matrix, cmap='YlGnBu', aspect='auto')
plt.colorbar(label='Label Count')
plt.xticks(np.arange(10), np.arange(10))
plt.yticks(np.arange(n_clients), np.arange(n_clients))
plt.xlabel('Label')
plt.ylabel('Client')
plt.title('Label Distribution of MINST Dataset across Clients')
plt.show()
