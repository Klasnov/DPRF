import os
import random
import numpy as np
import torch
from torchvision import datasets
import matplotlib.pyplot as plt


def main(visualize = False):
    if not os.path.exists('./data/cifar10/data'):
        os.makedirs('./data/cifar10/data')

    cifar10_trainset = datasets.CIFAR10(root='./data/cifar10/data', train=True, download=True)
    cifar10_testset = datasets.CIFAR10(root='./data/cifar10/data', train=False, download=True)

    X_train = cifar10_trainset.data
    X_train = X_train.transpose(0, 3, 1, 2)
    y_train = np.asarray(cifar10_trainset.targets)
    X_test = cifar10_testset.data
    X_test = X_test.transpose(0, 3, 1, 2)
    y_test = np.asarray(cifar10_testset.targets)

    label_to_indices = {label: np.where(y_train == label)[0] for label in range(10)}

    n_clients = 15
    np.random.seed(0)
    p = np.random.dirichlet(np.repeat(1e2, n_clients))
    allocation_vectors = {label: np.random.choice(n_clients, size=len(indices), p=p) for label, indices in label_to_indices.items()}

    client_data_idx = {i: [] for i in range(n_clients)}

    for label, indices in label_to_indices.items():
        for c in range(n_clients):
            client_indices = indices[allocation_vectors[label] == c]
            client_data_idx[c].extend(client_indices)
    
    for c in range(n_clients):
        random.shuffle(client_data_idx[c])
    
    for c in range(n_clients):
        client_dir = './data/cifar10/data/client{}'.format(c)
        if not os.path.exists(client_dir):
            os.makedirs(client_dir)
        torch.save(torch.tensor(X_train[client_data_idx[c]]), os.path.join(client_dir, 'x.pt'))
        torch.save(torch.tensor(y_train[client_data_idx[c]]), os.path.join(client_dir, 'y.pt'))
    torch.save(torch.tensor(X_test), './data/cifar10/data/x_test.pt')
    torch.save(torch.tensor(y_test), './data/cifar10/data/y_test.pt')

    if visualize:
        label_dist_matrix = np.zeros((10, n_clients))
        for c in range(n_clients):
            label_counts = np.bincount(y_train[client_data_idx[c]], minlength=10)
            label_dist_matrix[:, c] = label_counts

        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        plt.rcParams['legend.fontsize'] = 18
        plt.figure(figsize=(14, 8))
        plt.imshow(label_dist_matrix, cmap='YlGnBu', aspect='auto', origin='upper')
        plt.colorbar(label='Label Count')
        plt.xticks(np.arange(n_clients), np.arange(n_clients))
        plt.yticks(np.arange(10), np.arange(10))
        plt.ylabel('Label')
        plt.xlabel('Client')
        plt.title('Label Distribution of CIFAR-10 Dataset across Clients')
        plt.savefig('./img/cifar10_label_distribution.png')

if __name__ == '__main__':
    main(visualize=False)
