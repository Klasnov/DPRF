import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import random


def main(visualize=False, chinese=False):
    # Download MNIST dataset
    if not os.path.exists('./data/mnist/data'):
        os.makedirs('./data/mnist/data')
    mnist_trainset = datasets.MNIST(root='./data/mnist/data', train=True, download=True)
    mnist_testset = datasets.MNIST(root='./data/mnist/data', train=False, download=True)

    # Preprocess the dataset
    X_train = mnist_trainset.data.numpy()
    y_train = mnist_trainset.targets.numpy()
    X_test = mnist_testset.data.numpy()
    y_test = mnist_testset.targets.numpy()
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    # Split the dataset into 10 clients
    label_to_indices = {label: np.where(y_train == label)[0] for label in range(10)}
    n_clients = 10
    np.random.seed(6)
    p = np.random.dirichlet(np.repeat(1e2, n_clients))
    allocation_vectors = {label: np.random.choice(n_clients, size=len(indices), p=p) for label, indices in label_to_indices.items()}
    client_data_idx = {i: [] for i in range(n_clients)}
    for label, indices in label_to_indices.items():
        for c in range(n_clients):
            client_indices = indices[allocation_vectors[label] == c]
            client_data_idx[c].extend(client_indices)
    for c in range(n_clients):
        random.shuffle(client_data_idx[c])

    # Save the dataset
    for c in range(n_clients):
        client_dir = './data/mnist/data/client{}'.format(c)
        if not os.path.exists(client_dir):
            os.makedirs(client_dir)
        torch.save(torch.tensor(X_train[client_data_idx[c]]), os.path.join(client_dir, 'x.pt'))
        torch.save(torch.tensor(y_train[client_data_idx[c]]), os.path.join(client_dir, 'y.pt'))
    torch.save(torch.tensor(X_test), './data/mnist/data/x_test.pt')
    torch.save(torch.tensor(y_test), './data/mnist/data/y_test.pt')

    # Visualize the dataset
    if visualize:
        label_dist_matrix = np.zeros((10, n_clients))
        for c in range(n_clients):
            label_counts = np.bincount(y_train[client_data_idx[c]], minlength=10)
            label_dist_matrix[:, c] = label_counts
        # Plot the label distribution
        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        plt.rcParams['legend.fontsize'] = 18
        plt.figure(figsize=(10, 8))
        plt.imshow(label_dist_matrix, cmap='YlGnBu', aspect='auto', origin='upper')
        plt.xticks(np.arange(n_clients), np.arange(n_clients))
        plt.yticks(np.arange(10), np.arange(10))
        plt.ylabel('Label')
        plt.xlabel('Client')
        plt.title('Label Distribution of MNIST Dataset across Clients')
        if chinese:
            # 设置中文字体
            plt.colorbar(label='标签计数')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.title('MNIST数据集在客户端上的标签分布')
            plt.ylabel('标签')
            plt.xlabel('客户端')
            plt.savefig('./img/mnist数据分布.png')
        else:
            # Set English font
            plt.colorbar(label='Label Count')
            plt.savefig('./img/mnist_label_distribution.png')
        

if __name__ == '__main__':
    main(visualize=False)
