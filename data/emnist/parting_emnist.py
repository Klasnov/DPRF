import numpy as np
import torch
from torchvision import datasets
import os
import random


def main():
    if not os.path.exists("./data/emnist/data"):
        os.makedirs("./data/emnist/data")

    emnist_trainset = datasets.EMNIST(root="./data/emnist/data", split="balanced", train=True, download=True)
    emnist_testset = datasets.EMNIST(root="./data/emnist/data", split="balanced", train=False, download=True)

    X_train = emnist_trainset.data.numpy()
    X_train = np.expand_dims(X_train, axis=1)
    y_train = emnist_trainset.targets.numpy()
    X_test = emnist_testset.data.numpy()
    X_test = np.expand_dims(X_test, axis=1)
    y_test = emnist_testset.targets.numpy()

    label_to_indices = {label: np.where(y_train == label)[0] for label in range(47)}

    n_clients = 20
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

    for c in range(n_clients):
        client_dir = "./data/emnist/data/client{}".format(c)
        if not os.path.exists(client_dir):
            os.makedirs(client_dir)
        torch.save(torch.tensor(X_train[client_data_idx[c]]), os.path.join(client_dir, "x.pt"))
        torch.save(torch.tensor(y_train[client_data_idx[c]]), os.path.join(client_dir, "y.pt"))
    torch.save(torch.tensor(X_test), "./data/emnist/data/x_test.pt")
    torch.save(torch.tensor(y_test), "./data/emnist/data/y_test.pt")

if __name__ == "__main__":
    main()
