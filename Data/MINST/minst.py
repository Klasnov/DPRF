import numpy as np
import torch
from torchvision import datasets


def main():
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)

    x_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()

    x_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    num_users = 10
    num_labels = 10

    train_data = []
    for i in range(num_users):
        train_data.append(torch.tensor([]))

    labels_per_user = 1
    user_labels = []
    for user in range(num_users):
        user_labels.append(np.random.choice(num_labels, labels_per_user, replace=False))

    for i in range(num_users):
        user_idx = np.where(np.isin(y_train, user_labels[i]))[0]
        train_data[i] = torch.from_numpy(x_train[user_idx]).float()

        for i in range(num_users):
            torch.save(train_data[i], f'./data/user_{i}_train.pt')

        torch.save(x_test, './data/x_test.pt')
        torch.save(y_test, './data/y_test.pt')


if __name__ == '__main__':
    main()
