import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

if not os.path.exists('./data'):
    os.makedirs('./data')

# Load MNIST dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Generate 10 clients
n_clients = 10

# Distribution-based label imbalance
alphas = np.random.dirichlet([0.1] * n_clients, 1)[0]
client_samples = [round(alphas[i] * len(y_train)) for i in range(n_clients)]
client_data_idx = {i: [] for i in range(n_clients)}

for c, num_samples in enumerate(client_samples):
    label_set = set(np.random.choice(y_train, num_samples, replace=False))
    for label in label_set:
        idx = np.where(y_train == label)[0]
        sample_idx = np.random.choice(idx, round(num_samples / len(label_set)), replace=False)
        client_data_idx[c].extend(sample_idx)

# Noise-based feature imbalance
for c in range(n_clients):
    mean = 0
    var = c / n_clients
    num_samples = len(client_data_idx[c])
    gauss = np.random.normal(mean, var, (num_samples, X_train.shape[1]))
    client_data = X_train[client_data_idx[c]]
    client_data += gauss
    X_train[client_data_idx[c]] = client_data

# Save as .pt files
for c in range(n_clients):
    client_dir = './data/client{}'.format(c + 1)
    if not os.path.exists(client_dir):
        os.makedirs(client_dir)
    torch.save(torch.tensor(X_train[client_data_idx[c]]), os.path.join(client_dir, 'x.pt'))
    torch.save(torch.tensor(y_train[client_data_idx[c]]), os.path.join(client_dir, 'y.pt'))

torch.save(torch.tensor(X_test), './data/x_test.pt')
torch.save(torch.tensor(y_test), './data/y_test.pt')
