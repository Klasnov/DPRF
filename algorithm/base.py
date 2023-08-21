import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset, random_split


class BaseUser:
    def __init__(self, client_id, dataset, model: nn.Module, local_epochs, local_batch_size):
        self.client_id = client_id
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        self.local_model = deepcopy(model)
        self.personalized_model = deepcopy(list(model.parameters()))

        # Load data for each client separately
        data = torch.load(f'./data/{dataset}/client{client_id}/x.pt')
        labels = torch.load(f'./data/{dataset}/client{client_id}/y.pt')

        # Split data into train and test datasets
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        train_dataset, test_dataset = random_split(TensorDataset(data, labels), [train_size, test_size])

        self.train_data = train_dataset.dataset.tensors[0]
        self.train_labels = train_dataset.dataset.tensors[1]
        self.test_data = test_dataset.dataset.tensors[0]
        self.test_labels = test_dataset.dataset.tensors[1]

        self.train_dataloader = DataLoader(train_dataset, batch_size=local_batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=local_batch_size, shuffle=False)

    def update_model(self, new_model_state_dict):
        self.local_model.load_state_dict(new_model_state_dict)

class BaseServer:
    def __init__(self):
        pass
