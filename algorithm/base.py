import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from abc import ABC, abstractclassmethod
from torch.utils.data import DataLoader, TensorDataset, random_split

class BaseClient(ABC):
    def __init__(self, client_id: int, dataset: str, device: str, model: nn.Module, local_epochs: int, local_batch_size: int):
        self.client_id: int = client_id
        self.dataset: str = dataset
        self.device: str = device
        self.local_epochs: int = local_epochs
        self.local_batch_size: int = local_batch_size
        self.local_model = deepcopy(model).to(self.device)
        self.personal_model = deepcopy(model).to(self.device)

        data = torch.load(f'./data/{dataset}/data/client{client_id}/x.pt')
        labels = torch.load(f'./data/{dataset}/data/client{client_id}/y.pt')
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        train_dataset, test_dataset = random_split(TensorDataset(data, labels), [train_size, test_size])
        self.train_data = train_dataset.dataset.tensors[0]
        self.train_labels = train_dataset.dataset.tensors[1]
        self.test_data = test_dataset.dataset.tensors[0]
        self.test_labels = test_dataset.dataset.tensors[1]
        self.train_dataloader = DataLoader(train_dataset, batch_size=local_batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=local_batch_size, shuffle=False)
        self.train_full_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        self.test_full_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    def set_model(self, new_model_state_dict: dict) -> None:
        self.local_model.load_state_dict(new_model_state_dict)
    
    @abstractclassmethod
    def local_train(self) -> None:
        pass

    def train_inform(self) -> tuple[int, float, int]:
        correct_counts = 0
        losses = []
        total_samples = len(self.train_data)
        self.local_model.eval()
        with torch.no_grad():
            for inputs, labels in self.train_full_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                loss = F.cross_entropy(outputs, labels)
                losses.append(loss.item())
                predictions = torch.argmax(outputs, dim=1)
                correct_count = torch.sum(predictions == labels).item()
                correct_counts += correct_count
        return correct_counts.to("cpu"), np.mean(losses).to("cpu"), total_samples.to("cpu")

    def test_inform(self) -> tuple[int, int]:
        correct_counts = 0
        total_samples = len(self.test_data)
        self.local_model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_full_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                correct_count = torch.sum(predictions == labels).item()
                correct_counts += correct_count
        return correct_counts.to("cpu"), total_samples.to("cpu")

    def test_per_inform(self) -> tuple[int, int]:
        params_backup = deepcopy(self.local_model.state_dict())
        self.local_model.load_state_dict(self.personal_model.state_dict())
        correct_counts = 0
        total_samples = len(self.test_data)
        self.local_model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_full_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                correct_count = torch.sum(predictions == labels).item()
                correct_counts += correct_count
        self.set_model(params_backup)
        return correct_counts.to("cpu"), total_samples.to("cpu")

class BaseServer(ABC):
    def __init__(self, algorithm: str, dataset: str, device: str, model: nn.Module, global_learning_rate: float,
                 selection_ratio: float, round: int):
        self.algorithm: str = algorithm
        self.dataset: str = dataset
        self.device: str = device
        self.global_model = model.to(self.device)
        self.global_learning_rate: float = global_learning_rate
        self.clients: list[BaseClient] = []
        self.selection_ratio: float = selection_ratio
        self.round: int = round
        self.train_accuracies = []
        self.train_losses = []
        self.test_accuracies = []
        self.personalized_accuracies = []
        self.global_accuracies = []
        self.test_data = torch.load(f'./data/{dataset}/data/x_test.pt')
        self.test_labels = torch.load(f'./data/{dataset}/data/y_test.pt')
        self.test_dataloader = DataLoader(TensorDataset(self.test_data, self.test_labels),
                                          batch_size=len(self.test_data), shuffle=False)

    def add_client(self, client: BaseClient) -> None:
        self.clients.append(client)

    def send_global_model(self) -> None:
        for client in self.clients:
            client.set_model(self.global_model.state_dict())

    def select_clients(self) -> list[BaseClient]:
        num_selected_users = int(len(self.clients) * self.selection_ratio)
        if num_selected_users == 0:
            return self.clients
        selected_users = np.random.choice(self.clients, num_selected_users, replace=False)
        return list(selected_users)
    
    @abstractclassmethod
    def global_train(self) -> None:
        pass

    @abstractclassmethod
    def update_global_model(self) -> None:
        pass

    def model_evaluate(self) -> None:
        train_acc = []
        train_loss = []
        train_sample_num = []
        test_acc = []
        test_sample_num = []

        for client in self.clients:
            client_train_acc, client_train_loss, client_train_num = client.train_inform()
            client_test_acc, client_test_num = client.test_inform()
            train_acc.append(client_train_acc)
            train_loss.append(client_train_loss)
            train_sample_num.append(client_train_num)
            test_acc.append(client_test_acc)
            test_sample_num.append(client_test_num)
        
        train_acc = np.asarray(train_acc)
        train_loss = np.asarray(train_loss)
        train_sample_num = np.asarray(train_sample_num)
        test_acc = np.asarray(test_acc)
        test_sample_num = np.asarray(test_sample_num)

        train_total_num = np.sum(train_sample_num)
        test_total_num = np.sum(test_sample_num)

        self.train_accuracies.append(np.sum(train_acc * train_sample_num / train_total_num))
        self.train_losses.append(np.sum(train_loss * train_sample_num / train_total_num))
        self.test_accuracies.append(np.sum(test_acc * test_sample_num / test_total_num))

    def model_per_evaluate(self) -> None:
        test_per_acc = []
        test_sample_num = []
        for client in self.clients:
            per_acc, sample_num = client.test_per_inform()
            test_per_acc.append(per_acc)
            test_sample_num.append(sample_num)
        
        test_per_acc = np.asarray(test_per_acc)
        test_sample_num = np.asarray(test_sample_num)
        test_total_num = np.sum(test_sample_num)
        self.personalized_accuracies.append(np.sum(test_per_acc * test_sample_num / test_total_num))

    def model_global_test(self) -> None:
        correct_counts = 0
        self.global_model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.global_model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                correct_count = torch.sum(predictions == labels).item()
                correct_counts += correct_count
        accuracy = correct_counts / len(self.test_data)
        self.global_accuracies.append(accuracy.to("cpu"))

    def save_result(self, addition: str) -> None:
        result_data = {
            "Train Accuracy": self.train_accuracies,
            "Train Loss": self.train_losses,
            "Test Accuracy": self.test_accuracies,
            "Personalized Accuracy": self.personalized_accuracies,
            "Global Accuracy": self.global_accuracies
        }
        result_df = pd.DataFrame(result_data)
        result_dir = f'./result/{self.algorithm}'
        os.makedirs(result_dir, exist_ok=True)
        result_file = f'{result_dir}/{self.dataset}d_{self.round}r_{addition}.csv'
        result_df.to_csv(result_file, index=False)
