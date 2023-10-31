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
    def __init__(self, client_id: int, algorithm: str, dataset: str, device: str, model: nn.Module,
                 local_epoch: int, local_batch_size: int, lr_local: float):
        self.client_id: int = client_id
        self.algorithm: str = algorithm
        self.dataset: str = dataset
        self.device: str = device
        self.local_epoch: int = local_epoch
        self.local_batch_size: int = local_batch_size
        self.lr_local: float = lr_local
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
        self.iter_train = iter(self.train_dataloader)
        self.iter_test = iter(self.test_dataloader)

    def get_train_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        inputs: torch.Tensor
        labels: torch.Tensor
        try:
            inputs, labels = next(self.iter_train)
        except:
            self.iter_train = iter(self.train_dataloader)
            inputs, labels = next(self.iter_train)
        return inputs.to(self.device), labels.to(self.device)
    
    def set_model(self, new_model_state_dict: dict) -> None:
        self.local_model.load_state_dict(new_model_state_dict)
    
    @abstractclassmethod
    def local_train(self) -> None:
        pass

    def train_inform(self) -> tuple[float, float, int]:
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
                correct_counts += torch.sum(predictions == labels).item()
        return (correct_counts / total_samples), np.mean(losses), total_samples

    def test_inform(self) -> tuple[float, int]:
        correct_counts = 0
        total_samples = len(self.test_data)
        self.local_model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_full_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                correct_counts += torch.sum(predictions == labels).item()
        return (correct_counts / total_samples), total_samples

    def test_per_inform(self) -> tuple[float, int]:
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
                correct_counts += torch.sum(predictions == labels).item()
        self.set_model(params_backup)
        return (correct_counts / total_samples), total_samples
    
    def save_local_model(self, client_addition: str = "") -> None:
        client_model_dir = f"./model/{self.algorithm}/clients"
        if not os.path.exists(client_model_dir):
            os.makedirs(client_model_dir)
        client_model_path = f"{client_model_dir}/{self.dataset}_{self.client_id}id_{self.local_epoch}epc"
        client_model_path = f"{client_model_path}_{self.local_batch_size}bch_{self.lr_local}ll{client_addition}_local.pt"
        torch.save(self.local_model.state_dict(), client_model_path)
    
    def save_personal_model(self, client_addition: str = "") -> None:
        client_model_dir = f"./model/{self.algorithm}/clients"
        if not os.path.exists(client_model_dir):
            os.makedirs(client_model_dir)
        client_model_path = f"{client_model_dir}/{self.dataset}_{self.client_id}id_{self.local_epoch}epc"
        client_model_path = f"{client_model_path}_{self.local_batch_size}bch_{self.lr_local}ll{client_addition}_personal.pt"
        torch.save(self.personal_model.state_dict(), client_model_path)
    
    def load_local_model(self, client_addition: str = "") -> None:
        client_model_dir = f"./model/{self.algorithm}/clients"
        client_model_path = f"{client_model_dir}/{self.dataset}_{self.client_id}id_{self.local_epoch}epc"
        client_model_path = f"{client_model_path}_{self.local_batch_size}bch_{self.lr_local}ll{client_addition}_local.pt"
        client_state_dict = torch.load(client_model_path)
        self.local_model.load_state_dict(client_state_dict)
    
    def load_personal_model(self, client_addition: str = "") ->None:
        client_model_dir = f"./model/{self.algorithm}/clients"
        client_model_path = f"{client_model_dir}/{self.dataset}_{self.client_id}id_{self.local_epoch}epc"
        client_model_path = f"{client_model_path}_{self.local_batch_size}bch_{self.lr_local}ll{client_addition}_personal.pt"
        client_state_dict = torch.load(client_model_path)
        self.personal_model.load_state_dict(client_state_dict)


class BaseServer(ABC):
    def __init__(self, algorithm: str, dataset: str, device: str, model: nn.Module, lr_global: float,
                 selection_ratio: float, round: int):
        self.algorithm: str = algorithm
        self.dataset: str = dataset
        self.device: str = device
        self.global_model = model.to(self.device)
        self.lr_global: float = lr_global
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
        test_acc = np.asarray(test_acc)

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
        self.global_accuracies.append(accuracy)

    def save_result(self, server_addition: str = "", client_addition: str = "") -> None:
        result_data = {
            "Train Accuracy": self.train_accuracies,
            "Train Loss": self.train_losses,
            "Test Accuracy": self.test_accuracies,
            "Personalized Accuracy": self.personalized_accuracies,
            "Global Accuracy": self.global_accuracies
        }
        result_df = pd.DataFrame(result_data)
        result_dir = f"./result/{self.algorithm}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        server_settings = f"{self.dataset}d_{self.round}r_{self.lr_global}lg{server_addition}"
        client = self.clients[0]
        client_settings = f"{client.local_epoch}epc_{client.local_batch_size}bch_{client.lr_local}ll{client_addition}"
        result_file = f"{result_dir}/{server_settings}_{client_settings}.csv"
        result_df.to_csv(result_file, index=False)
    
    def save_model(self, server_addition: str = "", client_addition: str = "") -> None:
        server_model_dir = f"./model/{self.algorithm}"
        if not os.path.exists(server_model_dir):
            os.makedirs(server_model_dir)
        server_model_path = f"{server_model_dir}/{self.dataset}d_{self.round}r_{self.lr_global}lg{server_addition}.pt"
        torch.save(self.global_model.state_dict(), server_model_path)
        for client in self.clients:
            client.save_local_model(client_addition)
            client.save_personal_model(client_addition)
    
    def load_model(self, server_addition: str = "", client_addition: str = "") -> None:
        server_model_dir = f"./model/{self.algorithm}"
        server_model_path = f"{server_model_dir}/{self.dataset}d_{self.round}r_{self.lr_global}lg{server_addition}.pt"
        server_state_dict = torch.load(server_model_path)
        self.global_model.load_state_dict(server_state_dict)
        for client in self.clients:
            client.load_local_model(client_addition)
            client.load_personal_model(client_addition)
    
    def global_train(self) -> None:
        for i in range(self.round):
            self.send_global_model()
            for client in self.clients:
                client.local_train()
            self.update_global_model()
            self.model_evaluate()
            self.model_per_evaluate()
            self.model_global_test()
            print("####### Round %d (%.3f%%) ########" % ((i + 1), (i + 1) * 100 / self.round))
            print("  - trai_acc = %.4f%%" % (self.train_accuracies[i] * 100))
            print("  - locl_acc = %.4f%%" % (self.test_accuracies[i] * 100))
            print("  - pern_acc = %.4f%%" % (self.personalized_accuracies[i] * 100))
            print("  - glob_acc = %.4f%%" % (self.global_accuracies[i] * 100))
            print()
