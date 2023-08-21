import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset, random_split


class BaseClient:
    def __init__(self, client_id: int, dataset: str, model: nn.Module, local_epochs: int, local_batch_size: int):
        """
        Initializes a BaseUser instance.

        Args:
            client_id (int): ID of the client.
            dataset (str): Dataset name.
            model (nn.Module): The base model to use.
            local_epochs (int): Number of local training epochs.
            local_batch_size (int): Batch size for local training.

        Attributes:
            client_id (int): ID of the client.
            dataset (str): Dataset name.
            local_epochs (int): Number of local training epochs.
            local_batch_size (int): Batch size for local training.
            local_model (nn.Module): Local model for client.
            personalized_model (Dict[str, Tensor]): Personalized model parameters.
            train_data (Tensor): Training data.
            train_labels (Tensor): Training labels.
            test_data (Tensor): Test data.
            test_labels (Tensor): Test labels.
            train_dataloader (DataLoader): Dataloader for training data.
            test_dataloader (DataLoader): Dataloader for test data.
            train_full_dataloader (DataLoader): Dataloader for full training data.
            test_full_dataloader (DataLoader): Dataloader for full test data.
        """
        self.client_id = client_id
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        self.local_model = deepcopy(model)
        self.personalized_model = deepcopy(model.state_dict())

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
        """
        Sets the local model's parameters to the given state dictionary.

        Args:
            new_model_state_dict (dict): New state dictionary for the local model.
        """
        self.local_model.load_state_dict(new_model_state_dict)

    def train_inform(self) -> tuple[list[int], list[float], int]:
        """
        Performs training on the full training dataset and records accuracy and loss.

        Returns:
            tuple[list[int], list[float], int]: Correct counts, losses, and total number of samples.
        """
        correct_counts = []
        losses = []
        total_samples = len(self.train_data)

        self.local_model.train()
        with torch.no_grad():
            for inputs, labels in self.train_full_dataloader:
                outputs = self.local_model(inputs)
                loss = F.cross_entropy(outputs, labels)
                losses.append(loss.item())
                predictions = torch.argmax(outputs, dim=1)
                correct_count = torch.sum(predictions == labels).item()
                correct_counts.append(correct_count)

        return correct_counts, losses, total_samples

    def test_inform(self) -> tuple[list[int], int]:
        """
        Performs testing on the full test dataset and records accuracy.

        Returns:
            tuple[list[int], int]: Correct counts and total number of samples.
        """
        correct_counts = []
        total_samples = len(self.test_data)

        self.local_model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_full_dataloader:
                outputs = self.local_model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                correct_count = torch.sum(predictions == labels).item()
                correct_counts.append(correct_count)

        return correct_counts, total_samples

    def test_per_inform(self) -> tuple[list[int], int]:
        """
        Performs testing on the full test dataset using personalized model and records accuracy.

        Returns:
            tuple[list[int], int]: Correct counts and total number of samples.
        """
        backup_params = deepcopy(list(self.local_model.parameters()))
        self.local_model.load_state_dict(self.personalized_model)
        correct_counts = []
        total_samples = len(self.test_data)
        self.local_model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_full_dataloader:
                outputs = self.local_model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                correct_count = torch.sum(predictions == labels).item()
                correct_counts.append(correct_count)

        self.local_model.load_state_dict({name: param for name, param in zip(self.local_model.state_dict().keys(),
                                                                             backup_params)})
        return correct_counts, total_samples


class BaseServer:
    def __init__(self, algorithm: str, dataset: str, model: nn.Module, global_learning_rate: float,
                 user_selection_ratio: float):
        """
        Initialize the BaseServer for federated learning.

        Args:
            algorithm (str): Algorithm name for identification.
            dataset (str): Dataset name.
            model (nn.Module): Global model for federated learning.
            global_learning_rate (float): Global learning rate for model aggregation.
            user_selection_ratio (float): Ratio of users selected for model aggregation.

        Attributes:
            algorithm (str): Algorithm name.
            dataset (str): Dataset name.
            global_model (nn.Module): Global model for federated learning.
            global_learning_rate (float): Global learning rate.
            clients (list): List of participating clients (BaseClient instances).
            user_selection_ratio (float): Ratio of users selected for each model aggregation round.
            train_accuracies (list): List to store training accuracies over rounds.
            train_losses (list): List to store training losses over rounds.
            test_accuracies (list): List to store test accuracies over rounds.
            personalized_accuracies (list): List to store personalized test accuracies over rounds.
            global_accuracies (list): List to store global test accuracies over rounds.
            test_data (Tensor): Test data for global testing.
            test_labels (Tensor): Test labels for global testing.
            test_dataloader (DataLoader): DataLoader for global testing data.
        """
        self.algorithm = algorithm
        self.dataset = dataset
        self.global_model = model
        self.global_learning_rate = global_learning_rate
        self.clients = []
        self.user_selection_ratio = user_selection_ratio
        self.train_accuracies = []
        self.train_losses = []
        self.test_accuracies = []
        self.personalized_accuracies = []
        self.global_accuracies = []
        self.test_data = torch.load(f'./data/{dataset}/data/x_test.pt')
        self.test_labels = torch.load(f'./data/{dataset}/data/y_test.pt')
        self.test_dataloader = DataLoader(TensorDataset(self.test_data, self.test_labels),
                                          batch_size=len(self.test_data), shuffle=False)

    def add_client(self, client: 'BaseClient') -> None:
        """
        Add a client to the list of participating clients.

        Args:
            client (BaseClient): Client instance to be added.
        """
        self.clients.append(client)

    def send_global_model(self) -> None:
        """
        Send the current global model to all participating clients.
        """
        for client in self.clients:
            client.set_model(self.global_model.state_dict())

    def select_users(self) -> list:
        """
        Select users based on the user_selection_ratio.

        Returns:
            list: List of selected clients.
        """
        num_selected_users = int(len(self.clients) * self.user_selection_ratio)
        selected_users = np.random.choice(self.clients, num_selected_users, replace=False)
        return list(selected_users)

    def model_evaluate(self) -> None:
        """
        Evaluate the global model's performance using all participating clients' data.
        Update the training and test accuracies and losses.
        """
        train_acc = []
        train_loss = []
        train_sample_num = []
        test_acc = []
        test_sample_num = []
        for client in self.clients:
            caa, cal, asm = client.train_inform()
            cea, esm = client.test_inform()
            train_acc.append(caa)
            train_loss.append(cal)
            train_sample_num.append(asm)
            test_acc.append(cea)
            test_sample_num.append(esm)
        train_acc = np.asarray(train_acc)
        train_loss = np.asarray(train_loss)
        train_sample_num = np.asarray(train_sample_num)
        test_acc = np.asarray(test_acc)
        test_sample_num = np.asarray(test_sample_num)
        train_total_num = np.sum(train_sample_num)
        test_total_num = np.sum(test_sample_num)
        self.train_accuracies.append(np.sum(train_acc / train_total_num))
        self.train_losses.append(np.sum(train_loss / train_total_num))
        self.test_accuracies.append(np.sum(test_acc / test_total_num))

    def model_per_evaluate(self) -> None:
        """
        Evaluate the personalized models' performance using all participating clients' data.
        Update the personalized accuracies.
        """
        test_per_acc = []
        test_sample_num = []
        for client in self.clients:
            per_acc, sample_num = client.test_per_inform()
            test_per_acc.append(per_acc)
            test_sample_num.append(sample_num)
        
        test_per_acc = np.asarray(test_per_acc)
        test_sample_num = np.asarray(test_sample_num)
        test_total_num = np.sum(test_sample_num)
        self.personalized_accuracies.append(np.sum(test_per_acc / test_total_num))

    def model_global_test(self) -> None:
        """
        Evaluate the global model's performance using global test data.
        Update the global accuracies.
        """
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                outputs = self.global_model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        self.global_accuracies.append(accuracy)

    def save_result(self) -> None:
        """
        Save the evaluation results to a CSV file.
        """
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
        result_file = f'{result_dir}/{self.dataset}.csv'
        result_df.to_csv(result_file, index=False)