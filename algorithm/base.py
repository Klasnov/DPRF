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
        """
        Initializes a client object with the following parameters.

        Args:
            - client_id (int): Unique identifier for the client.
            - dataset (str): Name of the dataset used by the client.
            - device (str): Specifies the device on which the client runs, e.g., "cpu" or "cuda".
            - model (nn.Module): The client's local model, a PyTorch neural network model.
            - local_epochs (int): Number of local training epochs for the client.
            - local_batch_size (int): Batch size for local training.
        
        This constructor performs the following actions:
            1. Initializes client attributes such as client_id, dataset, device, and local training parameters.
            2. Creates two deep copies of the provided model for local and personal training.
            3. Loads training data and labels from the client's data directory.
            4. Splits the data into training and testing sets.
            5. Creates data loaders for training and testing.
        """
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
        """
        Sets the local model's state dictionary with a new model's state dictionary.

        Args:
            - new_model_state_dict (dict): A dictionary containing the state of a model.
            
        This method is used to update the local model of the client with a new model's state dictionary.
        It is typically used during the federated learning process when the global model is sent to the client,
        and the client needs to adopt the global model's parameters.
        """
        self.local_model.load_state_dict(new_model_state_dict)
    
    @abstractclassmethod
    def local_train(self) -> None:
        """
        Abstract method for performing local training on the client's local dataset.

        This method should be implemented in subclasses to define the specific training logic for the client.
        During local training, the client's local model should learn from its local dataset for a certain number
        of epochs (specified by self.local_epochs).
        """
        pass

    def train_inform(self) -> tuple[int, float, int]:
        """
        Calculates training statistics for the client's local model.

        Returns:
            - correct_counts (int): Total number of correctly classified samples in the training set.
            - mean_loss (float): Mean loss (error) of the local model on the training set.
            - total_samples (int): Total number of samples in the training set.

        This method evaluates the local model's performance on the entire training dataset
        and returns the number of correctly classified samples, mean loss, and the total number of samples.
        It is typically used to gather training statistics after local training.
        """
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

    def test_inform(self) -> tuple[int, int]:
        """
        Calculates testing statistics for the client's local model.

        Returns:
            - correct_counts (int): Total number of correctly classified samples in the test set.
            - total_samples (int): Total number of samples in the test set.

        This method evaluates the local model's performance on the entire test dataset
        and returns the number of correctly classified samples and the total number of samples.
        It is typically used to gather testing statistics after testing the local model.
        """
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

    def test_per_inform(self) -> tuple[int, int]:
        """
        Calculates testing statistics for the client's personal local model.

        Returns:
            - correct_counts (int): Total number of correctly classified samples by the personal local model.
            - total_samples (int): Total number of samples in the test set.

        This method evaluates the client's personal local model's performance on the entire test dataset
        and returns the number of correctly classified samples and the total number of samples.
        It is typically used to gather testing statistics for the client's personal local model.
        """
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


class BaseServer(ABC):
    def __init__(self, algorithm: str, dataset: str, device: str, model: nn.Module, lr_global: float,
                 selection_ratio: float, round: int):
        """
        Initializes a server object with the following parameters.

        Args:
            - algorithm (str): Name of the federated learning algorithm.
            - dataset (str): Name of the dataset used for federated learning.
            - device (str): Specifies the device on which the server runs, e.g., "cpu" or "cuda:0".
            - model (nn.Module): The global model used in federated learning, a PyTorch neural network model.
            - lr_global (float): Learning rate for global model updates.
            - selection_ratio (float): Ratio of selected clients for each communication round.
            - round (int): Current communication round number.

        This constructor performs the following actions:
            1. Initializes server attributes such as algorithm name, dataset name, and device.
            2. Initializes the global model with the provided model and sends it to the specified device.
            3. Sets the global learning rate for model updates.
            4. Initializes an empty list to store client objects.
            5. Sets the selection ratio for client selection during communication rounds.
            6. Tracks the current communication round number.
            7. Loads the test dataset and labels for global model evaluation.
        """
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
        """
        Adds a client to the list of clients associated with the server.

        Args:
            - client (BaseClient): The client object to be added to the server's list of clients.

        This method allows the server to keep track of individual client objects participating in
        the federated learning process.
        Clients are typically added to the server before the start of communication rounds.
        """
        self.clients.append(client)

    def send_global_model(self) -> None:
        """
        Sends the global model to all registered clients.

        This method is responsible for distributing the current global model to all registered clients.
        After receiving the global model, clients typically use it for local training in each communication round.
        """
        for client in self.clients:
            client.set_model(self.global_model.state_dict())

    def select_clients(self) -> list[BaseClient]:
        """
        Selects a subset of clients for communication in a round.

        Returns:
            - selected_clients (list[BaseClient]): List of selected clients for communication.

        This method selects a subset of clients from the registered clients list for participation in a communication round.
        The number of selected clients is determined by the selection_ratio specified during server initialization.
        """
        num_selected_users = int(len(self.clients) * self.selection_ratio)
        if num_selected_users == 0:
            return self.clients
        selected_users = np.random.choice(self.clients, num_selected_users, replace=False)
        return list(selected_users)
    
    @abstractclassmethod
    def global_train(self) -> None:
        """
        Abstract method for performing global model training using client updates.

        This method should be implemented in subclasses to define the logic for aggregating
        client updates and updating the global model. During global training, the server
        collects client updates, aggregates them, and updates the global model accordingly.
        """
        pass

    @abstractclassmethod
    def update_global_model(self) -> None:
        """
        Abstract method for updating the global model based on the aggregated client updates.

        This method should be implemented in subclasses to define the logic for updating
        the global model based on the aggregated client updates obtained during global training.
        """
        pass

    def model_evaluate(self) -> None:
        """
        Evaluates the global model on the training and test data of registered clients.

        This method evaluates the performance of the global model on both the training and test datasets
        of all registered clients. It calculates accuracy and loss statistics for tracking model performance.
        """
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
        """
        Evaluates the personalized local models of registered clients on the test data.

        This method evaluates the performance of the personalized local models of all registered clients
        on the test dataset. It calculates accuracy statistics for tracking the performance of individual
        client models.
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
        self.personalized_accuracies.append(np.sum(test_per_acc * test_sample_num / test_total_num))

    def model_global_test(self) -> None:
        """
        Evaluates the global model on the test data.

        This method evaluates the performance of the global model on the test dataset.
        It calculates accuracy statistics for tracking the overall performance of the global model.
        """
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

    def save_result(self, addition: str) -> None:
        """
        Saves evaluation results and statistics to a CSV file.

        Args:
            - addition (str): A custom addition to the filename for result identification.

        This method saves the evaluation results and statistics, including train accuracy, train loss, test accuracy,
        personalized accuracy, and global accuracy, to a CSV file. The file is saved in a directory specific to
        the algorithm and dataset used in federated learning.
        """
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
        result_file = f"{result_dir}/{self.dataset}d_{self.round}r_{self.lr_global}lg_{addition}.csv"
        result_df.to_csv(result_file, index=False)
    
    def save_model(self, addition: str) -> None:
        """
        Saves the global model's state dictionary to a file.

        Args:
            - addition (str): A custom addition to the filename for model identification.

        This method saves the state dictionary of the global model to a file in a directory specific to
        the algorithm and dataset used in federated learning. The model is saved in PyTorch format (.pt).
        """
        model_dir = f"./model/{self.algorithm}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = f"{model_dir}/{self.dataset}d_{self.round}r_{self.lr_global}lg_{addition}.pt"
        torch.save(self.global_model.state_dict(), model_path)
    
    def load_model(self, model_path) -> None:
        """
        Loads a global model's state dictionary from a file.

        Args:
            - model_path (str): The path to the saved model file.

        This method loads a global model's state dictionary from a specified file. It performs checks to ensure
        that the file exists and is in the correct PyTorch format (.pt) before loading the model weights.
        """
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist.")
        if not model_path.endswith(".pt"):
            raise ValueError(f"{model_path} is not a pytorch model file (.pt)")
        state_dict = torch.load(model_path)
        self.global_model.load_state_dict(state_dict)
