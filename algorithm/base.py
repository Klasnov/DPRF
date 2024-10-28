import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from copy import deepcopy
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset


MALICIOUS = {0: "normal", 1: "amplifying", 2: "noise", 3: "flipping"}


class BaseClient(ABC):
    def __init__(self, client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lr_local):
        self.client_id = client_id
        self.algorithm = algorithm
        self.dataset = dataset
        self.device = device
        self.local_epoch = local_epoch
        self.local_batch_size = local_batch_size
        self.lr_local = lr_local
        self.global_model = deepcopy(model)
        self.local_model = deepcopy(model)
        self.personal_model = deepcopy(model).to(self.device)

        self.train_data = torch.load(f'data/{dataset}/data/client{client_id}/x.pt', weights_only=True).to(torch.float32)
        self.train_labels = torch.load(f'data/{dataset}/data/client{client_id}/y.pt', weights_only=True).to(torch.int64)
        train_dataset = TensorDataset(self.train_data, self.train_labels)

        self.test_data = torch.load(f'data/{dataset}/data/x_test.pt', weights_only=True).to(torch.float32)
        self.test_labels = torch.load(f'data/{dataset}/data/y_test.pt', weights_only=True).to(torch.int64)
        test_dataset = TensorDataset(self.test_data, self.test_labels)

        self.train_dataloader = DataLoader(train_dataset, batch_size=local_batch_size, shuffle=True)
        self.iter_train = iter(self.train_dataloader)
        self.train_full_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        self.test_full_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        self.malicious_type = 0
        self.amplifying_factor = 0

    def get_train_batch(self):
        try:
            inputs, labels = next(self.iter_train)
        except:
            self.iter_train = iter(self.train_dataloader)
            inputs, labels = next(self.iter_train)
        return inputs, labels
    
    def set_model(self, new_model_state_dict):
        self.global_model.load_state_dict(new_model_state_dict)
        self.local_model.load_state_dict(new_model_state_dict)
    
    @abstractmethod
    def local_train(self):
        pass

    def train_inform(self):
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
                del inputs, labels, outputs, loss, predictions
        return (correct_counts / total_samples), np.mean(losses), total_samples

    def test_inform(self):
        correct_counts = 0
        losses = []
        total_samples = len(self.test_data)
        self.local_model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_full_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                loss = F.cross_entropy(outputs, labels)
                losses.append(loss.item())
                predictions = torch.argmax(outputs, dim=1)
                correct_counts += torch.sum(predictions == labels).item()
                del inputs, labels, outputs, loss, predictions
        return (correct_counts / total_samples), np.mean(losses), total_samples

    def test_per_inform(self):
        correct_counts = 0
        losses = []
        total_samples = len(self.test_data)
        self.personal_model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_full_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.personal_model(inputs)
                loss = F.cross_entropy(outputs, labels)
                losses.append(loss.item())
                predictions = torch.argmax(outputs, dim=1)
                correct_counts += torch.sum(predictions == labels).item()
                del inputs, labels, outputs, loss, predictions
        return (correct_counts / total_samples), np.mean(losses), total_samples
    
    def save_local_model(self, client_addition = ""):
        client_model_dir = f"model/{self.algorithm}/clients"
        if not os.path.exists(client_model_dir):
            os.makedirs(client_model_dir)
        client_model_path = f"{client_model_dir}/{self.dataset}_{self.client_id}id_{self.local_epoch}epc"
        client_model_path = f"{client_model_path}_{self.local_batch_size}bch_{self.lr_local}ll{client_addition}_local.pt"
        torch.save(self.local_model.state_dict(), client_model_path)
    
    def save_personal_model(self, client_addition = ""):
        client_model_dir = f"model/{self.algorithm}/clients"
        if not os.path.exists(client_model_dir):
            os.makedirs(client_model_dir)
        client_model_path = f"{client_model_dir}/{self.dataset}_{self.client_id}id_{self.local_epoch}epc"
        client_model_path = f"{client_model_path}_{self.local_batch_size}bch_{self.lr_local}ll{client_addition}_personal.pt"
        torch.save(self.personal_model.state_dict(), client_model_path)
    
    def load_local_model(self, client_addition = ""):
        client_model_dir = f"model/{self.algorithm}/clients"
        client_model_path = f"{client_model_dir}/{self.dataset}_{self.client_id}id_{self.local_epoch}epc"
        client_model_path = f"{client_model_path}_{self.local_batch_size}bch_{self.lr_local}ll{client_addition}_local.pt"
        client_state_dict = torch.load(client_model_path)
        self.local_model.load_state_dict(client_state_dict)
    
    def load_personal_model(self, client_addition = ""):
        client_model_dir = f"model/{self.algorithm}/clients"
        client_model_path = f"{client_model_dir}/{self.dataset}_{self.client_id}id_{self.local_epoch}epc"
        client_model_path = f"{client_model_path}_{self.local_batch_size}bch_{self.lr_local}ll{client_addition}_personal.pt"
        client_state_dict = torch.load(client_model_path)
        self.personal_model.load_state_dict(client_state_dict)
    
    def set_malicious(self, malicious_type, amplifying_factor):
        self.malicious_type = malicious_type
        self.amplifying_factor = amplifying_factor


class BaseServer(ABC):
    def __init__(self, algorithm, dataset, device, model, lr_g, selection_ratio):
        self.algorithm = algorithm
        self.dataset = dataset
        self.device = device
        self.global_model = model.to(self.device)
        self.lr_global = lr_g
        self.clients = []
        self.selection_ratio = selection_ratio
        self.train_accuracies = []
        self.train_losses = []
        self.local_accuracies = []
        self.personal_accuracies = []
        self.global_accuracies = []
        self.test_std_devs = []
        self.test_data = torch.load(f'data/{dataset}/data/x_test.pt', weights_only=True).to(torch.float32)
        self.test_labels = torch.load(f'data/{dataset}/data/y_test.pt', weights_only=True).to(torch.int64)
        self.test_dataloader = DataLoader(TensorDataset(self.test_data, self.test_labels), batch_size=len(self.test_data), shuffle=False)
        self.malicious_type = 0

    def add_client(self, client):
        self.clients.append(client)

    def send_global_model(self):
        for client in self.clients:
            client.set_model(self.global_model.state_dict())

    def select_clients(self):
        num_selected_users = int(len(self.clients) * self.selection_ratio)
        if num_selected_users == 0:
            return self.clients
        selected_users = np.random.choice(self.clients, num_selected_users, replace=False)
        return list(selected_users)

    @abstractmethod
    def update_global_model(self):
        pass

    def model_evaluate(self):
        train_acc = []
        train_loss = []
        train_sample_num = []
        test_acc = []
        test_loss = []
        test_sample_num = []

        for client in self.clients:
            client_train_acc, client_train_loss, client_train_num = client.train_inform()
            client_test_acc, client_test_loss, client_test_num = client.test_inform()
            train_acc.append(client_train_acc)
            train_loss.append(client_train_loss)
            train_sample_num.append(client_train_num)
            test_acc.append(client_test_acc)
            test_loss.append(client_test_loss)
            test_sample_num.append(client_test_num)
        
        train_acc = np.asarray(train_acc)
        train_loss = np.asarray(train_loss)
        test_acc = np.asarray(test_acc)
        train_sample_num = np.asarray(train_sample_num)
        test_sample_num = np.asarray(test_sample_num)
        test_loss = np.asarray(test_loss)

        train_total_num = np.sum(train_sample_num)
        test_total_num = np.sum(test_sample_num)

        self.train_accuracies.append(np.sum(train_acc * train_sample_num / train_total_num))
        self.train_losses.append(np.sum(train_loss * train_sample_num / train_total_num))
        self.local_accuracies.append(np.sum(test_acc * test_sample_num / test_total_num))
        
        if self.algorithm == "FedMGDA+":
            self.test_std_devs.append(np.std(test_loss))

    def model_per_evaluate(self):
        test_per_acc = []
        test_per_loss = []
        test_sample_num = []
        for client in self.clients:
            per_acc, per_loss, sample_num = client.test_per_inform()
            test_per_acc.append(per_acc)
            test_per_loss.append(per_loss)
            test_sample_num.append(sample_num)
        
        test_per_acc = np.asarray(test_per_acc)
        test_sample_num = np.asarray(test_sample_num)
        test_per_loss = np.asarray(test_per_loss)
        
        test_total_num = np.sum(test_sample_num)

        self.personal_accuracies.append(np.sum(test_per_acc * test_sample_num / test_total_num))
        self.test_std_devs.append(np.std(test_per_loss))

    def model_global_evaluate(self):
        correct_counts = 0
        self.global_model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.global_model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                correct_count = torch.sum(predictions == labels).item()
                correct_counts += correct_count
        global_acc = correct_counts / len(self.test_data)
        self.global_accuracies.append(global_acc)

    def save_result(self, server_addition="", client_addition="", use_addition=False):
        result_data = {
            "Train Accuracy": self.train_accuracies,
            "Train Loss": self.train_losses,
            "Local Accuracy": self.local_accuracies,
            "Global Accuracy": self.global_accuracies,
            "Loss Standard Deviation": self.test_std_devs
        }
        if self.algorithm != "FedMGDA+":
            result_data["Personal Accuracy"] = self.personal_accuracies
        result_df = pd.DataFrame(result_data)
        result_dir = f"result/{self.dataset}/{MALICIOUS[self.malicious_type]}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if use_addition:
            server_settings = f"{self.dataset}_{self.lr_global}lg{server_addition}"
            client = self.clients[0]
            client_settings = f"{client.local_epoch}epc_{client.local_batch_size}bch_{client.lr_local}ll{client_addition}"
            malicious_str = f"_{self.malicious_type}mali"
            result_file = f"{result_dir}/{server_settings}_{client_settings}{malicious_str}.csv"
        else:
            result_file = f"{result_dir}/{self.algorithm}.csv"

        if self.load_save_model:
            if os.path.exists(result_file):
                existing_df = pd.read_csv(result_file)
                result_df = pd.concat([existing_df, result_df], ignore_index=True)

        result_df.to_csv(result_file, index=False)
    
    def save_model(self, server_addition = "", client_addition = ""):
        server_model_dir = f"model/{self.algorithm}"
        if not os.path.exists(server_model_dir):
            os.makedirs(server_model_dir)
        server_model_path = f"{server_model_dir}/{self.dataset}_{self.lr_global}lg{server_addition}.pt"
        torch.save(self.global_model.state_dict(), server_model_path)
        for client in self.clients:
            client.save_local_model(client_addition)
            client.save_personal_model(client_addition)
    
    def load_model(self, server_addition = "", client_addition = ""):
        server_model_dir = f"model/{self.algorithm}"
        server_model_path = f"{server_model_dir}/{self.dataset}_{self.lr_global}lg{server_addition}.pt"
        if os.path.exists(server_model_path):
            server_state_dict = torch.load(server_model_path)
            self.global_model.load_state_dict(server_state_dict)
            for client in self.clients:
                client.load_local_model(client_addition)
                client.load_personal_model(client_addition)
    
    def print_inform(self, round):
        print("####### Round %d (%.3f%%) ########" % ((round + 1), (round + 1) * 100 / self.round))
        print(f" algorithm {self.algorithm}, dataset {self.dataset}")
        print(f" malicious: {MALICIOUS[self.malicious_type]}")
        print("  - trai_acc = %.4f%%" % (self.train_accuracies[round] * 100))
        print("  - locl_acc = %.4f%%" % (self.local_accuracies[round] * 100))
        if self.algorithm != "FedMGDA+":
            print("  - pern_acc = %.4f%%" % (self.personal_accuracies[round] * 100))
        print("  - glob_acc = %.4f%%" % (self.global_accuracies[round] * 100))
        print()
    
    def global_train(self, round, malicious_type, malicious_ratio, amplifying_factor, load_save_model):
        self.round = round
        self.malicious_type = malicious_type
        self.malicious_ratio = malicious_ratio
        self.amplifying_factor = amplifying_factor
        self.load_save_model = load_save_model

        if malicious_type != 0:
            client = np.random.choice(self.clients)
            client.set_malicious(malicious_type, self.amplifying_factor)

        for i in range(self.round):
            self.send_global_model()
            
            for client in self.clients:
                client.local_train()
            self.update_global_model()
            
            self.model_evaluate()
            if self.algorithm != "FedMGDA+":
                self.model_per_evaluate()
            self.model_global_evaluate()
            
            self.print_inform(i)

            torch.cuda.empty_cache()
