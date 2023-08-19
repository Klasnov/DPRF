import torch
import os
from torch.utils.data import DataLoader
import copy


class User:
    def __init__(self, device, user_id, dataset, model: torch.nn.Module, train_data, test_data,
                 batch_size=0, local_epochs=0, learning_rate=0):
        self.device = device
        self.user_id = user_id
        self.dataset = dataset
        self.model = copy.deepcopy(model)
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.train_sample_num = len(train_data)
        self.test_sample_num = len(test_data)
        self.train_loader = DataLoader(train_data, self.batch_size)
        self.test_loader = DataLoader(test_data, self.batch_size)
        self.test_loader_full = DataLoader(test_data, self.test_sample_num)
        self.train_loader_full = DataLoader(train_data, self.train_sample_num)
        self.iter_train_loader = iter(self.train_loader)
        self.iter_test_loader = iter(self.test_loader)
        self.loss = torch.nn.CrossEntropyLoss()

    def set_model(self, model):
        for local_param, new_param in zip(self.model.parameters(), model.parameters()):
            local_param.data = new_param.data.clone()
            local_param.grad = None

    def set_parameters(self, new_params):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.test_loader_full:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        return test_acc, self.test_sample_num

    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        for x, y in self.train_loader_full:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        return train_acc, loss, self.train_sample_num

    def get_next_train_batch(self):
        try:
            (X, y) = next(self.iter_train_loader)
        except StopIteration:
            self.iter_train_loader = iter(self.train_loader)
            (X, y) = next(self.iter_train_loader)
        return X.to(self.device), y.to(self.device)

    def get_next_test_batch(self):
        try:
            (X, y) = next(self.iter_test_loader)
        except StopIteration:
            self.iter_test_loader = iter(self.test_loader)
            (X, y) = next(self.iter_test_loader)
        return X.to(self.device), y.to(self.device)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.user_id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
