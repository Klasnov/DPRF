import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .base import BaseClient, BaseServer

class DittoClient(BaseClient):
    def __init__(self, client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lamda, lr_local):
        super().__init__(client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lr_local)
        self.global_model = deepcopy(list(model.parameters()))
        self.lamda = lamda

    def calculate_grad_per(self, inputs, labels):
        grads = []
        self.personal_model.zero_grad()
        outputs = self.personal_model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        for parm in self.personal_model.parameters():
            grads.append(parm.grad)
        return grads

    def calculate_grad_local(self, inputs, labels):
        per_grads = self.calculate_grad_per(inputs, labels)
        regular_terms = []
        for param_local, param_theta in zip(self.local_model.parameters(), self.personal_model.parameters()):
            regular_term = self.lamda * (param_theta - param_local)
            regular_terms.append(regular_term)
        grads_local = []
        for per_grad, regular_term in zip(per_grads, regular_terms):
            grad_local = per_grad + regular_term
            grads_local.append(grad_local)
        return grads_local

    def update_per_model(self):
        self.personal_model.train()
        inputs, labels = self.get_train_batch()
        grad_h = self.calculate_grad_local(inputs, labels)
        for param_per, grad in zip(self.personal_model.parameters(), grad_h):
            param_per.data -= self.lr_local * grad
            

    def update_local_model(self):
        for param_local, param_per in zip(self.local_model.parameters(), self.personal_model.parameters()):
            param_local.data -= self.lr_local * self.lamda * (param_local - param_per)

    def local_train(self):
        for _ in range(self.local_epoch):
            self.update_per_model()
            self.update_local_model()
    
    def get_local_model(self):
        if not self.malicious:
            return self.local_model
        else:
            return self.personal_model


class DittoServer(BaseServer):
    def __init__(self, algorithm, dataset, device, model, lr_g,
                 user_selection_ratio, round):
        super().__init__(algorithm, dataset, device, model, lr_g, user_selection_ratio, round)
        self.clients = []

    def update_global_model(self):
        clients_selected = self.select_clients()
        n = len(clients_selected)
        params_clients = [torch.zeros_like(global_param) for global_param in self.global_model.parameters()]
        for client in clients_selected:
            for param_clients, param_client in zip(params_clients, client.get_local_model().parameters()):
                param_clients += param_client / n
        for para_global, param_clients in zip(self.global_model.parameters(), params_clients):
            para_global.data = param_clients
