import torch
import torch.nn.functional as F
from copy import deepcopy
from .base import BaseClient, BaseServer

class DittoClient(BaseClient):
    def __init__(self, client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lamda, lr_local):
        super().__init__(client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lr_local)
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
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        grad_h = self.calculate_grad_local(inputs, labels)
        for param_per, grad in zip(self.personal_model.parameters(), grad_h):
            param_per.data -= self.lr_local * grad
        del inputs, labels, grad_h
            

    def update_local_model(self):
        for param_local, param_per in zip(self.local_model.parameters(), self.personal_model.parameters()):
            param_local.data -= self.lr_local * self.lamda * (param_local - param_per)

    def local_train(self):
        for _ in range(self.local_epoch):
            self.update_per_model()
            self.update_local_model()
    
    def get_local_model(self):
        if self.malicious_type == 0:
            return self.local_model
        else:
            malicious_model = deepcopy(self.global_model)
            if self.malicious_type == 1:
                updates = []
                for param_per, param_global in zip(self.personal_model.parameters(), self.global_model.parameters()):
                    updates.append((param_per.data - param_global.data))
                for param_malicious, update in zip(malicious_model.parameters(), updates):
                    param_malicious.data = param_malicious.data + update * self.amplifying_factor
            else:
                for param_malicious in malicious_model.parameters():
                    if self.malicious_type == 2:
                        param_random = torch.rand_like(param_malicious)
                        param_malicious.data = param_malicious.data + param_random
                    else:
                        param_malicious.data = -param_malicious.data
            return malicious_model


class DittoServer(BaseServer):
    def __init__(self, algorithm, dataset, device, model, lr_g, selection_ratio):
        super().__init__(algorithm, dataset, device, model, lr_g, selection_ratio)
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
