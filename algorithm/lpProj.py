import torch
import random
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from .base import BaseClient, BaseServer


class lpProjClient(BaseClient):
    def __init__(self, client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lamda, k, lr_local):
        super().__init__(client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lr_local)
        self.lamda = lamda
        self.k = k

    def initialize_projection(self, P, P_inv):
        self.P = P
        self.P_inv = P_inv
    
    def set_projection(self, y):
        self.y = y
    
    def revert_projection(self):
        x = self.P_inv @ self.y
        start = 0
        for param in self.local_model.parameters():
            end = start + param.numel()
            param.data = x[start:end].view(param.size())
            start = end
        for param_local, param_per in zip(self.local_model.parameters(), self.personal_model.parameters()):
            param_local.data = param_per.data

    def projection(self, model):
        x = torch.cat([param.view(-1) for param in model.parameters()]).cpu()
        return self.P @ x

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
        count = 0
        while count < self.k:
            count += 1
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
        self.revert_projection()
        for _ in range(self.local_epoch):
            self.update_per_model()
            self.update_local_model()
    
    def get_projection(self):
        if self.malicious_type == 0:
            return self.projection(self.local_model)
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
            return self.projection(malicious_model)


class lpProjServer(BaseServer):
    def __init__(self, algorithm, dataset, device, model, lr_g, selection_ratio, beta):
        super().__init__(algorithm, dataset, device, model, lr_g, selection_ratio)
        self.clients: list[lpProjClient] = []
        self.beta = beta
    
    def initialize_projection(self):
        total_size = 0
        for param in self.global_model.parameters():
            total_size += param.numel()
        self.subdom = 10

        self.P = torch.randn(int(self.subdom), total_size)

        for i in range(self.subdom):
            self.P[i] /= torch.norm(self.P[i], p=2)

        self.P_inv = torch.pinverse(self.P)
        self.P_inv.to(self.device)

        for client in self.clients:
            client.initialize_projection(self.P, self.P_inv)
    
    def dispatch_projection(self):
        x = torch.cat([param.view(-1) for param in self.global_model.parameters()]).cpu()
        self.y = self.P @ x
        for client in self.clients:
            client.set_projection(self.y)
    
    def revert_projection(self, y_client):
        x = self.P_inv @ y_client
        start = 0
        client_model = deepcopy(self.global_model)
        for param in client_model.parameters():
            end = start + param.numel()
            param.data = x[start:end].view(param.size())
            start = end
        return client_model

    def update_global_model(self):
        y_clients = torch.zeros_like(self.y)
        for client in self.clients:
            y_clients += client.get_projection() / len(self.clients)
        client_model = self.revert_projection(y_clients)
        client_model.to(self.device)
        del y_clients
        for param_global, param_client in zip(self.global_model.parameters(), client_model.parameters()):
            param_global.data = (1 - self.beta) * param_global + self.beta * param_client
        client_model.cpu()

    def global_train(self, round, malicious_type, malicious_ratio, amplifying_factor, load_save_model):
        self.round = round
        self.malicious_type = malicious_type
        self.malicious_ratio = malicious_ratio
        self.amplifying_factor = amplifying_factor
        self.load_save_model = load_save_model

        if malicious_type == 1:
            client = np.random.choice(self.clients)
            client.set_malicious(malicious_type, self.amplifying_factor)
        elif malicious_type != 0:
            clients_to_set_malicious = random.sample(self.clients, int(len(self.clients) * self.malicious_ratio))
            for client in clients_to_set_malicious:
                client.set_malicious(malicious_type, self.amplifying_factor)

        self.initialize_projection()

        for i in range(self.round):
            self.dispatch_projection()
            
            for client in self.clients:
                client.local_train()
            self.update_global_model()
            
            self.model_evaluate()
            if self.algorithm != "FedMGDA+":
                self.model_per_evaluate()
            self.model_global_evaluate()
            
            self.print_inform(i)

            torch.cuda.empty_cache()
