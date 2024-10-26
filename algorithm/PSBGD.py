import torch
from copy import deepcopy
import torch.nn.functional as F
from .base import BaseClient, BaseServer
    


class PSBGDClient(BaseClient):
    def __init__(self, client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, beta, lr_local):
        super().__init__(client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lr_local)
        self.beta = beta
    
    def calculate_grad_per(self, inputs, labels):
        grads = []
        self.personal_model.zero_grad()
        outputs = self.personal_model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        for parm in self.personal_model.parameters():
            grads.append(parm.grad)
        return grads

    def update_per_model(self):
        self.personal_model.train()
        inputs, labels = self.get_train_batch()
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        grad = self.calculate_grad_per(inputs, labels)
        for param_per, param_local, grad in zip(self.personal_model.parameters(), self.local_model.parameters(), grad):
            param_per.data -= self.lr_local * grad + self.beta * (param_per - param_local)
        del inputs, labels, grad

    def local_train(self):
        for _ in range(self.local_epoch):
            self.update_per_model()

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


class PSBGDServer(BaseServer):
    def __init__(self, algorithm, dataset, device, model, lr_g, selection_ratio, beta, mu, t):
        super().__init__(algorithm, dataset, device, model, lr_g, selection_ratio)
        self.clients = []
        self.beta = beta
        self.mu = mu
        self.t = t

    def h_grad(self, local_model):
        diffs = []
        total_param_num = 0
        less_than_t_num = 0
        for param_global, update_local in zip(self.global_model.parameters(), local_model.parameters()):
            diff = param_global - update_local
            total_param_num += diff.numel()
            grad = torch.zeros_like(diff)
            mask = torch.abs(diff) <= self.t
            less_than_t_num += mask.sum().item()
            grad[mask] = diff[mask]
            grad[~mask] = self.t * torch.sign(diff[~mask])
            diffs.append(grad)
        return diffs

    def update_global_model(self):
        clients_update = [torch.zeros_like(param) for param in self.global_model.parameters()]
        for client in self.clients:
            local_model = client.get_local_model()
            client_h_grad = self.h_grad(local_model)
            for client_update, h_grad in zip(clients_update, client_h_grad):
                client_update += h_grad / len(self.clients)
        regular_term = [param * self.mu for param in self.global_model.parameters()]
        for param_global, client_update, regular in zip(self.global_model.parameters(), clients_update, regular_term):
            param_global.data -= self.lr_global * (client_update + regular)
