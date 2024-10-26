import torch
import random
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from .base import BaseClient, BaseServer


class APFLClient(BaseClient):
    def __init__(self, client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lr_local, alpha):
        super().__init__(client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lr_local)
        self.alpha = alpha

    def local_train(self):
        self.local_model.train()
        self.personal_model.train()
        self.local_model.zero_grad()
        self.personal_model.zero_grad()
        local_optim = torch.optim.SGD(self.local_model.parameters(), self.lr_local)
        personal_optim = torch.optim.SGD(self.personal_model.parameters(), self.lr_local)
        for inputs, labels in self.train_full_dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.local_model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            local_optim.step()

            outputs = self.personal_model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            personal_optim.step()

            del inputs, labels, outputs, loss
        
        for param_personal, param_local in zip(self.personal_model.parameters(), self.local_model.parameters()):
            param_personal.data = self.alpha * param_personal.data + (1 - self.alpha) * param_local.data
    
    def update_alpha(self):
        self.personal_model.zero_grad()
        self.personal_model.train()
        grads = []
        for inputs, labels in self.train_full_dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.personal_model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            for parm in self.personal_model.parameters():
                grads.append(parm.grad.cpu())
            del inputs, labels, outputs, loss

        grads_flatten = []
        for grad in grads:
            grads_flatten.append(grad.flatten())
        grads_flatten = torch.cat(grads_flatten)
        del grads

        params_per_flatten = []
        for param_personal in self.personal_model.parameters():
            params_per_flatten.append(param_personal.cpu().flatten())
        params_per_flatten = torch.cat(params_per_flatten)

        params_local_flatten = []
        for param_local in self.local_model.parameters():
            params_local_flatten.append(param_local.cpu().flatten())
        params_local_flatten = torch.cat(params_local_flatten)

        self.alpha = self.alpha - self.lr_local * torch.dot((params_per_flatten - params_local_flatten), grads_flatten)

        del grads_flatten, params_per_flatten, params_local_flatten
        torch.cuda.empty_cache()
    
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


class APFLServer(BaseServer):
    def __init__(self, algorithm, dataset, device, model, lr_g, selection_ratio):
        super().__init__(algorithm, dataset, device, model, lr_g, selection_ratio)
        self.clients = []

    def update_global_model(self, clients_selected):
        n = len(clients_selected)
        params_clients = [torch.zeros_like(global_param) for global_param in self.global_model.parameters()]
        for client in clients_selected:
            for param_clients, param_client in zip(params_clients, client.get_local_model().parameters()):
                param_clients += param_client / n
        for para_global, param_clients in zip(self.global_model.parameters(), params_clients):
            para_global.data = param_clients
        del params_clients
    
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

        for i in range(self.round):
            clients_selected = self.select_clients()
            for client in clients_selected:
                client.local_train()
            self.update_global_model(clients_selected)
            
            self.model_evaluate()
            if self.algorithm != "FedMGDA+":
                self.model_per_evaluate()
            self.model_global_evaluate()
            
            self.print_inform(i)

            self.send_global_model()
            for client in self.clients:
                client.update_alpha()
            
            torch.cuda.empty_cache()
