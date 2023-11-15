import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from .base import BaseClient, BaseServer


class pFedMeClient(BaseClient):
    def __init__(self, client_id, algorithm, dataset, device, model,
                 local_epoch, local_batch_size, lamda, k, lr_local):
        super().__init__(client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lr_local)
        self.lamda = lamda
        self.k = k
    
    def set_p(self, p_matrix):
        self.p = p_matrix

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
        
        params_local_flatten = []
        for param_local in self.local_model.parameters():
            params_local_flatten.append(param_local.flatten())
        params_local_flatten = torch.cat(params_local_flatten)
        params_personal_flatten = []
        for param_personal in self.personal_model.parameters():
            params_personal_flatten.append(param_personal.flatten())
        params_personal_flatten = torch.cat(params_personal_flatten)
        # TODO: Finish the projection part

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
            malicious_model = deepcopy(self.global_model)
            updates = []
            for param_per, param_global in zip(self.personal_model.parameters(), self.global_model.parameters()):
                updates.append(param_per.data - param_global.data)
            for param_malicious, update in zip(malicious_model.parameters(), updates):
                param_malicious.data = param_malicious.data + update * 1e3
            return malicious_model


class pFedMeServer(BaseServer):
    def __init__(self, algorithm, dataset, device, model, lr_g,
                 user_selection_ratio, round, beta):
        super().__init__(algorithm, dataset, device, model, lr_g, user_selection_ratio, round)
        self.beta = beta

    def update_global_model(self):
        clients_selected = self.select_clients()
        s = len(clients_selected)
        params_clients = [torch.zeros_like(global_param) for global_param in self.global_model.parameters()]
        for client in clients_selected:
            for param_clients, param_client in zip(params_clients, client.get_local_model().parameters()):
                param_clients += param_client / s
        for para_global, param_clients in zip(self.global_model.parameters(), params_clients):
            para_global.data = (1 - self.beta) * para_global + self.beta * param_clients
    
    def global_train(self, malicious = False):
        if self.dataset == "mnist":
            dsub = 10
        elif self.dataset == "cifar10":
            dsub = 100
        else:
            dsub = 20
        param_num = sum(param.numel() for param in self.global_model.parameters())
        p = np.random.randn(param_num, dsub)
        norms = np.linalg.norm(p, axis=0)
        self.p = p / norms
        for client in self.clients():
            client.set_p(self.p)

        if malicious:
            self.malicious = True
            client = np.random.choice(self.clients)
            client.set_malicious()

        for i in range(self.round):
            self.send_global_model()
            
            for client in self.clients:
                client.local_train()
            self.update_global_model()
            
            self.model_evaluate()
            self.model_per_evaluate()
            self.model_global_evaluate()

            if (i + 1) % self.decay_round == 0:
                self.lr_decay()
            
            self.print_inform(i)
