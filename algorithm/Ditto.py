import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .base import BaseClient, BaseServer

class DittoClient(BaseClient):
    def __init__(self, client_id: int, algorithm: str, dataset: str, device: str, model: nn.Module,
                 local_epoch: int, local_batch_size: int, lamda: float, lr_local: float):
        super().__init__(client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lr_local)
        self.global_model = deepcopy(list(model.parameters()))
        self.lamda: float = lamda

    def calculate_grad_per(self, inputs: torch.Tensor, labels: torch.Tensor) -> list[torch.Tensor]:
        grads: list[torch.Tensor] = []
        self.personal_model.zero_grad()
        outputs = self.personal_model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        for parm in self.personal_model.parameters():
            grads.append(parm.grad)
        return grads

    def calculate_grad_local(self, inputs: torch.Tensor, labels: torch.Tensor) -> list[torch.Tensor]:
        per_grads = self.calculate_grad_per(inputs, labels)
        regular_terms: list[torch.Tensor] = []
        for param_local, param_theta in zip(self.local_model.parameters(), self.personal_model.parameters()):
            regular_term = self.lamda * (param_theta - param_local)
            regular_terms.append(regular_term)
        grads_local: list[torch.Tensor] = []
        for per_grad, regular_term in zip(per_grads, regular_terms):
            grad_local = per_grad + regular_term
            grads_local.append(grad_local)
        return grads_local

    def update_per_model(self) -> None:
        self.personal_model.train()
        inputs, labels = self.get_train_batch()
        grad_h = self.calculate_grad_local(inputs, labels)
        for param_per, grad in zip(self.personal_model.parameters(), grad_h):
            param_per.data -= self.lr_local * grad
            

    def update_local_model(self) -> None:
        for param_local, param_per in zip(self.local_model.parameters(), self.personal_model.parameters()):
            param_local.data -= self.lr_local * self.lamda * (param_local - param_per)

    def local_train(self) -> None:
        for _ in range(self.local_epoch):
            self.update_per_model()
            self.update_local_model()
    
    def get_local_model(self) -> nn.Module:
        if not self.malicious:
            return self.local_model
        else:
            return self.personal_model


class DittoServer(BaseServer):
    def __init__(self, algorithm: str, dataset: str, device: str, model: nn.Module, lr_g: float,
                 user_selection_ratio: float, round: int):
        super().__init__(algorithm, dataset, device, model, lr_g, user_selection_ratio, round)
        self.clients: list[DittoClient] = []

    def update_global_model(self) -> None:
        clients_selected: list[DittoClient] = self.select_clients()
        n = len(clients_selected)
        params_clients: list[torch.Tensor] = [torch.zeros_like(global_param) for global_param in self.global_model.parameters()]
        for client in clients_selected:
            for param_clients, param_client in zip(params_clients, client.get_local_model().parameters()):
                param_clients += param_client / n
        for para_global, param_clients in zip(self.global_model.parameters(), params_clients):
            para_global.data = param_clients
