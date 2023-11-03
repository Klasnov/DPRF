import torch
import cvxpy as cp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .base import BaseClient, BaseServer

class FedMGDAClient(BaseClient):
    def __init__(self, client_id: int, algorithm: str, dataset: str, device: str, model: nn.Module,
                 local_epoch: int, local_batch_size: int, lr_local: float):
        super().__init__(client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lr_local)
        self.updates: list[torch.Tensor] = []
        self.params_global: list[torch.Tensor] = [torch.zeros_like(param) for param in model.parameters()]
    
    def local_train(self):
        for param_global, param_local in zip(self.params_global, self.local_model.parameters()):
            param_global.data = deepcopy(param_local.data)
        self.updates.clear()
        
        self.local_model.train()
        optim = torch.optim.SGD(self.local_model.parameters(), self.lr_local)
        inputs: torch.Tensor
        labels: torch.Tensor
        for inputs, labels in self.train_dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.local_model(inputs)
            optim.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optim.step()
        
        for param_global, param_local in zip(self.local_model.parameters(), self.params_global):
            self.updates.append(param_global - param_local)
    
    def get_update(self):
        return self.params_global
    
class FedMGDAServer(BaseServer):
    def __init__(self, algorithm: str, dataset: str, device: str, model: nn.Module,
                 lr_global: float, selection_ratio: float, round: int):
        super().__init__(algorithm, dataset, device, model, lr_global, selection_ratio, round)
        self.clients: list[FedMGDAClient] = []
    
    def calculate_weights(self) -> torch.Tensor:
        clients_selected: list[FedMGDAClient] = self.select_clients()
        updates: list[list[torch.Tensor]] = []
        for client in clients_selected:
            updates.append(client.get_update())
        
        updates_flatten: list[torch.Tensor] = []
        for update in updates:
            update_flatten = []
            for param_update in update:
                update_flatten.append(param_update.flatten())
            updates_flatten.append(torch.cat(update_flatten))

        norm_updates: list[np.ndarray] = []
        for update_flatten in updates_flatten:
            norm = torch.norm(update_flatten, p=1)
            if norm == 0:
                zero_update = torch.zeros_like(update_flatten)
                norm_updates.append(zero_update.numpy())
            else:
                norm_updates.append((update_flatten / norm).numpy())
        norm_updates: np.ndarray = np.asarray(norm_updates)
        
        weights = cp.Variable(len(clients_selected))
        objective = cp.Minimize(cp.norm(norm_updates.T @ weights, "fro") ** 2)
        constraints = [weights >= 0, cp.sum(weights) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        return torch.from_numpy(weights.value)
    
    def update_global_model(self):
        pass
