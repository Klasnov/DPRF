import torch
import cvxpy as cp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .base import BaseClient, BaseServer


class DPRFClient(BaseClient):
    def __init__(self, client_id: int, algorithm: str, dataset: str, device: str, model: nn.Module,
                 local_epoch: int, local_batch_size: int, lr_local: float, alpha: float, k: int):
        super().__init__(client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lr_local)
        self.global_model = deepcopy(list(model.parameters()))
        self.alpha: float = alpha
        self.k: int = k
        self.local_update: list[torch.Tensor] = []

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
            regular_term = self.alpha * (param_theta - param_local)
            regular_terms.append(regular_term)
        grads_local: list[torch.Tensor] = []
        for per_grad, regular_term in zip(per_grads, regular_terms):
            grad_local = per_grad + regular_term
            grads_local.append(grad_local)
        return grads_local

    def update_per_model(self) -> None:
        self.personal_model.train()
        count = 0
        while count < self.k:
            count += 1
            inputs, labels = self.get_train_batch()
            grad_h = self.calculate_grad_local(inputs, labels)
            for param_per, grad in zip(self.personal_model.parameters(), grad_h):
                param_per.data -= self.lr_local * grad

    def update_local_model(self) -> None:
        for param_local, param_per in zip(self.local_model.parameters(), self.personal_model.parameters()):
            param_local.data -= self.lr_local * self.alpha * (param_local - param_per)

    def local_train(self) -> None:
        self.local_update.clear()
        for _ in range(self.local_epoch):
            self.update_per_model()
            self.update_local_model()
        for param_local, param_global in zip(self.local_model.parameters(), self.global_model):
            self.local_update.append((param_local - param_global) / self.local_epoch)
    
    def get_update(self) -> list[torch.Tensor]:
        if not self.malicious:
            return self.local_update
        else:
            multi_update: list[torch.Tensor] = []
            for update in self.local_update:
                multi_update.append(update * 3)
            return multi_update

class DPRFServer(BaseServer):
    def __init__(self, algorithm: str, dataset: str, device: str, model: nn.Module, lr_g: float,
                 user_selection_ratio: float, round: int):
        super().__init__(algorithm, dataset, device, model, lr_g, user_selection_ratio, round)
        self.clients: list[DPRFClient] = []
    
    def calculate_weights(self) -> tuple[list[list[torch.Tensor]], torch.Tensor]:
        clients_selected: list[DPRFClient] = self.select_clients()
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
                norm_updates.append((update_flatten / norm).detach().numpy())
        norm_updates = np.asarray(norm_updates)
        
        weights = cp.Variable(len(clients_selected))
        objective = cp.Minimize(cp.norm(norm_updates.T @ weights, "fro") ** 2)
        constraints = [weights >= 0, cp.sum(weights) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)

        restored_updates: list[list[torch.Tensor]] = []
        client_idx = 0
        for update in updates:
            restored_update: list[torch.Tensor] = []
            param_idx = 0
            for param_update in update:
                shape = param_update.shape
                flat_size = param_update.numel()
                restored_param_update = torch.from_numpy(norm_updates[client_idx][param_idx : param_idx + flat_size].reshape(shape))
                restored_update.append(restored_param_update)
                param_idx += flat_size
            client_idx += 1
            restored_updates.append(restored_update)

        return restored_updates, torch.from_numpy(weights.value)

    def update_global_model(self) -> None:
        clients_updates, weights = self.calculate_weights()
        global_updates: list[torch.Tensor] = [torch.zeros_like(global_param) for global_param in self.global_model.parameters()]
        for client_updates, weight in zip(clients_updates, weights):
            for global_update, client_update in zip(global_updates, client_updates):
                global_update += client_update * weight
        
        for global_param, global_update in zip(self.global_model.parameters(), global_updates):
            global_param.data += self.lr_global * global_update
        
        global_updates_flatten: list[torch.Tensor] = []
        for global_update in global_updates:
            global_updates_flatten.append(global_update.flatten())
