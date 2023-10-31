import torch
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
        self.local_update: list[torch.Tensor] = list()

    def calculate_grad_per(self, batch_idx: int) -> list[torch.Tensor]:
        grads: list[torch.Tensor] = []
        self.personal_model.train()
        self.personal_model.zero_grad()
        for i, (inputs, labels) in enumerate(self.train_dataloader):
            if i == batch_idx:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.personal_model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                for parm in self.personal_model.parameters():
                    grads.append(parm.grad)
                break
        return grads

    def calculate_grad_local(self, batch_idx: int) -> list[torch.Tensor]:
        per_grads = self.calculate_grad_per(batch_idx)
        regular_terms: list[torch.Tensor] = []
        for param_local, param_theta in zip(self.local_model.parameters(), self.personal_model.parameters()):
            regular_term = self.alpha * (param_theta - param_local)
            regular_terms.append(regular_term)
        grads_local: list[torch.Tensor] = []
        for per_grad, regular_term in zip(per_grads, regular_terms):
            grad_local = per_grad + regular_term
            grads_local.append(grad_local)
        return grads_local

    def update_per_model(self, batch_idx: int) -> None:
        count = 0
        while count < self.k:
            count += 1
            grad_h = self.calculate_grad_local(batch_idx)
            for param_per, grad in zip(self.personal_model.parameters(), grad_h):
                param_per.data -= self.lr_local * grad

    def update_local_model(self) -> None:
        for param_local, param_per in zip(self.local_model.parameters(), self.personal_model.parameters()):
            param_local.data -= self.lr_local * self.alpha * (param_local - param_per)

    def local_train(self) -> None:
        self.local_update.clear()
        for i in range(self.local_epoch):
            self.update_per_model(batch_idx=i)
            self.update_local_model()
        for param_local, param_global in zip(self.local_model.parameters(), self.global_model):
            self.local_update.append((param_local - param_global) / self.local_epoch)
    
    def get_update(self) -> list[torch.Tensor]:
        return self.local_update

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
        
        norm_updates: list[torch.Tensor] = []
        for update_flatten in updates_flatten:
            norm = torch.norm(update_flatten, p=1)
            if norm == 0:
                zero_update = torch.zeros_like(update_flatten)
                norm_updates.append(zero_update)
            else:
                norm_updates.append(update_flatten / norm)
        
        norms: list[torch.Tensor] = []
        for norm_update in norm_updates:
            norms.append(torch.norm(norm_update, p=1))

        num_nonzero = 0
        for norm in norms:
            if norm != 0:
                num_nonzero += 1

        weights = torch.zeros(len(clients_selected))
        if num_nonzero >= 2:
            for i in range(len(weights)):
                if norms[i] == 0:
                    continue
                sum = 0
                for j in range(len(norms)):
                    if norms[j] != 0 and j != i:
                        sum += 1 / norms[j]
                weights[i] = (1 / norms[i]) / sum
        else:
            for i in range(len(weights)):
                weights[i] = 1 / len(clients_selected)
        return updates, weights

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
