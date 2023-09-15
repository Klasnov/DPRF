import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .base import BaseClient, BaseServer


class pFMeMoClient(BaseClient):
    def __init__(self, client_id, dataset, model: nn.Module, local_epochs, local_batch_size, alpha, delta, lr_p, lr_l):
        super().__init__(client_id, dataset, model, local_epochs, local_batch_size)
        self.global_model = deepcopy(list(model.parameters()))
        self.alpha: float = alpha
        self.delta: float = delta
        self.lr_p: float = lr_p
        self.lr_l: float = lr_l
        self.local_update: list[torch.Tensor] = list()

    def calculate_grad_per(self, batch_idx: int) -> list[torch.Tensor]:
        grads: list[torch.Tensor] = []
        self.personal_model.train()
        self.personal_model.zero_grad()
        for i, (inputs, labels) in enumerate(self.train_dataloader):
            if i == batch_idx:
                outputs = self.personal_model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                for parm in self.personal_model.parameters():
                    grads.append(parm.grad)
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
        while True:
            grad_h = self.calculate_grad_local(batch_idx)
            grad_h_cat: list[torch.Tensor] = []
            for g in grad_h:
                grad_h_cat.append(g.flatten())
            grad_h_cat = torch.cat(grad_h_cat)

            norm = torch.norm(grad_h_cat)
            if norm ** 2 <= self.delta ** 2:
                break

            for param_per, grad in zip(self.personal_model.parameters(), grad_h):
                param_per.data -= self.lr_p * grad

    def update_local_model(self) -> None:
        for param_local, param_per in zip(self.local_model.parameters(), self.personal_model.parameters()):
            param_local.data -= self.lr_l * self.alpha * (param_local - param_per)

    def local_train(self) -> None:
        self.local_update.clear()
        for i in range(self.local_epochs):
            self.update_per_model(batch_idx=i)
            self.update_local_model()
        for param_local, param_global in zip(self.local_model.parameters(), self.global_model):
            self.local_update.append((param_local - param_global) / self.local_epochs)
    
    def get_update(self) -> list[torch.Tensor]:
        return self.local_update

class pFMeMoServer(BaseServer):
    def __init__(self, algorithm, dataset, model, lr_g, user_selection_ratio, round):
        super().__init__(algorithm, dataset, model, lr_g, user_selection_ratio, round)
        self.lamda = None
    
    def calculate_weights(self) -> tuple[list[list[torch.Tensor]], torch.Tensor]:
        clients_selected: list[pFMeMoClient] = self.select_clients()
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
        norm_square_updates = [torch.norm(norm_grad, p=2) ** 2 for norm_grad in norm_updates]
        num_nonzero = 0
        for update in norm_square_updates:
            if update != 0:
                num_nonzero += 1

        weights = torch.zeros(len(clients_selected))
        if num_nonzero >= 2:
            for i in range(len(weights)):
                if norm_square_updates[i] == 0:
                    continue
                sum = 0
                for j in range(len(norm_square_updates)):
                    if norm_square_updates[j] != 0 and j != i:
                        sum += 1 / norm_square_updates[j]
                weights[i] = (1 / norm_square_updates[i]) / sum
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
            global_param.data += self.global_learning_rate * global_update

    def global_train(self, save_name_addition: str) -> None:
        for _ in range(self.round):
            self.send_global_model()
            for client in self.clients:
                client.local_train()
            self.update_global_model()
            self.model_evaluate()
            self.model_per_evaluate()
            self.model_global_test()
        self.save_result(save_name_addition)
