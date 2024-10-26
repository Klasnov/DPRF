import torch
import cvxpy as cp
import numpy as np
import torch.nn.functional as F
from .base import BaseClient, BaseServer

class FedMGDAClient(BaseClient):
    def __init__(self, client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lr_local):
        super().__init__(client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lr_local)
        self.local_update = []
    
    def local_train(self):
        self.local_update.clear()
        self.local_model.train()
        optim = torch.optim.SGD(self.local_model.parameters(), self.lr_local)
        for inputs, labels in self.train_dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.local_model(inputs)
            optim.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optim.step()
            del inputs, labels, outputs
        
        for param_global, param_local in zip(self.local_model.parameters(), self.global_model.parameters()):
            self.local_update.append(param_global - param_local)
    
    def get_update(self):
        if self.malicious_type == 0:
            return self.local_update
        else:
            malicious_update = []
            if self.malicious_type == 1:
                for update in self.local_update:
                    malicious_update.append(update * self.amplifying_factor)
            else:
                for update in self.local_update:
                    if self.malicious_type == 2:
                        random_update = torch.rand_like(update)
                        malicious_update.append(update + random_update)
                    else:
                        malicious_update.append(-update)
            return malicious_update
    
class FedMGDAServer(BaseServer):
    def __init__(self, algorithm, dataset, device, model, lr_g, selection_ratio):
        super().__init__(algorithm, dataset, device, model, lr_g, selection_ratio)
        self.clients = []
    
    def calculate_weights(self):
        clients_selected = self.select_clients()
        updates = []
        for client in clients_selected:
            updates.append(client.get_update())
        
        updates_flatten = []
        for update in updates:
            update_flatten = []
            for param_update in update:
                update_flatten.append(param_update.flatten())
            updates_flatten.append(torch.cat(update_flatten))

        norm_updates = []
        for update_flatten in updates_flatten:
            norm = torch.norm(update_flatten, p=1)
            if norm == 0:
                zero_update = torch.zeros_like(update_flatten)
                norm_updates.append(zero_update.numpy())
            else:
                norm_updates.append((update_flatten / norm).detach().cpu().numpy())
        norm_updates = np.asarray(norm_updates)
        
        weights = cp.Variable(len(clients_selected), nonneg=True)
        objective = cp.Minimize(cp.sum_squares(norm_updates.T @ weights))
        constraints = [weights >= 0, cp.sum(weights) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)

        restored_updates = []
        client_idx = 0
        for update in updates:
            restored_update = []
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
    
    def update_global_model(self):
        clients_updates, weights = self.calculate_weights()
        global_updates = [torch.zeros_like(global_param) for global_param in self.global_model.parameters()]
        for client_updates, weight in zip(clients_updates, weights):
            for global_update, client_update in zip(global_updates, client_updates):
                client_update = client_update.to(self.device)
                global_update += client_update * weight
        
        for global_param, global_update in zip(self.global_model.parameters(), global_updates):
            global_param.data += self.lr_global * global_update
        
        global_updates_flatten = []
        for global_update in global_updates:
            global_updates_flatten.append(global_update.flatten())
