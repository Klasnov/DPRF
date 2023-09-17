import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .base import BaseClient, BaseServer


class pFedMeClient(BaseClient):
    def __init__(self, client_id: int, algorithm: str, dataset: str, device: str, model: nn.Module,
                 local_epochs: int, local_batch_size: int, lamda: float, k: int, lr_local: float):
        super().__init__(client_id, algorithm, dataset, device, model, local_epochs, local_batch_size, lr_local)
        self.global_model = deepcopy(list(model.parameters()))
        self.lamda: float = lamda
        self.k: int = k

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
            regular_term = self.lamda * (param_theta - param_local)
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
            param_local.data -= self.lr_local * self.lamda * (param_local - param_per)

    def local_train(self) -> None:
        self.local_update.clear()
        for i in range(self.local_epochs):
            self.update_per_model(batch_idx=i)
            self.update_local_model()
    
    def get_local_model(self) -> nn.Module:
        return self.local_model


class pFedMeServer(BaseServer):
    def __init__(self, algorithm: str, dataset: str, device: str, model: nn.Module, lr_g: float,
                 user_selection_ratio: float, round: int, beta: float):
        super().__init__(algorithm, dataset, device, model, lr_g, user_selection_ratio, round)
        self.clients: list[pFedMeClient] = []
        self.beta = beta

    def update_global_model(self) -> None:
        pass

    def global_train(self, save_name_addition: str) -> None:
        for i in range(self.round):
            self.send_global_model()
            for client in self.clients:
                client.local_train()
            self.update_global_model()
            self.model_evaluate()
            self.model_per_evaluate()
            self.model_global_test()

            print("####### Round %d (%.3f%%) ########" % ((i + 1), (i + 1) * 100 / self.round))
            print("  - train_acc = %.4f%%" % (self.train_accuracies[i] * 100))
            print("  - test_acc = %.4f%%" % (self.test_accuracies[i] * 100))
            print("  - peronal_acc = %.4f%%" % (self.personalized_accuracies[i] * 100))
            print()

        self.save_result(save_name_addition)
        # self.save_model(save_name_addition)
