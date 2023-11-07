import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseClient, BaseServer

class FedFomoClient(BaseClient):
    def __init__(self, client_id: int, algorithm: str, dataset: str, device: str, model: nn.Module, local_epoch: int, local_batch_size: int, lr_local: float, client_num: int):
        super().__init__(client_id, algorithm, dataset, device, model, local_epoch, local_batch_size, lr_local)
        self.models: list[nn.Module] = [self.personal_model for _ in range(client_num)]
        self.losses: list[torch.Tensor] = []
        self.loss: torch.Tensor = 0
        self.weights: list[float] = [1 for _ in range(client_num)]
    
    def calculate_losses(self) -> None:
        self.losses.clear()
        for model in self.models:
            model.eval()
            for inputs, labels in self.test_full_dataloader:
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)
                self.losses.append(loss)
        for inputs, labels in self.test_full_dataloader:
            outputs = self.local_model(inputs)
            self.loss = F.cross_entropy(outputs, labels)

    def calculate_weights(self) -> None:
        sum = 0
        for i, model in enumerate(self.models):
            param_list = []
            for param_model, param_local in zip(model.parameters(), self.local_model.parameters()):
                param_list.append((param_model - param_local).flatten())
            norm = torch.norm((torch.cat(param_list)), p=1)
            difference = self.loss - self.losses[i]
            weight = 0
            if difference >= 0 and norm != 0:
                weight = difference / norm
            self.weights[i] = weight
            sum += weight
        if sum != 0:
            for i in range(len(self.weights)):
                self.weights[i] /= sum

    def update_personal_model(self) -> None:
        self.calculate_losses()
        self.calculate_weights()
        suffix_terms: list[torch.Tensor] = [torch.zeros_like(param) for param in self.personal_model.parameters()]
        for i, model in enumerate(self.models):
            for param_suffix, param_other, param_local in zip(suffix_terms, model.parameters(), self.local_model.parameters()):
                param_suffix += self.weights[i] * (param_other.data - param_local.data)
        for param_person, param_local, param_suffix in zip(self.personal_model.parameters(), self.local_model.parameters(), suffix_terms):
            param_person.data = param_local.data + param_suffix
    
    def local_train(self, models: list[nn.Module], indexes: list[int]) -> tuple[nn.Module, torch.Tensor]:
        for model, index in zip(models, indexes):
            self.models[index] = model
        self.update_personal_model()
        optim = torch.optim.SGD(self.personal_model.parameters(), self.lr_local)
        self.personal_model.train()
        for _ in range(self.local_epoch):
            self.personal_model.zero_grad()
            inputs, labels = self.get_train_batch()
            outputs = self.personal_model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optim.step()
        for param_personal, param_local in zip(self.personal_model.parameters(), self.local_model.parameters()):
            param_local.data = param_personal.data
        return self.personal_model, torch.tensor(self.weights)
    
    def get_train_num(self) -> int:
        return len(self.train_full_dataloader)
    

class FedFomoServer(BaseServer):
    def __init__(self, algorithm: str, dataset: str, device: str, model: nn.Module, lr_global: float, selection_ratio: float, round: int, client_num: int):
        super().__init__(algorithm, dataset, device, model, lr_global, selection_ratio, round)
        self.clients: list[FedFomoClient] = []
        self.models: list[nn.Module] = [self.global_model for _ in range(client_num)]
        self.p: list[torch.Tensor] = [torch.ones(client_num) for _ in range(client_num)]
        self.down_load_num = int(client_num * self.selection_ratio)
    
    def update_global_model(self) -> None:
        total_train_num = 0
        for client in self.clients:
            total_train_num += client.get_train_num()
        for param_global in self.global_model.parameters():
            param_global.data = torch.zeros_like(param_global).data
        for i, client in enumerate(self.clients):
            for param_global, param_client in zip(self.global_model.parameters(), self.models[i].parameters()):
                param_global.data += param_client * client.get_train_num() / total_train_num

    def global_train(self) -> None:
        for i in range(self.round):
            for j, client in enumerate(self.clients):
                _, indexs = torch.topk(self.p[j], self.down_load_num)
                models = []
                for index in indexs:
                    models.append(self.models[index])
                model, weights = client.local_train(models, indexs)
                self.models[j] = model
                if i != 0:
                    self.p[j] = weights
            self.update_global_model()
            self.model_evaluate()
            self.model_per_evaluate()
            self.model_global_test()
            print("####### Round %d (%.3f%%) ########" % ((i + 1), (i + 1) * 100 / self.round))
            print("  - trai_acc = %.4f%%" % (self.train_accuracies[i] * 100))
            print("  - locl_acc = %.4f%%" % (self.local_accuracies[i] * 100))
            print("  - pern_acc = %.4f%%" % (self.personalized_accuracies[i] * 100))
            print("  - glob_acc = %.4f%%" % (self.global_accuracies[i] * 100))
            print()
