import torch
import torch.nn.functional as F
from copy import deepcopy
from .base import BaseClient, BaseServer


class pFMeMoClient(BaseClient):
    def __init__(self, client_id, dataset, device, model, local_epochs, local_batch_size, alpha, delta, lr_p, lr_l):
        """
        Initializes a pFMeMoClient instance for personalized federated learning.

        Args:
            - client_id (int): Unique identifier for the client.
            - dataset (str): Name of the dataset used for training.
            - device (str): Device ('cuda' or 'cpu') on which the client operates.
            - model (nn.Module): The deep learning model used by the client.
            - local_epochs (int): Number of local training epochs per round.
            - local_batch_size (int): Batch size for local training.
            - alpha (float): Weighting factor for regularization between global and personal models.
            - delta (float): Threshold for early stopping in local model updates.
            - lr_p (float): Learning rate for personal model updates.
            - lr_l (float): Learning rate for local model updates.

        This constructor initializes a pFMeMoClient object with the specified parameters.
        It sets up client-specific properties and initializes personal and global models.
        """
        super().__init__(client_id, dataset, device, model, local_epochs, local_batch_size)
        self.global_model = deepcopy(list(model.parameters()))
        self.alpha: float = alpha
        self.delta: float = delta
        self.lr_p: float = lr_p
        self.lr_l: float = lr_l
        self.lr_p_init: float = lr_p
        self.lr_l_init: float = lr_l
        self.local_update: list[torch.Tensor] = list()

    def calculate_grad_per(self, batch_idx: int) -> list[torch.Tensor]:
        """
        Calculate gradients for the personal model on a specific batch.

        Args:
            - batch_idx (int): Index of the batch for which gradients are calculated.

        Returns:
            list[torch.Tensor]: List of gradients for each parameter of the personal model.

        This method calculates gradients for the personal model on a specific batch of training data.
        It computes the gradients of the loss with respect to the model parameters using backpropagation.
        """
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
        """
        Calculate local gradients for personal model with regularization terms.

        Args:
            - batch_idx (int): Index of the batch for which gradients are calculated.

        Returns:
            list[torch.Tensor]: List of local gradients for each parameter of the personal model.

        This method calculates local gradients for the personal model with regularization terms
        on a specific batch of training data. It computes gradients of the loss with respect to
        the model parameters using backpropagation and adds regularization terms to the gradients.
        """
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
        # TODO: Adjust the calculation process.
        """
        Update the personal model with early stopping.

        Args:
            - batch_idx (int): Index of the batch for which updates are performed.

        This method updates the personal model with early stopping based on the specified batch index.
        It iteratively calculates the local gradients and updates the personal model parameters
        until the norm of the gradient falls below the given threshold (delta).
        """
        count = 0
        norms = []
        while count < 1e4:
            count += 1
            grad_h = self.calculate_grad_local(batch_idx)
            grad_h_cat: list[torch.Tensor] = []
            for g in grad_h:
                grad_h_cat.append(g.flatten())
            grad_h_cat = torch.cat(grad_h_cat)

            norm = torch.norm(grad_h_cat)
            norms.append(norm)
            if norm <= self.delta:
                break

            for param_per, grad in zip(self.personal_model.parameters(), grad_h):
                param_per.data -= self.lr_p * grad
        # print(f"count = {count} norm = {norm}")
        print(f"norm = {torch.mean(torch.tensor(norms)).item()}")

    def update_local_model(self) -> None:
        """
        Update the personal model with early stopping.

        Args:
            - batch_idx (int): Index of the batch for which updates are performed.

        This method updates the personal model with early stopping based on the specified batch index.
        It iteratively calculates the local gradients and updates the personal model parameters
        until the norm of the gradient falls below the given threshold (delta).
        """
        for param_local, param_per in zip(self.local_model.parameters(), self.personal_model.parameters()):
            param_local.data -= self.lr_l * self.alpha * (param_local - param_per)

    def local_train(self) -> None:
        """
        Perform local training and update the local model.

        This method performs local training for the client's personal model. It iteratively
        updates the personal model with early stopping and incorporates regularization terms.
        After training, it calculates the updates made to the local model and stores them.
        """
        print(f"# Client {self.client_id} Training...")
        self.local_update.clear()
        for i in range(self.local_epochs):
            self.update_per_model(batch_idx=i)
            self.update_local_model()
        for param_local, param_global in zip(self.local_model.parameters(), self.global_model):
            self.local_update.append((param_local - param_global) / self.local_epochs)
    
    def get_update(self) -> list[torch.Tensor]:
        """
        Get the updates made to the local model during local training.

        Returns:
            list[torch.Tensor]: List of updates made to the local model's parameters.

        This method retrieves and returns the updates that were made to the local model
        during the client's local training. These updates represent the difference between
        the local model and the global model.
        """
        return self.local_update
    
    def step_decay(self, epoch: int) -> None:
        """
        Apply exponential decay to self.delta, self.lr_p, and self.lr_l.

        Args:
            - epoch (int): Current epoch during training.

        This method applies exponential decay to self.delta, self.lr_p, and self.lr_l based on the current epoch.
        """
        decay_factor = 0.95
        self.lr_p = self.lr_p_init * (decay_factor ** epoch)
        self.lr_l = self.lr_l_init * (decay_factor ** epoch)


class pFMeMoServer(BaseServer):
    def __init__(self, algorithm, dataset, device, model, lr_g, user_selection_ratio, round):
        """
        Initialize the pFMeMoServer.

        Args:
            - algorithm (str): The name of the algorithm.
            - dataset (str): The name of the dataset.
            - device (str): The device (e.g., 'cuda' or 'cpu') for model training.
            - model (nn.Module): The global model used in the federated learning.
            - lr_g (float): The global learning rate for updating the global model.
            - user_selection_ratio (float): The ratio of selected clients for each round.
            - round (int): The number of communication rounds.

        This constructor initializes the pFMeMoServer with the specified algorithm, dataset,
        device, global model, learning rate, client selection ratio, and communication rounds.
        """
        super().__init__(algorithm, dataset, device, model, lr_g, user_selection_ratio, round)
        self.clients: list[pFMeMoClient] = []
    
    def calculate_weights(self) -> tuple[list[list[torch.Tensor]], torch.Tensor]:
        """
        Calculate client weights for aggregation and normalized updates.

        Returns:
            tuple[list[list[torch.Tensor]], torch.Tensor]:
                - updates (list[list[torch.Tensor]]): List of client updates for aggregation.
                - weights (torch.Tensor): Client weights for aggregation.

        This method calculates client weights for aggregation and retrieves the normalized
        updates from selected clients. The weights are computed based on the normalized L1
        norm of client updates.
        """
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
        """
        Update the global model by aggregating client updates.

        This method updates the global model by aggregating updates from selected clients.
        It computes a weighted average of the client updates based on client weights and
        adjusts the global model accordingly.
        """
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
        print(f"global update = {torch.mean(torch.cat(global_updates_flatten)).item()}")
    
    def global_decay(self, epoch: int) -> None:
        """
        Apply global learning rate decay for the server and the clients.

        Args:
            - epoch (int): Current epoch during training.

        This method applies learning rate decay globally. It can be used to adjust the global
        learning rate of both the server's and the clients'.
        """
        decay_factor = 0.95
        self.lr_global = self.lr_global_init * (decay_factor ** epoch)
        for client in self.clients:
            client.step_decay(epoch)

    def global_train(self, save_name_addition: str) -> None:
        """
        Perform global training rounds in the federated learning process.

        Args:
            save_name_addition (str): An additional string identifier for result saving.

        This method performs multiple rounds of global training in the federated learning process.
        It iteratively communicates with selected clients, updates the global model, evaluates
        model performance, and saves the results.
        """
        for i in range(self.round):
            print(f"\n**********  ROUND {i}  **********")
            self.send_global_model()
            for client in self.clients:
                client.local_train()
            self.update_global_model()
            self.model_evaluate()
            self.model_per_evaluate()
            self.model_global_test()
            if i % 10 == 0:
                self.global_decay(i)
        self.save_result(save_name_addition)
        self.save_model(save_name_addition)