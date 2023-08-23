import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseClient, BaseServer


class pFMeMoClient(BaseClient):
    def __init__(self, client_id, dataset, model: nn.Module, local_epochs, local_batch_size, alpha, delta, lr_p, lr_l):
        super().__init__(client_id, dataset, model, local_epochs, local_batch_size)
        self.alpha = alpha  # Regularization parameter for personalized updates
        self.delta = delta  # Maximum squared norm of personalized update
        self.lr_p = lr_p    # Learning rate for projected gradient descent
        self.lr_l = lr_l    # Learning rate for local model updates
        self.local_gradient = {}  # Store local gradients

    def calculate_grad_estimate(self, theta_i, data_loader, batch_idx):
        """
        Calculate an estimate of the gradient of the loss with respect to 'theta_i'
        using the provided data loader and considering only the 'batch_idx' batch.

        Args:
            theta_i (Tensor): Current parameter tensor 'theta_i'.
            data_loader (DataLoader): Data loader for local training data.
            batch_idx (int): Index of the batch to compute the gradient estimate for.

        Returns:
            Tensor: Estimated gradient of the loss with respect to 'theta_i'.
        """
        gradients = []
        for i, (inputs, labels) in enumerate(data_loader):
            if i == batch_idx:
                self.local_model.zero_grad()
                outputs = self.local_model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                gradients.append(theta_i.grad.clone())
        return torch.stack(gradients).mean(dim=0)

    def calculate_h_i(self, theta_i, w_t_i, batch_idx):
        """
        Calculate the personalized gradient update 'tilde_h_i' for the given 'theta_i'
        using the provided 'w_t_i' and considering only the 'batch_idx' batch.

        Args:
            theta_i (Tensor): Current parameter tensor 'theta_i'.
            w_t_i (dict): State dictionary of the global model at time 't'.
            batch_idx (int): Index of the batch to compute the personalized update for.

        Returns:
            Tensor: Personalized gradient update 'tilde_h_i' for 'theta_i'.
        """
        gradient_estimate = self.calculate_grad_estimate(theta_i, self.train_dataloader, batch_idx=batch_idx)
        regularization_term = self.alpha * (theta_i - w_t_i)
        return gradient_estimate + regularization_term

    def update_theta_i(self, theta_i, w_t_i, batch_idx):
        """
        Update 'theta_i' using projected gradient descent based on personalized update.

        Args:
            theta_i (Tensor): Current parameter tensor 'theta_i'.
            w_t_i (dict): State dictionary of the global model at time 't'.
            batch_idx (int): Index of the batch to compute the personalized update for.

        Returns:
            Tensor: Updated 'theta_i' after projected gradient descent.
        """
        theta_i_updated = torch.cat([param.data.view(-1) for param in theta_i])
        while True:
            grad_tilde_h_i = self.calculate_h_i(theta_i_updated, w_t_i, batch_idx=batch_idx)
            norm_grad_tilde_h_i = torch.norm(grad_tilde_h_i)
            if norm_grad_tilde_h_i ** 2 <= self.delta:
                break
            theta_i_updated -= self.lr_p * grad_tilde_h_i
        return theta_i_updated

    def update_local_model(self, w_t_i, batch_idx):
        """
        Update the local model using personalized updates based on 'w_t_i'
        for the given 'batch_idx' batch of local training data.

        Args:
            w_t_i (dict): State dictionary of the global model at time 't'.
            batch_idx (int): Index of the batch to perform local model update on.
        """
        self.local_model.train()
        for i, (inputs, labels) in enumerate(self.train_dataloader):
            if i == batch_idx:
                self.local_model.zero_grad()
                outputs = self.local_model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                grad_local = {name: self.alpha * (w_t_i[name] - param.grad) for name, param in
                              self.local_model.named_parameters()}
                self.local_model.parameters_update(grad_local, lr=self.lr_l)

    def local_train(self):
        """
        Perform local training using personalized updates over 'local_epochs'.

        Returns:
            dict: Local gradient after local training.
        """
        w_t_i = self.local_model.state_dict()
        theta_i = None
        for i in range(self.local_epochs):
            theta_i = self.update_theta_i(self.local_model.parameters(), w_t_i, batch_idx=i)
            self.update_local_model(w_t_i, batch_idx=i)
        self.personalized_model = theta_i.clone().detach()
        for name, param in self.local_model.named_parameters():
            self.local_gradient[name] = param.data - w_t_i[name]
    
    def get_gradient(self):
        """
        Get the local gradients computed during local training.

        Returns:
            dict: A dictionary containing the local gradients of each parameter.
        """
        return self.local_gradient

class pFMeMoServer(BaseServer):
    def __init__(self, algorithm, dataset, model, lr_g, user_selection_ratio, round):
        """
        Initialize the pFMeMoServer for federated learning using the pFMeMo algorithm.

        Args:
            algorithm (str): Algorithm name for identification.
            dataset (str): Dataset name.
            model (nn.Module): Global model for federated learning.
            lr_g (float): Global learning rate for model aggregation.
            user_selection_ratio (float): Ratio of users selected for model aggregation.
            round (int): Total number of federated learning rounds.

        Attributes:
            clients_selected (list): List of selected participating clients for model aggregation.
            gradients (list): List to store gradients received from selected clients.
            lamda (np.array): Weight vector for gradient aggregation.
        """
        super().__init__(algorithm, dataset, model, lr_g, user_selection_ratio, round)
        self.clients_selected = None
        self.gradients = None
        self.lamda = None
    
    def calculate_lambda(self):
        """
        Calculate the weight vector 'lambda_t' using the provided gradients.
        The weight vector is used to aggregate gradients from clients.

        Returns:
            None
        """
        self.gradients = []
        for client in self.clients_selected:
            self.gradients.append(client.get_gradient())
        normalized_gradients = [grad / torch.norm(grad, p=2) for grad in self.gradients]
        gradients_norm_squared = [torch.norm(grad, p=2) ** 2 for grad in normalized_gradients]
        num_nonzero_norms = sum(1 for norm in gradients_norm_squared if norm != 0)
        lambda_t = np.zeros(len(gradients_norm_squared))
        if num_nonzero_norms >= 2:
            for i, grad_norm_squared in enumerate(gradients_norm_squared):
                if grad_norm_squared != 0:
                    lambda_t[i] = 1 / grad_norm_squared
            lambda_t /= lambda_t.sum()
        else:
            lambda_t = np.full(len(gradients_norm_squared), 1 / len(self.clients_selected))
        self.lamda = lambda_t

    def update_global_model(self):
        """
        Update the global model using personalized updates from clients' gradients.
        The update is performed based on the pFMeMo algorithm.

        Returns:
            None
        """
        self.calculate_lambda()
        global_gradient = np.zeros_like(self.gradients[0])
        for i in range(len(self.clients)):
            global_gradient += self.lamda[i] * self.gradients[i]
        for param_name in global_gradient:
            self.global_model.state_dict()[param_name] -= self.global_learning_rate * global_gradient[param_name]

    def global_train(self):
        """
        Perform global training using the pFMeMo algorithm over multiple rounds.
        Each round involves client local training, gradient aggregation, global model update,
        and evaluation of model performance.

        Returns:
            None
        """
        for _ in range(self.round):
            self.send_global_model()
            for client in self.clients:
                client.local_train()
            self.clients_selected = self.select_clients()
            self.update_global_model()
            self.model_evaluate()
            self.model_per_evaluate()
            self.model_global_test()
        self.save_result()
