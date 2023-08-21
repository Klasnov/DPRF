import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseClient, BaseServer


class pFMeMoClient(BaseClient):
    def __init__(self, client_id, dataset, model: nn.Module, local_epochs, local_batch_size, alpha, delta, lr_p, lr_l):
        super().__init__(client_id, dataset, model, local_epochs, local_batch_size)
        self.alpha = alpha  # Regularization parameter for personalized updates
        self.delta = delta  # Maximum squared norm of personalized update
        self.lr_p = lr_p    # Learning rate for projected gradient descent
        self.lr_l = lr_l    # Learning rate for local model updates

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
        local_gradient = {}
        for name, param in self.local_model.named_parameters():
            local_gradient[name] = param.data - w_t_i[name]
        return local_gradient

class pFMeMoServer(BaseServer):
    pass
