import torch
import copy
from userbase import User


class PFMeMoUser(User):
    def __init__(self, device, user_id, dataset, model, train_data, test_data, batch_size=0, local_epochs=0,
                 learning_rate=0, alpha=0, eta_p=0, eta_l=0, delta=0):
        super().__init__(device, user_id, dataset, model, train_data, test_data, batch_size,
                         local_epochs, learning_rate)
        self.alpha = alpha
        self.eta_p = eta_p
        self.eta_l = eta_l
        self.delta = delta

    def local_training(self):
        theta_i = None
        gradient_update_direction = None
        self.model.train()
        local_model = copy.deepcopy(self.model)
        w_t = self.model.parameters()
        for r in range(self.local_epochs):
            local_optimizer = torch.optim.SGD(local_model.parameters(), lr=self.eta_l)
            for _ in range(self.train_sample_num // self.batch_size):
                local_optimizer.zero_grad()
                inputs, labels = self.get_next_train_batch()
                theta_i = copy.deepcopy(local_model)
                theta_i_optimizer = torch.optim.SGD(theta_i.parameters(), lr=self.eta_p)
                while True:
                    local_model.zero_grad()
                    output = local_model(inputs)
                    loss = self.loss(output, labels)
                    loss.backward()
                    theta_i_grad = self.get_grads()
                    tilde_h_i = self.compute_tilde_h_i(theta_i.parameters(), theta_i_grad)
                    tilde_h_i.backward()
                    theta_i_optimizer.step()
                    if torch.norm(tilde_h_i) ** 2 <= self.delta:
                        break
                local_model = copy.deepcopy(theta_i)
            gradient_update_direction = w_t - theta_i.parameters()
        return gradient_update_direction

    def compute_tilde_h_i(self, theta_i_params, theta_i_grad):
        param = None
        i = None
        tilde_f_i_grad = torch.zeros_like(theta_i_grad)
        for i, (param, grad) in enumerate(zip(theta_i_params, theta_i_grad)):
            tilde_f_i_grad[i] = torch.sum(grad) / self.batch_size
        tilde_h_i = tilde_f_i_grad + self.alpha * (param - self.model.parameters()[i])
        return tilde_h_i
