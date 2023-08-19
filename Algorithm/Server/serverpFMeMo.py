import torch
from serverbase import Server


def update_weights(norm_gradients):
    lambda_star = torch.zeros_like(norm_gradients)
    non_zero_indices = norm_gradients > 0
    num_non_zero = torch.sum(non_zero_indices)
    if num_non_zero > 0:
        lambda_star[non_zero_indices] = 1 / (norm_gradients[non_zero_indices] ** 2)
        lambda_star /= torch.sum(lambda_star)
    return lambda_star


class PFMeMoServer(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, local_epochs, learning_rate, user_num,
                 global_learning_rate):
        super().__init__(device, dataset, algorithm, model, batch_size, local_epochs, learning_rate, user_num)
        self.global_learning_rate = global_learning_rate

    def normalize_gradients(self):
        norm_gradients = []
        for user in self.selected_users:
            norm_gradient = torch.norm(user.gradient_update_direction, p=2)
            norm_gradients.append(norm_gradient)
        norm_gradients = torch.stack(norm_gradients)
        return norm_gradients

    def global_update(self, lambda_star):
        aggregated_direction = torch.zeros_like(self.model.parameters())
        for user, lambda_i in zip(self.selected_users, lambda_star):
            aggregated_direction += lambda_i * user.gradient_update_direction
        for param, direction in zip(self.model.parameters(), aggregated_direction):
            param.data -= self.global_learning_rate * direction

    def federated_learning(self, num_rounds):
        self.model.to(self.device)
        if self.model_exists():
            self.load_model()
        for t in range(num_rounds):
            print("Round {}/{}".format(t + 1, num_rounds))
            self.selected_users = self.select_users(self.user_num)
            self.send_parameters()
            for user in self.selected_users:
                user.gradient_update_direction = user.local_training()
            norm_gradients = self.normalize_gradients()
            lambda_star = update_weights(norm_gradients)
            self.global_update(lambda_star)
            self.evaluate()
        self.save_model()
        self.save_results()
