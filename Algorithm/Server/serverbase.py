import torch
import os
import numpy as np
import h5py
import copy


class Server:
    def __init__(self, device, dataset, algorithm, model, batch_size, local_epochs, learning_rate, user_num):
        self.device = device
        self.dataset = dataset
        self.algorithm = algorithm
        self.model = copy.deepcopy(model)
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.user_num = user_num
        self.users = []
        self.selected_users = []
        self.rs_train_acc = []
        self.rs_train_loss = []
        self.rs_glob_acc = []

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = None
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_sample_num
        for user in self.selected_users:
            self.add_parameters(user, user.train_sample_num / total_train)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def select_users(self, num_users):
        if num_users == len(self.users):
            print("All users are selected")
            return self.users
        num_users = min(num_users, len(self.users))
        return np.random.choice(self.users, num_users, replace=False)

    def save_results(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.user_num) + "u" + "_" + str(self.batch_size) + "b"
        alg = alg + "_" + str(self.local_epochs)
        if len(self.rs_glob_acc) != 0 & len(self.rs_train_acc) & len(self.rs_train_loss):
            with h5py.File("./results/" + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.close()

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
        ids = [c.id for c in self.users]
        return ids, num_samples, tot_correct, losses

    def test(self):
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]
        return ids, num_samples, tot_correct

    def evaluate(self):
        stats = self.test()
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        print("Average Global Accuracy: ", glob_acc)
        print("Average Global Training Accuracy: ", train_acc)
        print("Average Global Training Loss: ", train_loss)
