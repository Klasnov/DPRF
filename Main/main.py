import os
import torch
from Algorithm.User.userpFMeMo import PFMeMoUser
from Algorithm.Server.serverpFMeMo import PFMeMoServer
from Model.model import MNISTConvNet
from Data.MINST.minst import main as prepare_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare the MNIST dataset
    prepare_data()
    # Create the global model
    global_model = MNISTConvNet()
    # Initialize users and server
    user_num = 10
    users = []
    for i in range(user_num):
        user = PFMeMoUser(device, str(i), "MNIST", global_model, torch.load(f'../Data/MINST/user_{i}_train.pt'),
                          torch.load('../Data/MINST/x_test.pt'), batch_size=64, local_epochs=5,
                          learning_rate=0.01, alpha=0.1, eta_p=0.01, eta_l=0.01, delta=0.01)
        users.append(user)
    server = PFMeMoServer(device, "MNIST", "pFMeMo", global_model, batch_size=64, local_epochs=5,
                          learning_rate=0.01, user_num=user_num, global_learning_rate=0.1)
    # Run federated learning
    num_rounds = 10
    server.federated_learning(num_rounds)

if __name__ == '__main__':
    main()
