import torch
from algorithm.pFMeMo import pFMeMoServer, pFMeMoClient
from model.model import Minst_Model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = "mnist"
    algorithm = "pFMeMo"

    # print("Welcome to Personalized Federated Learning Program!")
    # print("Please follow the prompts to select the dataset and algorithm for training.")
    # print()

    # # Dataset selection
    # while True:
    #     print("Select the dataset.")
    #     print("For example, enter A or a to indicate that the MNIST dataset is selected.")
    #     print("A. MNIST")
    #     print("B. Cifar10")
    #     dataset_choice = input("Your choice: ").lower()
    #     if dataset_choice == 'A' or dataset_choice == 'a':
    #         dataset = "mnist"
    #         break
    #     elif dataset_choice == 'b':
    #         dataset = "cifar10"
    #         break
    #     else:
    #         print("Invalid dataset choice.")
    #         print()
    # print()

    # Model instantiation
    if dataset == "mnist":
        model = Minst_Model()

    # # Algorithm selection
    # while True:
    #     print("Select the algorithm for training.")
    #     print("For example, enter A or a to indicate that the pFMeMo algorithm is selected.")
    #     print("A. pFMeMo")
    #     print("B. FedMGDA+")
    #     print("C. FedMe")
    #     print("D. per-FedAvg")
    #     algorithm_choice = input("Your choice: ").lower()

    #     if algorithm_choice == 'A' or algorithm_choice == 'a':
    #         algorithm = "pFMeMo"
    #         break
    #     elif algorithm_choice == 'b':
    #         algorithm = "FedMGDA+"
    #     elif algorithm_choice == 'c':
    #         algorithm = "FedMe"
    #     elif algorithm_choice == 'd':
    #         algorithm = "per-FedAvg"
    #     else:
    #         print("Invalid algorithm choice.")
    #         print()
    # print()

    # Hyperparameters (you need to tune these values)
    lr_g = 1e-2
    user_selection_ratio = 0.3
    round = 100
    local_epochs = 5
    local_batch_size = 32

    if algorithm == "pFMeMo":
        server = pFMeMoServer(algorithm, dataset, device, model, lr_g, user_selection_ratio, round)
        
        alpha = 20
        delta = 7.5
        lr_p = 1e-2
        lr_l = 1e-2

        if dataset == "mnist":
            num_clients = 10
            for i in range(num_clients):
                client = pFMeMoClient(i, dataset, device, model, local_epochs, local_batch_size, alpha, delta, lr_p, lr_l)
                server.add_client(client)
        server.global_train(f"{alpha}a_{delta}d_{lr_p}lp_{lr_l}ll")

if __name__ == "__main__":
    main()