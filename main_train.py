import torch
from time import time
from algorithm.pFMeMo import pFMeMoServer, pFMeMoClient
from algorithm.pFedMe import pFedMeClient, pFedMeServer
from utils.util_models import Minst_Model

def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = "mnist"
    algorithm = "pFMeMo"

    # print("Welcome to Personalized Federated Learning Program!")
    # print("Please follow the prompts to select the dataset and algorithm for training.")
    # print()

    # TODO: 完成用户选择模型本地导入的情况

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

    # Hyperparameters
    lr_global = 1e-3
    user_selection_ratio = 0.3
    round = 800
    local_epochs = 20
    local_batch_size = 64

    if algorithm == "pFMeMo":
        alpha = 275
        k = 9
        lr_local = 1e-3
        server = pFMeMoServer(algorithm, dataset, device, model, lr_global, user_selection_ratio, round)
        if dataset == "mnist":
            num_clients = 10
            for i in range(num_clients):
                client = pFMeMoClient(i, dataset, device, model, local_epochs, local_batch_size, alpha, k, lr_local)
                server.add_client(client)
        server.global_train()
        server.save_result(f"{local_epochs}epc_{local_batch_size}bch_{alpha}a_{k}k_{lr_local}ll")
        server.save_model(f"{local_epochs}epc_{local_batch_size}bch_{alpha}a_{k}k_{lr_local}ll")
    
    elif algorithm == "pFedMe":
        server = pFedMeServer(algorithm)


if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    print(f"\nThe totla training time is {end_time - start_time}s.")
