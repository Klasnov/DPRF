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
    SELECT_RATIO = 0.3
    ROUND_NUM = 2
    LOCAL_EPOCH = 20
    LOCAL_BATCH_SIZE = 64
    LR_GLOBAL = 1e-3
    LR_LOCAL = 1e-3

    if algorithm == "pFMeMo":
        ALPHA = 275
        K = 9
        server = pFMeMoServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, ROUND_NUM)
        if dataset == "mnist":
            CLIENT_NUM = 10
            for i in range(CLIENT_NUM):
                server.add_client(pFMeMoClient(i, algorithm, dataset, device, model, LOCAL_EPOCH, LOCAL_BATCH_SIZE, LR_LOCAL, ALPHA, K))
        # server.load_model(client_addition=f"_{ALPHA}a_{K}k")
        server.global_train()
        server.save_result(client_addition=f"_{ALPHA}a_{K}k")
        # server.save_model(client_addition=f"_{ALPHA}a_{K}k")
    
    elif algorithm == "pFedMe":
        pass


if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    print(f"\nThe totla training time is {end_time - start_time}s.")
    print()
