import torch
from time import time
from algorithm.DPRF import DPRFServer, DPRFClient
from algorithm.pFedMe import pFedMeClient, pFedMeServer
from utils.util_models import Minst_Model

def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print("Welcome to Personalized Federated Learning Program!")
    # print("Please follow the prompts to select the dataset and algorithm for training.")
    # print()

    # Dataset selection
    # while True:
    #     print("Select the dataset.")
    #     print("For example, enter A or a to indicate that the MNIST dataset is selected.")
    #     print("A. MNIST")
    #     print("B. Cifar10")
    #     dataset_choice = input("Your choice: ").lower()
    #     if dataset_choice == 'A' or dataset_choice == 'a':
    #         dataset = "mnist"
    #         break
    #     # elif dataset_choice == 'B' or dataset_choice == 'b':
    #     #     dataset = "cifar10"
    #     #     break
    #     # elif dataset_choice == 'C' or dataset_choice == 'c':
    #     #     dataset = "cifar100"
    #     #     break
    #     else:
    #         print("Invalid dataset choice.")
    #         print()
    # print()
    dataset = "mnist"

    # Model instantiation
    if dataset == "mnist":
        model = Minst_Model()

    # Algorithm selection
    # while True:
    #     print("Select the algorithm for training.")
    #     print("For example, enter A or a to indicate that the pFMeMo algorithm is selected.")
    #     print("A. DPRF")
    #     print("B. pFedMe")
    #     print("C. FedMGDA+")
    #     print("D. per-FedAvg")
    #     algorithm_choice = input("Your choice: ").lower()

    #     if algorithm_choice == 'A' or algorithm_choice == 'a':
    #         algorithm = "DPRF"
    #         break
    #     elif algorithm_choice == 'B' or algorithm_choice == 'b':
    #         algorithm = "pFedMe"
    #         break
    #     # elif algorithm_choice == 'c':
    #     #     algorithm = "FedMGDA+"
    #     # elif algorithm_choice == 'd':
    #     #     algorithm = "per-FedAvg"
    #     else:
    #         print("Invalid algorithm choice.")
    #         print()
    # print()
    algorithm = "DPRF"

    # Hyperparameters
    SELECT_RATIO = 0.3
    ROUND_NUM = 50
    LOCAL_EPOCH = 25
    LOCAL_BATCH_SIZE = 64

    if algorithm == "DPRF":
        LR_GLOBAL = 1
        LR_LOCAL = 1e-3
        ALPHA = 25
        K = 15
        server = DPRFServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, ROUND_NUM)
        if dataset == "mnist":
            CLIENT_NUM = 10
            for i in range(CLIENT_NUM):
                server.add_client(DPRFClient(i, algorithm, dataset, device, model, LOCAL_EPOCH,
                                             LOCAL_BATCH_SIZE, LR_LOCAL, ALPHA, K))
        # server.load_model(client_addition=f"_{ALPHA}a_{K}k")
        server.global_train()
        server.save_result(client_addition=f"_{ALPHA}a_{K}k")
        # server.save_model(client_addition=f"_{ALPHA}a_{K}k")
    
    elif algorithm == "pFedMe":
        LR_GLOBAL = 1e-3
        LR_LOCAL = 1e-3
        LAMDA = 15
        K = 5
        BETA = 2
        server = pFedMeServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, ROUND_NUM, BETA)
        if dataset == "mnist":
            CLIENT_NUM = 10
            for i in range(CLIENT_NUM):
                server.add_client(pFedMeClient(i, algorithm, dataset, device, model, LOCAL_EPOCH, LOCAL_BATCH_SIZE,
                                               LAMDA, K, LR_LOCAL))
        server.global_train()
        server.save_result(server_addition=f"_{BETA}b", client_addition=f"_{LAMDA}l_{K}k")


if __name__ == "__main__":
    start_time = time()
    main()
    end_time = time()
    print(f"\nThe totla training time is {end_time - start_time}s.")
    print()
