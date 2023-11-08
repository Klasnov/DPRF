import torch
from time import time
from algorithm.DPRF import DPRFServer, DPRFClient
from algorithm.pFedMe import pFedMeClient, pFedMeServer
from algorithm.FedMGDA import FedMGDAClient, FedMGDAServer
from algorithm.Ditto import DittoClient, DittoServer
from algorithm.FedFomo import FedFomoClient, FedFomoServer
from utils.util_models import Minst_Model, Cifar10_Model, Emnist_Model


def console() -> tuple[str, str]:
    print("Welcome to Personalized Federated Learning Program!")
    print("Please follow the prompts to select the dataset and algorithm for training.")
    print()

    # Dataset selection
    while True:
        print("Select the dataset.")
        print("For example, enter A or a to indicate that the MNIST dataset is selected.")
        print("A. MNIST")
        print("B. Cifar10")
        dataset_choice = input("Your choice: ").lower()
        if dataset_choice == 'a':
            dataset = "mnist"
            break
        elif dataset_choice == 'b':
            dataset = "cifar10"
            break
        elif dataset_choice == 'c':
            dataset = "emnist"
            break
        else:
            print("Invalid dataset choice.")
            print()
    print()

    # Algorithm selection
    while True:
        print("Select the algorithm for training.")
        print("For example, enter A or a to indicate that the pFMeMo algorithm is selected.")
        print("A. DPRF")
        print("B. pFedMe")
        print("C. FedMGDA+")
        print("D. Ditto")
        print("E. FedFomo")
        algorithm_choice = input("Your choice: ").lower()

        if algorithm_choice == 'a':
            algorithm = "DPRF"
            break
        elif algorithm_choice == 'b':
            algorithm = "pFedMe"
            break
        elif algorithm_choice == 'c':
            algorithm = "FedMGDA+"
            break
        elif algorithm_choice == 'd':
            algorithm = "Ditto"
            break
        elif algorithm_choice == 'e':
            algorithm = "FedFomo"
            break
        else:
            print("Invalid algorithm choice.")
            print()
    print()

    return dataset, algorithm

def main(dataset: str, algorithm: str) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time()

    # Model instantiation
    if dataset == "mnist":
        model = Minst_Model()
        CLIENT_NUM = 10
    elif dataset == "cifar10":
        model = Cifar10_Model()
        CLIENT_NUM = 20
    else:
        model = Emnist_Model()
        CLIENT_NUM = 50

    # Hyperparameters
    SELECT_RATIO = 0.3
    ROUND_NUM = 100
    LOCAL_EPOCH = 5
    LOCAL_BATCH_SIZE = 64

    if algorithm == "DPRF":
        LR_GLOBAL = 1
        LR_LOCAL = 1e-3
        ALPHA = 10
        K = 10
        server = DPRFServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, ROUND_NUM)
        for i in range(CLIENT_NUM):
            server.add_client(DPRFClient(i, algorithm, dataset, device, model, LOCAL_EPOCH,
                                            LOCAL_BATCH_SIZE, LR_LOCAL, ALPHA, K))
        server.global_train()
        server.save_result(client_addition=f"_{ALPHA}a_{K}k")
    
    elif algorithm == "pFedMe":
        LR_GLOBAL = 1e-3
        LR_LOCAL = 1e-3
        LAMDA = 15
        K = 5
        BETA = 2
        server = pFedMeServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, ROUND_NUM, BETA)
        for i in range(CLIENT_NUM):
            server.add_client(pFedMeClient(i, algorithm, dataset, device, model, LOCAL_EPOCH, LOCAL_BATCH_SIZE,
                                            LAMDA, K, LR_LOCAL))
        server.global_train()
        server.save_result(server_addition=f"_{BETA}b", client_addition=f"_{LAMDA}l_{K}k")

    elif algorithm == "FedMGDA+":
        LR_GLOBAL = 1
        LR_LOCAL = 1e-3
        server = FedMGDAServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, ROUND_NUM)
        for i in range(CLIENT_NUM):
            server.add_client(FedMGDAClient(i, algorithm, dataset, device, model, LOCAL_EPOCH, LOCAL_BATCH_SIZE, LR_LOCAL))
        server.global_train()
        server.save_result()
    
    elif algorithm == "Ditto":
        LR_GLOBAL = 0
        LR_LOCAL = 1e-3
        LAMDA = 1
        server = DittoServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, ROUND_NUM)
        for i in range(CLIENT_NUM):
            server.add_client(DittoClient(i, algorithm, dataset, device, model, LOCAL_EPOCH, LOCAL_BATCH_SIZE, LAMDA, LR_LOCAL))
        server.global_train()
        server.save_result()
    
    else:
        SELECT_RATIO = 0.5
        LR_GLOBAL = 0
        LR_LOCAL = 1e-3
        server = FedFomoServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, ROUND_NUM, CLIENT_NUM)
        for i in range(CLIENT_NUM):
            server.add_client(FedFomoClient(i, algorithm, dataset, device, model, LOCAL_EPOCH, LOCAL_BATCH_SIZE, LR_LOCAL, CLIENT_NUM))
        server.global_train()
        server.save_result()
    
    end_time = time()
    print(f"\nThe totla training time is {end_time - start_time}s.")
    print()


if __name__ == "__main__":
    main("mnist", "DPRF")
