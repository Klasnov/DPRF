import torch
from time import time
from algorithm.DPRF import DPRFServer, DPRFClient
from algorithm.pFedMe import pFedMeClient, pFedMeServer
from algorithm.FedMGDA import FedMGDAClient, FedMGDAServer
from algorithm.Ditto import DittoClient, DittoServer
from algorithm.APFL import APFLClient, APFLServer
from algorithm.models import Minst_Model, Cifar10_Model, Emnist_Model


DATASETS = ["MNIST", "cifar10", "emnist"]
ALGORITHMS = ["DPRF", "FedMGDA+", "pFedMe", "Ditto", "APFL"]

def console():
    print("Welcome to Personalized Federated Learning Program!")
    print("Please follow the prompts to select the dataset and algorithm for training.")
    print()

    # Dataset selection
    while True:
        print("Select the dataset.")
        print("For example, enter A or a to indicate that the MNIST dataset is selected.")
        print("A. MNIST")
        print("B. Cifar10")
        print("C. EMNIST")
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
        print("E. APFL")
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
            algorithm = "APFL"
        else:
            print("Invalid algorithm choice.")
            print()
    print()

    return dataset, algorithm

def main(dataset, algorithm, malicious = False):
    
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("************   TRAIN STRAT   ************")
    start_time = time()

    # Model instantiation
    if dataset == "mnist":
        model = Minst_Model()
        CLIENT_NUM = 10
        ROUND_NUM = 500
    elif dataset == "cifar10":
        model = Cifar10_Model()
        CLIENT_NUM = 20
        ROUND_NUM = 50
    else:
        model = Emnist_Model()
        CLIENT_NUM = 25
        ROUND_NUM = 1500

    # Hyperparameters
    SELECT_RATIO = 0.3
    EPOCH = 10
    BATCH_SIZE = 32
    LR_LOCAL = 1e-3
    if dataset == "cifar10":
        LR_LOCAL = 1e-2

    if algorithm == "DPRF":
        LR_GLOBAL = 1
        ALPHA = 7
        K = 10
        server = DPRFServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, ROUND_NUM)
        for i in range(CLIENT_NUM):
            server.add_client(DPRFClient(i, algorithm, dataset, device, model, EPOCH,
                                            BATCH_SIZE, LR_LOCAL, ALPHA, K))
        server.global_train(malicious)
        server.save_result(client_addition=f"_{ALPHA}a_{K}k")
    
    elif algorithm == "pFedMe":
        LR_GLOBAL = 0
        LAMDA = 15
        K = 5
        BETA = 2
        server = pFedMeServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, ROUND_NUM, BETA)
        for i in range(CLIENT_NUM):
            server.add_client(pFedMeClient(i, algorithm, dataset, device, model, EPOCH, BATCH_SIZE, LAMDA, K, LR_LOCAL))
        server.global_train(malicious)
        server.save_result(server_addition=f"_{BETA}b", client_addition=f"_{LAMDA}l_{K}k")

    elif algorithm == "FedMGDA+":
        LR_GLOBAL = 1
        server = FedMGDAServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, ROUND_NUM)
        for i in range(CLIENT_NUM):
            server.add_client(FedMGDAClient(i, algorithm, dataset, device, model, EPOCH, BATCH_SIZE, LR_LOCAL))
        server.global_train(malicious)
        server.save_result()
    
    elif algorithm == "Ditto":
        LR_GLOBAL = 0
        LAMDA = 1
        server = DittoServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, ROUND_NUM)
        for i in range(CLIENT_NUM):
            server.add_client(DittoClient(i, algorithm, dataset, device, model, EPOCH, BATCH_SIZE, LAMDA, LR_LOCAL))
        server.global_train(malicious)
        server.save_result(server_addition=f"_{LAMDA}l")
    
    else:
        LR_GLOBAL = 0
        ALPHA = 0.5
        server = APFLServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, ROUND_NUM)
        for i in range(CLIENT_NUM):
            server.add_client(APFLClient(i, algorithm, dataset, device, model, EPOCH, BATCH_SIZE, LR_LOCAL, ALPHA))
        server.global_train(malicious)
        server.save_result(client_addition=f"_{ALPHA}a")
    
    end_time = time()
    print("\nThe totla training time is %.2f min." % ((end_time - start_time) / 60))
    print("************   TRAIN STRAT   ************")
    print()


if __name__ == "__main__":
    for algorithm in ALGORITHMS:
        main("emnist", algorithm, False)
