import torch
from time import time
from algorithm.DPRF import DPRFServer, DPRFClient
from algorithm.pFedMe import pFedMeClient, pFedMeServer
from algorithm.FedMGDA import FedMGDAClient, FedMGDAServer
from algorithm.Ditto import DittoClient, DittoServer
from algorithm.APFL import APFLClient, APFLServer
from algorithm.lpProj import lpProjClient, lpProjServer
from algorithm.PSBGD import PSBGDClient, PSBGDServer
from algorithm.models import MnistModel, EmnistModel, Cifar10Model


DATASETS = ["mnist", "emnist", "cifar10"]
ALGORITHMS = ["lp-Proj", "Ditto", "PSBGD", "FedMGDA+", "DPRF", "APFL"]
MALICIOUS = {0: "normal (no attack)", 1: "amplifying attack", 2: "Gaussian noise attack", 3: "flipping attack"}


def console():
    print("Welcome to DPRF Personalized Federated Learning Program!")
    print("Please follow the prompts to select the dataset and algorithm for training.")
    print()

    # Dataset selection
    while True:
        print("Select the dataset.")
        print("For example, enter A or a to indicate that the MNIST dataset is selected.")
        print("A. MNIST")
        print("B. EMNIST")
        print("C. CIFAR-10")
        dataset_choice = input("Your choice: ").lower()
        if dataset_choice == 'a':
            dataset = "mnist"
            break
        elif dataset_choice == 'b':
            dataset = "emnist"
            break
        elif dataset_choice == 'c':
            dataset = "cifar10"
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
        print("B. lpProj")
        print("C. FedMGDA+")
        print("D. Ditto")
        print("E. APFL")
        print("F. PSBGD")
        print("G. pFedMe")
        algorithm_choice = input("Your choice: ").lower()

        if algorithm_choice == 'a':
            algorithm = "DPRF"
            break
        elif algorithm_choice == 'b':
            algorithm = "lp-Proj"
            break
        elif algorithm_choice == 'c':
            algorithm = "FedMGDA+"
            break
        elif algorithm_choice == 'd':
            algorithm = "Ditto"
            break
        elif algorithm_choice == 'e':
            algorithm = "APFL"
            break
        elif algorithm_choice == 'f':
            algorithm = "PSBGD"
            break
        elif algorithm_choice == 'g':
            algorithm = "pFedMe"
            break
        else:
            print("Invalid algorithm choice.")
            print()
    print()

    # Attack selection
    while True:
        print("Select the attack type.")
        print("For example, enter A or a to indicate that the attack type is selected.")
        print("A. Normal (no attack)")
        print("B. Aplifying attack")
        print("C. Gaussian noise attack")
        print("D. Flipping attack")
        attack_choice = input("Your choice: ").lower()

        if attack_choice == 'a':
            malicious_type = 0
            break
        elif attack_choice == 'b':
            malicious_type = 1
            break
        elif attack_choice == 'c':
            malicious_type = 2
            break
        elif attack_choice == 'd':
            malicious_type = 3
            break
        else:
            print("Invalid attack choice.")
            print()
    print()

    return dataset, algorithm, malicious_type

def main(dataset, algorithm, malicious_type = 0, indicated_round = 0, load_save_model = False, alpha=9, k=10):
    
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("************   TRAIN STRAT   ************")
    start_time = time()

    if dataset == "mnist":
        model = MnistModel()
        CLIENT_NUM = 10
        ROUND_NUM = 750
    elif dataset == "emnist":
        model = EmnistModel()
        CLIENT_NUM = 20
        ROUND_NUM = 1500
    else:
        model = Cifar10Model()
        CLIENT_NUM = 15
        ROUND_NUM = 2000

    if indicated_round != 0:
        ROUND_NUM = indicated_round

    SELECT_RATIO = 0.3
    EPOCH = 10
    BATCH_SIZE = 32
    LR_LOCAL = 1e-3
    MALICIOUS_RATIO = 0.2
    APLIFYING_FACTOR = 1e2

    server_addition = ""
    client_addition = ""

    if algorithm == "DPRF":
        LR_GLOBAL = 2
        if dataset == "mnist":
            LR_GLOBAL = 1
        ALPHA = alpha
        K = k
        server = DPRFServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO)
        for i in range(CLIENT_NUM):
            server.add_client(DPRFClient(i, algorithm, dataset, device, model, EPOCH, BATCH_SIZE, LR_LOCAL, ALPHA, K))
        client_addition=f"_{ALPHA}a_{K}k"
    
    elif algorithm == "pFedMe":
        LR_GLOBAL = 0
        LAMDA = 15
        K = 5
        BETA = 2
        server = pFedMeServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, BETA)
        for i in range(CLIENT_NUM):
            server.add_client(pFedMeClient(i, algorithm, dataset, device, model, EPOCH, BATCH_SIZE, LAMDA, K, LR_LOCAL))
        server_addition=f"_{BETA}b"
        client_addition=f"_{LAMDA}l_{K}k"

    elif algorithm == "FedMGDA+":
        LR_GLOBAL = 2
        if dataset == "mnist":
            LR_GLOBAL = 1
        server = FedMGDAServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO)
        for i in range(CLIENT_NUM):
            server.add_client(FedMGDAClient(i, algorithm, dataset, device, model, EPOCH, BATCH_SIZE, LR_LOCAL))
    
    elif algorithm == "Ditto":
        LR_GLOBAL = 0
        LAMDA = 1
        server = DittoServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO)
        for i in range(CLIENT_NUM):
            server.add_client(DittoClient(i, algorithm, dataset, device, model, EPOCH, BATCH_SIZE, LAMDA, LR_LOCAL))
        server_addition=f"_{LAMDA}l"
    
    elif algorithm == "APFL":
        LR_GLOBAL = 0
        ALPHA = 0.5
        server = APFLServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO)
        for i in range(CLIENT_NUM):
            server.add_client(APFLClient(i, algorithm, dataset, device, model, EPOCH, BATCH_SIZE, LR_LOCAL, ALPHA))
        client_addition=f"_{ALPHA}a"
    
    elif algorithm == "lp-Proj":
        LR_GLOBAL = 0
        LAMDA = 15
        K = 5
        BETA = 2
        server = lpProjServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, BETA)
        for i in range(CLIENT_NUM):
            server.add_client(lpProjClient(i, algorithm, dataset, device, model, EPOCH, BATCH_SIZE, LAMDA, K, LR_LOCAL))
        server_addition=f"_{BETA}b"
        client_addition=f"_{LAMDA}l_{K}k"
    
    else:
        LR_GLOBAL = 1e-2
        BETA = 1e-2
        MU = 2e-2
        T = 1e-2
        server = PSBGDServer(algorithm, dataset, device, model, LR_GLOBAL, SELECT_RATIO, BETA, MU, T)
        for i in range(CLIENT_NUM):
            server.add_client(PSBGDClient(i, algorithm, dataset, device, model, EPOCH, BATCH_SIZE, BETA, LR_LOCAL))
        server_addition=f"_{BETA}b_{MU}m_{T}t"
        client_addition=f"_{BETA}b"
    
    if load_save_model:
        server.load_model(server_addition=server_addition, client_addition=client_addition)
    server.global_train(ROUND_NUM, malicious_type, MALICIOUS_RATIO, APLIFYING_FACTOR, load_save_model)
    server.save_result(server_addition=server_addition, client_addition=client_addition, use_addition=False)
    if load_save_model:
        server.save_model(server_addition=server_addition, client_addition=client_addition)
    
    end_time = time()
    print("\nThe totla training time is %.2f min." % ((end_time - start_time) / 60))
    print("************   TRAIN END   ************")
    print()


if __name__ == "__main__":
    main(dataset="emnist", algorithm="pFedMe", indicated_round=10)
