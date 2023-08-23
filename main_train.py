from algorithm.pFMeMo import pFMeMoServer, pFMeMoClient
from model.model import Minst_Model

def main():
    print("Welcome to Personalized Federated Learning Program!")
    print("Please follow the prompts to select the dataset and algorithm for training.")
    print()

    # Dataset selection
    while True:
        print("Select the dataset.")
        print("For example, enter A or a to indicate that the MINST dataset is selected.")
        print("A. MNIST")
        print("B. Cifar100")
        dataset_choice = input("Your choice: ").lower()
        if dataset_choice == 'A' or dataset_choice == 'a':
            dataset = "minst"
            break
        # elif dataset_choice == 'b':
        #     dataset = "Cifar100"
        #     break
        else:
            print("Invalid dataset choice.")
            print()
    print()

    # Model instantiation
    if dataset == "minst":
        model = Minst_Model()

    # Algorithm selection
    while True:
        print("Select the algorithm for training.")
        print("For example, enter A or a to indicate that the pFMeMo algorithm is selected.")
        print("A. pFMeMo")
        print("B. FedMGDA+")
        print("C. FedMe")
        print("D. per-FedAvg")
        algorithm_choice = input("Your choice: ").lower()

        if algorithm_choice == 'A' or algorithm_choice == 'a':
            algorithm = "pFMeMo"
            break
        # elif algorithm_choice == 'b':
        #     algorithm = "FedMGDA+"
        # elif algorithm_choice == 'c':
        #     algorithm = "FedMe"
        # elif algorithm_choice == 'd':
        #     algorithm = "per-FedAvg"
        else:
            print("Invalid algorithm choice.")
            print()
    print()

    # Hyperparameters (you need to tune these values)
    lr_g = 0.01
    user_selection_ratio = 0.3
    round = 10

    if algorithm == "pFMeMo":
        server = pFMeMoServer(algorithm, dataset, model, lr_g, user_selection_ratio, round)

        if dataset == "minst":
            num_clients = 10
            for i in range(num_clients):
                client = pFMeMoClient(client_id=i, dataset=dataset, model=model, local_epochs=5, local_batch_size=32,
                                      alpha=0.01, delta=0.1, lr_p=0.01, lr_l=0.001)
                server.add_client(client)
        server.global_train()

if __name__ == "__main__":
    main()
