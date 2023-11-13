import pandas as pd
import matplotlib.pyplot as plt


ALGORITHMS = ["DPRF", "FedMGDA+", "pFedMe", "Ditto"]

def algorithm_compare_plot(pathes, index, dataset, item, algorithms):
    plt.figure(figsize=(10, 6))
    for i, path in enumerate(pathes):
        df = pd.read_csv(path)
        data = df.iloc[:, index - 1]
        plt.plot(data, label=algorithms[i])
    plt.xlabel("Round")
    plt.ylabel(item)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./img/{dataset}_comparison ({item}).png")
    plt.show()

def mnist_compare():
    dir_path = "./result/mnist"
    file_path = []
    for algorithm in ALGORITHMS:
        file_path.append(f"{dir_path}/{algorithm}.csv")
    algorithm_compare_plot(file_path, 4, "mnist", "Global Accuracy", ALGORITHMS)


if __name__ == "__main__":
    mnist_compare()
