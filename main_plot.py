import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ALGORITHMS = ["DPRF", "FedMGDA+", "pFedMe", "Ditto"]
ITEM_TO_LABEL = {0: "Global Model Accuracy", 1: "Personal (Local for FedMGDA+) Model Accuracy", 2: "Training Loss", 3: "Standard Deviation of Training Loss"}

def algorithm_compare_plot(pathes, dataset, item):
    save_dir = f"./img/{dataset}"
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if item == ITEM_TO_LABEL[0]:
        index = 3
    elif item == ITEM_TO_LABEL[1]:
        index = 5
    elif item == ITEM_TO_LABEL[2]:
        index = 1
    
    else:
        index = 4
        plt.figure(figsize=(8, 5))
        colors = plt.cm.rainbow(np.linspace(0.5, 0.3, len(pathes)))
        for i, path in enumerate(pathes):
            df = pd.read_csv(path)
            data = df.iloc[:, index]
            plt.bar(ALGORITHMS[i], data.mean(), color=colors[i])
        plt.xlabel("Algorithms")
        plt.ylabel(item)
        plt.savefig(f"./img/{dataset}/{item} Comparision.png")
        return
    
    plt.figure(figsize=(10, 6))
    for i, path in enumerate(pathes):
        df = pd.read_csv(path)
        if index != 5 or ALGORITHMS[i] != "FedMGDA+":
            data = df.iloc[:, index]
        else:
            data = df.iloc[:, 2]
        plt.plot(data, label=ALGORITHMS[i])
    plt.xlabel("Round")
    plt.ylabel(item)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./img/{dataset}/{item} Comparision.png")
    return

def mnist_compare(dataset):
    dir_path = f"./result/{dataset}"
    file_pathes = []
    for algorithm in ALGORITHMS:
        file_pathes.append(f"{dir_path}/{algorithm}.csv")
    algorithm_compare_plot(file_pathes, dataset, ITEM_TO_LABEL[0])
    algorithm_compare_plot(file_pathes, dataset, ITEM_TO_LABEL[1])
    algorithm_compare_plot(file_pathes, dataset, ITEM_TO_LABEL[2])
    algorithm_compare_plot(file_pathes, dataset, ITEM_TO_LABEL[3])


if __name__ == "__main__":
    mnist_compare("mnist")
