import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ALGORITHMS = ["DPRF", "FedMGDA+", "pFedMe", "Ditto"]

INDEX_TO_LABEL = {
    0: "Training Accuracy",
    1: "Training Loss",
    2: "Local Model Accuracy",
    3: "Global Model Accuracy",
    4: "Standard Deviation of Training Loss",
    5: "Personal (Local for FedMGDA+) Model Accuracy"
}

def algorithm_compare_plot(pathes, dataset, item_index):
    save_dir = f"./img/{dataset}"
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    if item_index == 4:
        plt.figure(figsize=(8, 5))
        colors = plt.cm.rainbow(np.linspace(0.5, 0.3, len(pathes)))
        for i, path in enumerate(pathes):
            df = pd.read_csv(path)
            data = df.iloc[:, item_index]
            plt.bar(ALGORITHMS[i], data.mean(), color=colors[i])
        plt.xlabel("Algorithms")
        plt.ylabel(INDEX_TO_LABEL[item_index])
        plt.savefig(f"./img/{dataset}/{INDEX_TO_LABEL[item_index]} Comparision.png")
        return        
    
    plt.figure(figsize=(10, 6))
    for i, path in enumerate(pathes):
        df = pd.read_csv(path)
        if item_index != 5 or ALGORITHMS[i] != "FedMGDA+":
            data = df.iloc[:, item_index]
        else:
            data = df.iloc[:, 2]
        plt.plot(data, label=ALGORITHMS[i])
    plt.xlabel("Round")
    plt.ylabel(INDEX_TO_LABEL[item_index])
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./img/{dataset}/{INDEX_TO_LABEL[item_index]} Comparision.png")
    return

def parameter_compare(param):
    dir_path = f"./result/param_comp/{param}_comp"
    files = os.listdir(dir_path)
    data = {}
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(dir_path, file)
            df = pd.read_csv(file_path)
            data[file] = df.iloc[:, 3]
    _, ax = plt.subplots()
    if param != "η":
        for file, values in sorted(data.items(), key=lambda x: int(x[0].split('=')[1].split('.')[0])):
            label = f"{param} = {file.split('=')[1].split('.')[0]}"
            ax.plot(values, label=label)
        axins = ax.inset_axes([0.55, 0.3, 0.4, 0.4])
        axins.set_xticklabels([])
        axins.tick_params(axis='both', which='both', length=0)
        for file, values in sorted(data.items(), key=lambda x: int(x[0].split('=')[1].split('.')[0])):
            acc_values = values[-50:]
            axins.plot(acc_values)
        ax.indicate_inset_zoom(axins)
    else:
        for file, values in sorted(data.items(), key=lambda x: float(x[0].split('=')[1].split('.csv')[0])):
            label = f"{param} = {file.split('=')[1].split('.csv')[0]}"
            ax.plot(values, label=label)
    plt.xlabel("Round")
    plt.ylabel("Global Model Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    img_dir = f"./img/param_comp"
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, f"{param}_comp.png")
    plt.savefig(img_path)
    plt.show()
    

def algorithm_compare(dataset):
    dir_path = f"./result/{dataset}"
    file_pathes = []
    for algorithm in ALGORITHMS:
        file_pathes.append(f"{dir_path}/{algorithm}.csv")
    for index in INDEX_TO_LABEL.keys():
        algorithm_compare_plot(file_pathes, dataset, index)

if __name__ == "__main__":
    # algorithm_compare("mnist")
    # algorithm_compare("mnist_malicious")
    # parameter_compare("a")
    # parameter_compare("k")
    parameter_compare("a")
