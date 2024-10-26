# Dynamic Personalized Federated Learning Against Byzantine Attacks

This repository contains the implementation of the research paper "Dynamic Personalized Federated Learning Against Byzantine Attacks". The project showcases different personalized federated learning algorithms and demonstrates their robustness against various types of Byzantine attacks. The implementation is designed to be easy to use and modify, accommodating a variety of datasets and personalization algorithms.

## Features

- **Personalized Federated Learning Algorithms**: Implements a variety of algorithms such as DPRF, pFedMe, FedMGDA+, Ditto, APFL, lp-Proj, and PSBGD.
  
- **Dataset Support**: Provides support for multiple datasets including MNIST, EMNIST, and CIFAR-10.
  
- **Byzantine Attack Handling**: Models are tested against different types of attacks such as amplifying attacks, Gaussian noise attacks, and flipping attacks.
  
- **Simulation Environment**: A console-based interface to select datasets, algorithms, and attack types for easy simulation.

## Getting Started

### Installation

To clone and run this repository, execute the following commands in your terminal:

```bash
git clone https://github.com/Klasnov/DPRF.git
cd DPRF
pip install -r requirements.txt
```

### Usage

To start the simulation, run the `main.py` script. You can follow the console prompts to select the desired dataset, algorithm, and attack type.

```bash
python main.py
```

The code will initialize the federated learning process and display the progress and results at each round of simulation.

## Code Structure

- **`main.py`**: The main entry point of the program, handling user input and initializing the federated learning setup.

- **`base.py`**: Contains the base classes for Server and Client used in federated learning, providing essential methods like model evaluation and dataset management.

- **`algorithm/`**: Contains different algorithm implementations for both client and server levels.

- **`data/`**: Directory to place your datasets, organized by dataset name and client data.

- **`model/`**: Directory used for saving and loading model states.

- **`result/`**: Directory where results of the simulations are stored.

## Data Preparation

Ensure your datasets are correctly formatted and saved in the `data/` directory. The structure should follow a pattern that matches the code expectations:

```
data/
│── mnist/
│    ├── data/
│         ├── client0/
│              ├── x.pt
│              └── y.pt
│         └── ...
└── emnist/ (same structure as above)
```

The data loading logic expects `.pt files` (PyTorch tensor files) containing input features and labels.

## Results

The results of each simulation, including accuracy and loss metrics, are saved in the `result/` directory. The results filenames are automatically generated based on selected parameters like the dataset name and chosen algorithm.

## License

This project is licensed under the MIT License.

