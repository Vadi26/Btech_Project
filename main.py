import pickle
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

import flwr as fl

from dataset import prepare_dataset, prepare_credit_card_dataset, prepare_iot_dataset
from client import generate_client_fn, generate_credit_card_client_fn, generate_iot_client
from server import get_on_fit_config, get_evaluate_fn

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    ## 1. Parse config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    # print(cfg)

    ## 2. Prepare dataset
    trainloaders, validationloaders, testloader = prepare_iot_dataset(cfg.data_path,
                                                                    cfg.num_clients,
                                                                    cfg.batch_size
                                                                    )
    
    visualize_client_attack_distribution(trainloaders)

    # print(len(trainloaders), len(trainloaders[0].dataset), len(trainloaders[1].dataset), len(trainloaders[2].dataset), len(trainloaders[8].dataset), len(trainloaders[9].dataset))
    for i in range(len(trainloaders)):
        print("Trainloaders ==> ", len(trainloaders[i].dataset), len(trainloaders[i]))
    ## 3. Define your clients
    # This function returns a function which is able to instantiate a client of a particular id
    client_fn = generate_iot_client(trainloaders, validationloaders, cfg.model)

    ## 4. Define your strategy
    # # strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001,
    # #                                      min_fit_clients=cfg.num_clients_per_round_fit,   # How many clients are gonna be used per round for training
    # #                                      fraction_evaluate=0.00001,
    # #                                      min_evaluate_clients=cfg.num_clients_per_round_eval,  # How many clients are gonna be used for evaluation
    # #                                      min_available_clients=cfg.num_clients,
    # #                                      on_fit_config_fn=get_on_fit_config(cfg.config_fit),   
    # #                                      evaluate_fn=get_evaluate_fn(cfg.num_classes,
    # #                                                                  testloader)
    # #                                      )
    
    strategy = instantiate(cfg.strategy, evaluate_fn=get_evaluate_fn(cfg.model,testloader))

    # ## 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,      # To spawn a client
        num_clients=cfg.num_clients,    # To know the number of clients
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),     
        strategy=strategy,
        client_resources={'num_cpus': 2, 'num_gpus': 0},    # Optional argument
    )

    # ## 6. Save results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results.pkl"

    results = {"history": history, "config": "some config"}

    with open(str(results_path), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

# def visualize_client_data_distribution(trainloaders):
#     plt.figure(figsize=(10, 6))
#     client_data_counts = [len(loader.dataset) for loader in trainloaders]
#     plt.bar(range(len(client_data_counts)), client_data_counts, color='skyblue')
#     plt.xlabel("Client ID")
#     plt.ylabel("Number of Samples")
#     plt.title("Data Distribution Across Clients")
#     plt.xticks(range(len(client_data_counts)))
#     plt.grid(axis='y', linestyle='--', linewidth=0.7)
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_client_attack_distribution(trainloaders, num_classes=6):
    client_attack_counts = []

    # Count occurrences of each attack type per client
    for loader in trainloaders:
        attack_counts = np.zeros(num_classes)  # Initialize count array for each attack type

        for _, labels in loader.dataset:
            unique, counts = torch.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                attack_counts[int(u)] += int(c)

        client_attack_counts.append(attack_counts)

    client_attack_counts = np.array(client_attack_counts)  # Convert to NumPy array

    # Plot the stacked bar chart
    client_ids = np.arange(len(trainloaders))
    bar_width = 0.6
    attack_labels = [f"Attack {i}" for i in range(num_classes)]  # Label each attack type

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each attack type
    bottom = np.zeros(len(trainloaders))  # Start stacking from zero
    for i in range(num_classes):
        ax.bar(client_ids, client_attack_counts[:, i], bar_width, label=attack_labels[i], bottom=bottom)
        bottom += client_attack_counts[:, i]  # Stack the bars

    ax.set_xlabel("Client ID")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Distribution of Attack Types Across Clients")
    ax.set_xticks(client_ids)
    ax.legend(title="Attack Type")
    plt.grid(axis='y', linestyle='--', linewidth=0.7)
    plt.show()



if __name__ == "__main__":
    main()