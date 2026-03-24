import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import datetime
import os
from torch.utils.data import Subset, Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from datasets.emnist import EMNISTDataset, load_emnist_data
from datasets.dataset_utils import make_non_iid, patho, split_labels_structured
from framework.client.client_flhc import ClientFLHC
from framework.server.server_flhc import ServerFLHC
from framework.models.computer_vision import LeNet5V1


def run_flhc_experiment(nb_runs=1, base_seed=42):

    for run in range(nb_runs):

        # Définir les transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        emnist_train_data, emnist_test_data = load_emnist_data()
        emnist_ds = EMNISTDataset(emnist_train_data)

        labels = emnist_ds.labels
        dataset = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

        # Targets train
        targets = dataset.targets

        # --- 1️⃣ Masques pour 3 groupes (avec chevauchement sur 4 et 7) ---
        mask_0_4 = (targets >= 0) & (targets <= 4)
        mask_4_7 = (targets >= 4) & (targets <= 7)
        mask_7_9 = (targets >= 7) & (targets <= 9)

        # Sous-datasets train
        dataset_0_4 = Subset(dataset, torch.where(mask_0_4)[0])
        dataset_4_7 = Subset(dataset, torch.where(mask_4_7)[0])
        dataset_7_9 = Subset(dataset, torch.where(mask_7_9)[0])

        def split_into_n(dataset, n, seed=42):
            length = len(dataset)
            base_size = length // n
            sizes = [base_size] * n

            remainder = length - base_size * n
            for i in range(remainder):
                sizes[i] += 1

            generator = torch.Generator().manual_seed(seed)
            return random_split(dataset, sizes, generator=generator)

        # --- 2️⃣ Nombre de partitions par groupe (total = 50) ---
        # On fait par exemple : 18 + 17 + 15 = 50
        partitions_0_4 = split_into_n(dataset_0_4, 18, seed=base_seed + run * 2)
        partitions_4_7 = split_into_n(dataset_4_7, 17, seed=base_seed + run * 3)
        partitions_7_9 = split_into_n(dataset_7_9, 15, seed=base_seed + run * 4)

        # --- 3️⃣ Dictionnaire clients 0 → 49 ---
        client_datasets = {}

        client_id = 0

        # clients 0–17 → groupe 0–4
        for p in partitions_0_4:
            client_datasets[client_id] = p
            client_id += 1

        # clients 18–34 → groupe 4–7
        for p in partitions_4_7:
            client_datasets[client_id] = p
            client_id += 1

        # clients 35–49 → groupe 7–9
        for p in partitions_7_9:
            client_datasets[client_id] = p
            client_id += 1  
        
        # client_indices
        clients_data_indices = make_non_iid("dirichlet", dataset.targets, 20)
        #make_non_iid("pathological", dataset.targets, num_clients=50, classes_per_client=6)


        stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
        output_dir = os.path.join("./RESULTS", f"result_flhc_{stamp}")

        args = {
        "local_epochs" : 3,
        "device" : "cpu",
        "optimizer" : torch.optim.Adam,
        "criterion": torch.nn.CrossEntropyLoss(reduction='mean'),
        "learning_rate":  0.01,
        "batch_size": 32, 
        "train_fraction": 0.2,
        "output_dir": output_dir
        }

        clients : list[ClientFLHC] = list()

        for client_id, ds in client_datasets.items():
            if len(ds) > 64: # Ensure each client has enough data
                clients.append(ClientFLHC(
                    client_id=client_id, 
                    dataset=ds,
                    output_dir=output_dir,
                    args=args
                ))    
            
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


        server_args = {
        "fraction" : 0.2,
        "device" : "cpu",
        "initial_rounds": 3,
        "cluster_rounds": 50,
        "distance_threshold": 0.5,
        "clustering_metric": "cosine",
        "output_dir": output_dir,
        "seed": base_seed + run * 5,
        }

        LeNet = LeNet5V1()

        server = ServerFLHC(
            global_model=LeNet,
            args=server_args,
            test_dataloader=test_loader
        )

        server.set_clients(clients=clients)

        specialized_models, training_loss_history, test_loss_history, accuracy_history, global_test_accuracy_history, global_test_loss_history = server.train()

if __name__ == "__main__":
    run_flhc_experiment(nb_runs=1, base_seed=42)