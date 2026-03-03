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


nb_runs = 1

distance_thresholds = [1, 1, 1, 1, 1, 1]#[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

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

    # Masques
    mask_0_4 = targets < 5
    mask_5_9 = targets >= 5

    # Sous-datasets train
    dataset_0_4 = torch.utils.data.Subset(dataset, torch.where(mask_0_4)[0])
    dataset_5_9 = torch.utils.data.Subset(dataset, torch.where(mask_5_9)[0])

    def split_into_n(dataset, n, seed=42):
        length = len(dataset)
        base_size = length // n
        sizes = [base_size] * n
        
        # distribuer le reste si non divisible
        remainder = length - base_size * n
        for i in range(remainder):
            sizes[i] += 1

        generator = torch.Generator().manual_seed(seed)
        return random_split(dataset, sizes, generator=generator)
    
    partitions_0_4 = split_into_n(dataset_0_4, 10)
    partitions_5_9 = split_into_n(dataset_5_9, 10)

    # --- 4️⃣ Créer le dictionnaire 0 → 19 ---
    client_datasets = {}

    # clients 0–9 → classes 0–4
    for i in range(10):
        client_datasets[i] = partitions_0_4[i]

    # clients 10–19 → classes 5–9
    for i in range(10):
        client_datasets[i + 10] = partitions_5_9[i]
    
    # client_indices
    clients_data_indices = make_non_iid("dirichlet", dataset.targets, 20)
    #make_non_iid("pathological", dataset.targets, num_clients=50, classes_per_client=6)


    stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    output_dir = os.path.join("./RESULTS", f"result_fmnist_{stamp}")

    args = {
    "local_epochs" : 2,
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
    "cluster_rounds": 25,
    "distance_threshold": 1,
    "clustering_metric": "cosine",
    "output_dir": output_dir
    }

    LeNet = LeNet5V1()

    server = ServerFLHC(
        global_model=LeNet,
        args=server_args,
        test_dataloader=test_loader
    )

    server.set_clients(clients=clients)

    specialized_models, training_loss_history, test_loss_history, accuracy_history, global_test_accuracy_history, global_test_loss_history = server.train()

    """
    training_loss_df = pd.DataFrame(training_loss_history)
    training_loss_df.to_csv(f'results/fl_hc/dirichet/{run+1}_training_loss_history.csv', index_label='Round')

    test_loss_df = pd.DataFrame(test_loss_history)
    test_loss_df.to_csv(f'results/fl_hc/dirichet/{run+1}_test_loss_history.csv', index_label='Round')

    accuracy_history_df = pd.DataFrame(accuracy_history)
    accuracy_history_df.to_csv(f'results/fl_hc/dirichet/{run+1}_accuracy_history.csv', index_label='Round')

    global_test_accuracy_history_df = pd.DataFrame(global_test_accuracy_history)
    global_test_accuracy_history_df.to_csv(f'results/fl_hc/dirichet/{run+1}_global_test_accuracy_history.csv', index_label='Round')

    global_test_loss_history_df = pd.DataFrame(global_test_loss_history)
    global_test_loss_history_df.to_csv(f'results/fl_hc/dirichet/{run+1}_global_test_loss_history.csv', index_label='Round')
    """