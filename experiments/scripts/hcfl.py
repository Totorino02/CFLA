"""
Training script HCFL (arXiv:2501.01850) — même style que ton training FLHC.

Hypothèses:
- Tu as déjà les 2 classes:
    * framework.client.client_hcfl.ClientHCFL
    * framework.server.server_hcfl.ServerHCFL
    (celles que je t’ai données au tour précédent, ou ta variante intégrée à ton framework).
- Ton modèle global DOIT être "split" avec:
    model.embed  (ϕ)  +  model.head (h)

Si ton LeNet5V1 n’est PAS split, je fournis un wrapper SplitLeNet5V1 plus bas.
"""

import numpy as np
import torch
import datetime
import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from datasets.emnist import EMNISTDataset, load_emnist_data
from torch.utils.data import Subset, Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from datasets.emnist import EMNISTDataset, load_emnist_data
from datasets.dataset_utils import make_non_iid, patho, split_labels_structured

# ---- imports HCFL
from framework.client.client_hcfl import ClientHCFL
from framework.server.server_hcfl import ServerHCFL

# ---- modèle (à adapter)
from framework.models.computer_vision import LeNet5V1


# ============================================================
# 1) (Optionnel) Wrapper pour "splitter" un LeNet5V1 en embed/head
# ============================================================
import torch.nn as nn
class SplitLeNet5V1(nn.Module):
    """
    Split de LeNet5V1 pour HCFL:
        - embed = feature + classifier[:-1]  -> produit un embedding de taille 84
        - head  = classifier[-1]            -> Linear(84 -> 10)

    IMPORTANT:
        - On garde le Flatten + Linear(120) + Linear(84) dans embed
        - La couche de décision = dernière Linear (84->10)
    """
    def __init__(self):
        super().__init__()
        base = LeNet5V1()

        # sécurité
        assert hasattr(base, "feature") and hasattr(base, "classifier"), "LeNet5V1 doit avoir feature et classifier."
        assert isinstance(base.classifier, nn.Sequential) and len(base.classifier) >= 2, \
            "classifier doit être un nn.Sequential avec au moins 2 modules."

        # embed = feature + classifier sans la dernière couche
        self.embed = nn.Sequential(
            base.feature,
            *list(base.classifier.children())[:-1]  # jusqu'à Tanh() après Linear(120->84)
        )

        # head = dernière couche (Linear 84->10)
        self.head = list(base.classifier.children())[-1]

    def forward(self, x):
        z = self.embed(x)     # shape [B, 84]
        return self.head(z)   # shape [B, 10]


# ============================================================
# 2) Split MNIST en 2 groupes (0-4 / 5-9) comme ton exemple
# ============================================================
def split_into_n(dataset, n, seed=42):
    length = len(dataset)
    base_size = length // n
    sizes = [base_size] * n
    remainder = length - base_size * n
    for i in range(remainder):
        sizes[i] += 1
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, sizes, generator=generator)


# ============================================================
# 3) Training run
# ============================================================

def run_hcfl_experiment(nb_runs=1, base_seed=42):

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
        output_dir = os.path.join("./RESULTS", f"result_hcfl_mnist_{stamp}")
        os.makedirs(output_dir, exist_ok=True)

        # ---- args client HCFL
        # Notes:
        # - mu  : régularisation intra-cluster (||ω-Ω||²)
        # - lambda : régularisation embedding global (||ϕ-Φ||²)
        # - tu peux utiliser local_epochs OU local_steps (si ton ClientHCFL le supporte)
        client_args = {
            "local_epochs": 3,
            "local_steps": 0,          # mets >0 si tu veux "N itérations" au lieu d'epochs
            "device": "cpu",
            "optimizer": torch.optim.SGD,
            "criterion": torch.nn.CrossEntropyLoss(reduction="mean"),
            "learning_rate": 0.01,
            "batch_size": 32,
            "train_fraction": 0.2,
            "output_dir": output_dir,

            # HCFL
            "mu": 0,
            "lambda": 2.0,

            # optionnel
            "measure_energy": False,
        }

        # ---- build clients
        clients = []
        for client_id, ds in client_datasets.items():
            if len(ds) > 64:
                clients.append(ClientHCFL(
                    client_id=client_id,
                    dataset=ds,
                    output_dir=output_dir,
                    args=client_args
                ))

        # Global test loader
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        # ---- args serveur LCFed
        server_args = {
            "fraction": 0.2,
            "device": "cpu",

            # LCFed
            "num_clusters": 10,
            "low_rank_dim": 50,         # D dans PCA
            "pca_sample_clients": 20,   # |Sd|
            "seed": base_seed + run * 5,

            "log_every": 10,
            "initial_rounds": 1,
            "cluster_rounds": 50,
            "distance_threshold": 0.5,  # pour AgglomerativeClustering (si num_clusters=None)
            "clustering_metric": "cosine",
            "output_dir": output_dir,
        }

        # IMPORTANT: modèle split
        # Si tu as déjà un modèle split, utilise-le directement.
        # Sinon, tente le wrapper:
        global_model = SplitLeNet5V1().to(server_args["device"])

        server = ServerHCFL(
            global_model=global_model,
            args=server_args,
            test_dataloader=test_loader
        )

        server.set_clients(clients=clients)

        # Train
        Omega_k, Phi, R  = server.train(verbose=True)

        # Tu peux sauvegarder ce que tu veux:
        # - Omega_k : liste des state_dict cluster centers
        # - Phi     : state_dict embedding global
        # - R       : assignation client->cluster
        # torch.save({"Omega_k": Omega_k, "Phi": Phi, "R": R}, os.path.join(output_dir, "hcfl_artifacts.pt"))

        # print(f"Saved artifacts to: {os.path.join(output_dir, 'hcfl_artifacts.pt')}")

if __name__ == "__main__":
    run_hcfl_experiment(nb_runs=1, base_seed=42)