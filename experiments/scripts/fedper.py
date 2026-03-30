import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
import datetime
import os

import numpy as np

from framework.client.client_fedper import ClientFedPer
from framework.models.computer_vision import SplitCNNCifar, SplitLeNet5V1
from framework.server.server_fedper import ServerFedPer


def _get_model(dataset: str, num_classes: int):
    if dataset == "mnist":
        return SplitLeNet5V1()
    return SplitCNNCifar(num_classes=num_classes)


def run_fedper_experiment(nb_runs=1, base_seed=42, dataset="mnist", noise_ratio=0.0, nb_rounds=50):
    from experiments.scripts.run_all_mnist import build_client_datasets

    for run in range(nb_runs):
        seed = base_seed + run
        torch.manual_seed(seed)
        np.random.seed(seed)

        client_datasets, test_loader, num_classes = build_client_datasets(
            base_seed, run, dataset, noise_ratio=noise_ratio
        )

        stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
        output_dir = os.path.join("./RESULTS", f"result_fedper_{dataset}_{stamp}")
        os.makedirs(output_dir, exist_ok=True)

        client_args = {
            "local_epochs": 3,
            "device": DEVICE,
            "learning_rate": 0.01,
            "batch_size": 32,
            "train_fraction": 0.2,
            "monitor_energy": False,
            "output_dir": output_dir,
            "seed": seed,
        }

        clients = [
            ClientFedPer(client_id=cid, dataset=ds, output_dir=output_dir, args=client_args)
            for cid, ds in client_datasets.items()
            if len(ds) > 64
        ]

        server_args = {
            "fraction": 0.2,
            "device": DEVICE,
            "rounds": nb_rounds,
            "output_dir": output_dir,
            "seed": seed,
            "log_every": 10,
        }

        server = ServerFedPer(
            global_model=_get_model(dataset, num_classes),
            args=server_args,
            test_dataloader=test_loader,
        )
        server.set_clients(clients=clients)
        server.train(verbose=True)


if __name__ == "__main__":
    run_fedper_experiment(nb_runs=1, base_seed=42, dataset="mnist")
