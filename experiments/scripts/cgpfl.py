import torch
import numpy as np
import datetime
import os

from framework.client.client_cgpfl import ClientCGPFL
from framework.server.server_cgpfl import ServerCGPFL
from framework.models.computer_vision import LeNet5V1, CNNCifar


def _get_model(dataset: str, num_classes: int):
    if dataset == "mnist":
        return LeNet5V1()
    return CNNCifar(num_classes=num_classes)


def run_cgpfl_experiment(nb_runs=1, base_seed=42, dataset="mnist"):
    from experiments.scripts.main import build_client_datasets

    for run in range(nb_runs):
        seed = base_seed + run
        torch.manual_seed(seed)
        np.random.seed(seed)

        client_datasets, test_loader, num_classes = build_client_datasets(base_seed, run, dataset)

        stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
        output_dir = os.path.join("./RESULTS", f"result_cgpfl_{dataset}_{stamp}")
        os.makedirs(output_dir, exist_ok=True)

        client_args = {
            "local_epochs": 3,
            "device": "cpu",
            "learning_rate": 0.01,
            "batch_size": 32,
            "train_fraction": 0.2,
            "mu": 1.0,
            "monitor_energy": False,
            "output_dir": output_dir,
            "seed": seed,
        }

        clients = [
            ClientCGPFL(client_id=cid, dataset=ds, output_dir=output_dir, args=client_args)
            for cid, ds in client_datasets.items()
            if len(ds) > 64
        ]

        server_args = {
            "fraction": 0.2,
            "device": "cpu",
            "rounds": 50,
            "num_clusters": 10,
            "output_dir": output_dir,
            "seed": seed,
            "log_every": 10,
        }

        server = ServerCGPFL(
            global_model=_get_model(dataset, num_classes),
            args=server_args,
            test_dataloader=test_loader,
        )
        server.set_clients(clients=clients)
        server.train(verbose=True)


if __name__ == "__main__":
    run_cgpfl_experiment(nb_runs=1, base_seed=42, dataset="mnist")
