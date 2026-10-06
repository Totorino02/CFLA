import torch

DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

import datetime
import os

import numpy as np

from cfla.framework.client.client_hcfl import ClientHCFL
from cfla.framework.models.computer_vision import SplitCNNCifar, SplitLeNet5V1, SplitLeNet5Cifar
from cfla.framework.server.server_hcfl import ServerHCFL


def _get_model(dataset: str, num_classes: int):
    if dataset == "mnist":
        return (SplitLeNet5V1(),)
    if dataset == "cifar10" or dataset == "cifar100":
        return SplitLeNet5Cifar(num_classes=num_classes)
    return SplitCNNCifar(num_classes=num_classes)


def run_hcfl_experiment(
    nb_runs=1,
    base_seed=42,
    dataset="mnist",
    noise_ratio=0.0,
    nb_rounds=50,
    local_epochs=3,
    save_test_dataset=False,
    output_dir="./RESULTS",
    lambda_0=1.0,
    lambda_alpha=0.1,
    lambda_p=1.0,
):
    from cfla.experiments.scripts.run_all_mnist import build_client_datasets

    for run in range(nb_runs):
        seed = base_seed + run
        torch.manual_seed(seed)
        np.random.seed(seed)

        client_datasets, test_loader, num_classes = build_client_datasets(
            base_seed, run, dataset, noise_ratio=noise_ratio
        )

        stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
        result_dir = os.path.join(output_dir, f"result_hcfl_{dataset}_{stamp}")
        os.makedirs(result_dir, exist_ok=True)

        client_args = {
            "local_epochs": local_epochs,
            "local_steps": 0,
            "device": DEVICE,
            "optimizer": torch.optim.SGD,
            "criterion": torch.nn.CrossEntropyLoss(reduction="mean"),
            "learning_rate": 0.01,
            "batch_size": 32,
            "train_fraction": 0.2,
            "mu": 1.0,
            "monitor_energy": False,
            "output_dir": result_dir,
            "seed": seed,
            "save_test_dataset": save_test_dataset,
        }

        clients = [
            ClientHCFL(client_id=cid, dataset=ds, output_dir=result_dir, args=client_args)
            for cid, ds in client_datasets.items()
            if len(ds) > 64
        ]

        server_args = {
            "fraction": 0.2,
            "device": DEVICE,
            "initial_rounds": 5,
            "cluster_rounds": nb_rounds,
            "distance_threshold": 0.5,
            "clustering_metric": "cosine",
            "lambda_0": lambda_0,
            "lambda_alpha": lambda_alpha,
            "lambda_p": lambda_p,
            "output_dir": result_dir,
            "seed": seed,
            "log_every": 10,
        }

        server = ServerHCFL(
            global_model=_get_model(dataset, num_classes),
            args=server_args,
            test_dataloader=test_loader,
        )
        server.set_clients(clients=clients)
        server.train(verbose=True)


if __name__ == "__main__":
    run_hcfl_experiment(
        nb_runs=1, base_seed=42, dataset="mnist", lambda_0=1.0, lambda_alpha=0.1, lambda_p=1.0
    )
