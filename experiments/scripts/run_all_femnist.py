"""
FEMNIST experiment entry point.

Partitioning: Dirichlet(alpha=0.5) on 62 classes, 50 clients.
Models: LeNet5FEMNIST (monolithic) / SplitLeNet5FEMNIST (split).
"""

import datetime
import inspect
import os

import numpy as np
import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

from datasets.femnist import build_client_datasets_femnist
from framework.models.computer_vision import LeNet5FEMNIST, SplitLeNet5FEMNIST


# ---------------------------------------------------------------------------
# Generic runner
# ---------------------------------------------------------------------------

def _run(
    algo_name: str,
    ClientCls,
    ServerCls,
    model_fn,
    client_args_extra: dict,
    server_args_extra: dict,
    nb_runs: int = 1,
    base_seed: int = 2026,
    n_clients: int = 50,
    alpha: float = 0.5,
    max_samples_per_client: int = None,
):
    for run in range(nb_runs):
        seed = base_seed + run
        torch.manual_seed(seed)
        np.random.seed(seed)

        client_datasets, test_loader, num_classes = build_client_datasets_femnist(
            base_seed, run, n_clients=n_clients, alpha=alpha,
            max_samples_per_client=max_samples_per_client,
        )

        stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
        output_dir = os.path.join("./RESULTS", f"result_{algo_name}_femnist_{stamp}")
        os.makedirs(output_dir, exist_ok=True)

        client_args = {
            "local_epochs": 3,
            "local_steps": 0,
            "device": DEVICE,
            "optimizer": torch.optim.SGD,
            "criterion": torch.nn.CrossEntropyLoss(reduction="mean"),
            "learning_rate": 0.01,
            "batch_size": 32,
            "train_fraction": 0.2,
            "mu": 1.0,
            "lambda": 2.0,
            "monitor_energy": False,
            "output_dir": output_dir,
            "seed": seed,
        }
        client_args.update(client_args_extra)

        clients = [
            ClientCls(client_id=cid, dataset=ds, output_dir=output_dir, args=client_args)
            for cid, ds in client_datasets.items()
            if len(ds) > 64
        ]

        server_args = {
            "fraction": 0.2,
            "device": DEVICE,
            "output_dir": output_dir,
            "seed": seed,
            "log_every": 10,
        }
        server_args.update(server_args_extra)

        server = ServerCls(
            global_model=model_fn(num_classes),
            args=server_args,
            test_dataloader=test_loader,
        )
        server.set_clients(clients=clients)
        if "verbose" in inspect.signature(server.train).parameters:
            server.train(verbose=True)
        else:
            server.train()


# ---------------------------------------------------------------------------
# Algorithm runners
# ---------------------------------------------------------------------------

def run_flhc_femnist(nb_runs=1, base_seed=2026, n_clients=50, alpha=0.5, max_samples_per_client=None):
    from framework.client.client_flhc import ClientFLHC
    from framework.server.server_flhc import ServerFLHC
    _run(
        "flhc", ClientFLHC, ServerFLHC,
        model_fn=lambda nc: LeNet5FEMNIST(num_classes=nc),
        client_args_extra={},
        server_args_extra={
            "initial_rounds": 3, "cluster_rounds": 50,
            "distance_threshold": 0.5, "clustering_metric": "cosine",
        },
        nb_runs=nb_runs, base_seed=base_seed, n_clients=n_clients, alpha=alpha,
        max_samples_per_client=max_samples_per_client,
    )


def run_hcfl_femnist(nb_runs=1, base_seed=2026, n_clients=50, alpha=0.5,
                     mu=1.0, lambda_0=1.0, lambda_alpha=0.1, lambda_p=1.0,
                     max_samples_per_client=None):
    from framework.client.client_hcfl import ClientHCFL
    from framework.server.server_hcfl import ServerHCFL
    # μ : régularisation cluster côté client (statique)
    # λ(t) = λ₀/(1+α·t)^p : mélange Ω_k ← (1-λ)·Avg + λ·Φ côté serveur
    _run(
        "hcfl", ClientHCFL, ServerHCFL,
        model_fn=lambda nc: SplitLeNet5FEMNIST(num_classes=nc),
        client_args_extra={"mu": mu},
        server_args_extra={
            "initial_rounds": 3, "cluster_rounds": 50,
            "distance_threshold": 0.5, "clustering_metric": "cosine",
            "lambda_0": lambda_0, "lambda_alpha": lambda_alpha, "lambda_p": lambda_p,
        },
        nb_runs=nb_runs, base_seed=base_seed, n_clients=n_clients, alpha=alpha,
        max_samples_per_client=max_samples_per_client,
    )


def run_lcfed_femnist(nb_runs=1, base_seed=2026, n_clients=50, alpha=0.5, mu=1.0, lam=2.0, max_samples_per_client=None):
    from framework.client.client_lcfed import ClientLCFed
    from framework.server.serveur_lcfed import ServerLCFed
    _run(
        "lcfed", ClientLCFed, ServerLCFed,
        model_fn=lambda nc: SplitLeNet5FEMNIST(num_classes=nc),
        client_args_extra={"mu": mu, "lambda": lam},
        server_args_extra={
            "rounds": 50, "num_clusters": 3,
            "low_rank_dim": 50, "pca_sample_clients": 20,
        },
        nb_runs=nb_runs, base_seed=base_seed, n_clients=n_clients, alpha=alpha,
        max_samples_per_client=max_samples_per_client,
    )


def run_fedper_femnist(nb_runs=1, base_seed=2026, n_clients=50, alpha=0.5, max_samples_per_client=None):
    from framework.client.client_fedper import ClientFedPer
    from framework.server.server_fedper import ServerFedPer
    _run(
        "fedper", ClientFedPer, ServerFedPer,
        model_fn=lambda nc: SplitLeNet5FEMNIST(num_classes=nc),
        client_args_extra={},
        server_args_extra={"rounds": 50},
        nb_runs=nb_runs, base_seed=base_seed, n_clients=n_clients, alpha=alpha,
        max_samples_per_client=max_samples_per_client,
    )


def run_fedgroup_femnist(nb_runs=1, base_seed=2026, n_clients=50, alpha=0.5, max_samples_per_client=None):
    from framework.client.client_fedgroup import ClientFedGroup
    from framework.server.server_fedgroup import ServerFedGroup
    _run(
        "fedgroup", ClientFedGroup, ServerFedGroup,
        model_fn=lambda nc: LeNet5FEMNIST(num_classes=nc),
        client_args_extra={},
        server_args_extra={
            "initial_rounds": 3, "cluster_rounds": 50,
            "distance_threshold": 0.5,
        },
        nb_runs=nb_runs, base_seed=base_seed, n_clients=n_clients, alpha=alpha,
        max_samples_per_client=max_samples_per_client,
    )


def run_fesem_femnist(nb_runs=1, base_seed=2026, n_clients=50, alpha=0.5, max_samples_per_client=None):
    from framework.client.client_fesem import ClientFeSEM
    from framework.server.server_fesem import ServerFeSEM
    _run(
        "fesem", ClientFeSEM, ServerFeSEM,
        model_fn=lambda nc: LeNet5FEMNIST(num_classes=nc),
        client_args_extra={},
        server_args_extra={"rounds": 50, "num_clusters": 10},
        nb_runs=nb_runs, base_seed=base_seed, n_clients=n_clients, alpha=alpha,
        max_samples_per_client=max_samples_per_client,
    )


def run_cgpfl_femnist(nb_runs=1, base_seed=2026, n_clients=50, alpha=0.5, max_samples_per_client=None):
    from framework.client.client_cgpfl import ClientCGPFL
    from framework.server.server_cgpfl import ServerCGPFL
    _run(
        "cgpfl", ClientCGPFL, ServerCGPFL,
        model_fn=lambda nc: LeNet5FEMNIST(num_classes=nc),
        client_args_extra={"mu": 1.0},
        server_args_extra={"rounds": 50, "num_clusters": 10},
        nb_runs=nb_runs, base_seed=base_seed, n_clients=n_clients, alpha=alpha,
        max_samples_per_client=max_samples_per_client,
    )


def run_ifca_femnist(nb_runs=1, base_seed=2026, n_clients=50, alpha=0.5, max_samples_per_client=None):
    from framework.client.client_ifca import ClientIFCA
    from framework.server.server_ifca import ServerIFCA
    _run(
        "ifca", ClientIFCA, ServerIFCA,
        model_fn=lambda nc: LeNet5FEMNIST(num_classes=nc),
        client_args_extra={},
        server_args_extra={"rounds": 50, "num_clusters": 3},
        nb_runs=nb_runs, base_seed=base_seed, n_clients=n_clients, alpha=alpha,
        max_samples_per_client=max_samples_per_client,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BASE_SEED = 2026
    NB_RUNS   = 1

    # -- Few-data regime: clustering beneficial over solo learning --
    # Each client has ~100 samples (too few to generalize alone).
    # Low alpha (0.1) creates strong heterogeneity across clients so that
    # clustering similar clients together gives a real benefit.
    # 200 clients × 100 samples = 20k samples used out of 671k available.
    N_CLIENTS            = 50
    ALPHA                = 0.1   # low = high heterogeneity between clients
    MAX_SAMPLES          = 300   # cap per client

    print("=" * 60)
    print(f"FEMNIST | Few-data regime | Dirichlet α={ALPHA}")
    print(f"{N_CLIENTS} clients | max {MAX_SAMPLES} samples/client")
    print("=" * 60)

    kwargs = dict(nb_runs=NB_RUNS, base_seed=BASE_SEED, n_clients=N_CLIENTS,
                alpha=ALPHA, max_samples_per_client=MAX_SAMPLES)

    run_flhc_femnist(**kwargs)
    run_hcfl_femnist(**kwargs, lambda_0=1.0, lambda_alpha=0.1, lambda_p=1.0)
    run_lcfed_femnist(**kwargs)
    run_fedper_femnist(**kwargs)
    run_fedgroup_femnist(**kwargs)
    run_fesem_femnist(**kwargs)
    run_cgpfl_femnist(**kwargs)
    run_ifca_femnist(**kwargs)
