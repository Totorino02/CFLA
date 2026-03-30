"""
AG News experiment entry point.

Partitioning: Dirichlet(alpha=0.1) on 4 classes, 40 clients.
Models: TextCNNAGNews (monolithic) / SplitTextCNNAGNews (split).
vocab_size=25000 is baked into model defaults — must match build_client_datasets_agnews.
"""

import datetime
import inspect
import os

import numpy as np
import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

from datasets.ag_news import build_client_datasets_agnews
from framework.models.nlp_models import SplitTextCNNAGNews, TextCNNAGNews

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
    n_clients: int = 40,
    alpha: float = 0.1,
    max_vocab: int = 25000,
    max_len: int = 128,
):
    for run in range(nb_runs):
        seed = base_seed + run
        torch.manual_seed(seed)
        np.random.seed(seed)

        client_datasets, test_loader, num_classes, vocab = build_client_datasets_agnews(
            base_seed,
            run,
            n_clients=n_clients,
            alpha=alpha,
            max_vocab=max_vocab,
            max_len=max_len,
        )

        stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
        output_dir = os.path.join("./RESULTS", f"result_{algo_name}_agnews_{stamp}")
        os.makedirs(output_dir, exist_ok=True)

        client_args = {
            "local_epochs": 3,
            "local_steps": 0,
            "device": DEVICE,
            "optimizer": torch.optim.Adam,
            "criterion": torch.nn.CrossEntropyLoss(reduction="mean"),
            "learning_rate": 1e-3,
            "batch_size": 64,
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

        # Always use max_vocab for the model so type(model)() recreates the same shape.
        # Token ids are in [0, len(vocab)-1] ≤ max_vocab-1, unused rows are ignored.
        server = ServerCls(
            global_model=model_fn(max_vocab, num_classes),
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


def run_flhc_agnews(nb_runs=1, base_seed=2026, n_clients=40, alpha=0.1):
    from framework.client.client_flhc import ClientFLHC
    from framework.server.server_flhc import ServerFLHC

    _run(
        "flhc",
        ClientFLHC,
        ServerFLHC,
        model_fn=lambda vs, nc: TextCNNAGNews(vocab_size=vs, num_classes=nc),
        client_args_extra={},
        server_args_extra={
            "initial_rounds": 3,
            "cluster_rounds": 50,
            "distance_threshold": 0.5,
            "clustering_metric": "cosine",
        },
        nb_runs=nb_runs,
        base_seed=base_seed,
        n_clients=n_clients,
        alpha=alpha,
    )


def run_hcfl_agnews(
    nb_runs=1,
    base_seed=2026,
    n_clients=40,
    alpha=0.1,
    mu=1.0,
    lambda_0=1.0,
    lambda_alpha=0.1,
    lambda_p=1.0,
):
    from framework.client.client_hcfl import ClientHCFL
    from framework.server.server_hcfl import ServerHCFL

    _run(
        "hcfl",
        ClientHCFL,
        ServerHCFL,
        model_fn=lambda vs, nc: SplitTextCNNAGNews(vocab_size=vs, num_classes=nc),
        client_args_extra={"mu": mu},
        server_args_extra={
            "initial_rounds": 3,
            "cluster_rounds": 50,
            "distance_threshold": 0.5,
            "clustering_metric": "cosine",
            "lambda_0": lambda_0,
            "lambda_alpha": lambda_alpha,
            "lambda_p": lambda_p,
        },
        nb_runs=nb_runs,
        base_seed=base_seed,
        n_clients=n_clients,
        alpha=alpha,
    )


def run_lcfed_agnews(nb_runs=1, base_seed=2026, n_clients=40, alpha=0.1, mu=1.0, lam=2.0):
    from framework.client.client_lcfed import ClientLCFed
    from framework.server.serveur_lcfed import ServerLCFed

    _run(
        "lcfed",
        ClientLCFed,
        ServerLCFed,
        model_fn=lambda vs, nc: SplitTextCNNAGNews(vocab_size=vs, num_classes=nc),
        client_args_extra={"mu": mu, "lambda": lam},
        server_args_extra={
            "rounds": 50,
            "num_clusters": 3,
            "low_rank_dim": 50,
            "pca_sample_clients": 20,
        },
        nb_runs=nb_runs,
        base_seed=base_seed,
        n_clients=n_clients,
        alpha=alpha,
    )


def run_fedper_agnews(nb_runs=1, base_seed=2026, n_clients=40, alpha=0.1):
    from framework.client.client_fedper import ClientFedPer
    from framework.server.server_fedper import ServerFedPer

    _run(
        "fedper",
        ClientFedPer,
        ServerFedPer,
        model_fn=lambda vs, nc: SplitTextCNNAGNews(vocab_size=vs, num_classes=nc),
        client_args_extra={},
        server_args_extra={"rounds": 50},
        nb_runs=nb_runs,
        base_seed=base_seed,
        n_clients=n_clients,
        alpha=alpha,
    )


def run_fedgroup_agnews(nb_runs=1, base_seed=2026, n_clients=40, alpha=0.1):
    from framework.client.client_fedgroup import ClientFedGroup
    from framework.server.server_fedgroup import ServerFedGroup

    _run(
        "fedgroup",
        ClientFedGroup,
        ServerFedGroup,
        model_fn=lambda vs, nc: TextCNNAGNews(vocab_size=vs, num_classes=nc),
        client_args_extra={},
        server_args_extra={
            "initial_rounds": 3,
            "cluster_rounds": 50,
            "distance_threshold": 0.5,
        },
        nb_runs=nb_runs,
        base_seed=base_seed,
        n_clients=n_clients,
        alpha=alpha,
    )


def run_fesem_agnews(nb_runs=1, base_seed=2026, n_clients=40, alpha=0.1):
    from framework.client.client_fesem import ClientFeSEM
    from framework.server.server_fesem import ServerFeSEM

    _run(
        "fesem",
        ClientFeSEM,
        ServerFeSEM,
        model_fn=lambda vs, nc: TextCNNAGNews(vocab_size=vs, num_classes=nc),
        client_args_extra={},
        server_args_extra={"rounds": 50, "num_clusters": 4},
        nb_runs=nb_runs,
        base_seed=base_seed,
        n_clients=n_clients,
        alpha=alpha,
    )


def run_cgpfl_agnews(nb_runs=1, base_seed=2026, n_clients=40, alpha=0.1):
    from framework.client.client_cgpfl import ClientCGPFL
    from framework.server.server_cgpfl import ServerCGPFL

    _run(
        "cgpfl",
        ClientCGPFL,
        ServerCGPFL,
        model_fn=lambda vs, nc: TextCNNAGNews(vocab_size=vs, num_classes=nc),
        client_args_extra={"mu": 1.0},
        server_args_extra={"rounds": 50, "num_clusters": 4},
        nb_runs=nb_runs,
        base_seed=base_seed,
        n_clients=n_clients,
        alpha=alpha,
    )


def run_ifca_agnews(nb_runs=1, base_seed=2026, n_clients=40, alpha=0.1):
    from framework.client.client_ifca import ClientIFCA
    from framework.server.server_ifca import ServerIFCA

    _run(
        "ifca",
        ClientIFCA,
        ServerIFCA,
        model_fn=lambda vs, nc: TextCNNAGNews(vocab_size=vs, num_classes=nc),
        client_args_extra={},
        server_args_extra={"rounds": 50, "num_clusters": 4},
        nb_runs=nb_runs,
        base_seed=base_seed,
        n_clients=n_clients,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BASE_SEED = 2026
    NB_RUNS = 1
    N_CLIENTS = 40
    ALPHA = 0.1  # Low α → high heterogeneity (4 classes, so already quite non-IID)

    print("=" * 60)
    print(f"AG News | Dirichlet α={ALPHA} | {N_CLIENTS} clients")
    print("=" * 60)

    run_flhc_agnews(nb_runs=NB_RUNS, base_seed=BASE_SEED, n_clients=N_CLIENTS, alpha=ALPHA)
    run_hcfl_agnews(nb_runs=NB_RUNS, base_seed=BASE_SEED, n_clients=N_CLIENTS, alpha=ALPHA)
    run_lcfed_agnews(nb_runs=NB_RUNS, base_seed=BASE_SEED, n_clients=N_CLIENTS, alpha=ALPHA)
    run_fedper_agnews(nb_runs=NB_RUNS, base_seed=BASE_SEED, n_clients=N_CLIENTS, alpha=ALPHA)
    run_fedgroup_agnews(nb_runs=NB_RUNS, base_seed=BASE_SEED, n_clients=N_CLIENTS, alpha=ALPHA)
    run_fesem_agnews(nb_runs=NB_RUNS, base_seed=BASE_SEED, n_clients=N_CLIENTS, alpha=ALPHA)
    run_cgpfl_agnews(nb_runs=NB_RUNS, base_seed=BASE_SEED, n_clients=N_CLIENTS, alpha=ALPHA)
    run_ifca_agnews(nb_runs=NB_RUNS, base_seed=BASE_SEED, n_clients=N_CLIENTS, alpha=ALPHA)
