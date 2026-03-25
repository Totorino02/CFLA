import torch
import numpy as np
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import datasets, transforms

from experiments.scripts.flhc import run_flhc_experiment
from experiments.scripts.hcfl import run_hcfl_experiment
from experiments.scripts.lcfed import run_lcfed_experiment
from experiments.scripts.fedper import run_fedper_experiment
from experiments.scripts.fedgroup import run_fedgroup_experiment
from experiments.scripts.fesem import run_fesem_experiment
from experiments.scripts.cgpfl import run_cgpfl_experiment
from experiments.scripts.ifca import run_ifca_experiment


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

_DATASET_CONFIGS = {
    "mnist": {
        "cls": datasets.MNIST,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        # (class_range_start, class_range_end_inclusive, nb_clients)
        "groups": [(0, 4, 18), (4, 7, 17), (7, 9, 15)],
        "num_classes": 10,
    },
    "cifar10": {
        "cls": datasets.CIFAR10,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]),
        # 3 groups: classes 0-3 / 3-6 / 6-9
        "groups": [(0, 3, 18), (3, 6, 17), (6, 9, 15)],
        "num_classes": 10,
    },
    "cifar100": {
        "cls": datasets.CIFAR100,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]),
        # 5 groups of 20 classes each, 10 clients per group → 50 clients
        "groups": [(0, 19, 10), (20, 39, 10), (40, 59, 10), (60, 79, 10), (80, 99, 10)],
        "num_classes": 100,
    },
}


def _split_into_n(ds, n, seed):
    length = len(ds)
    base = length // n
    sizes = [base] * n
    for i in range(length - base * n):
        sizes[i] += 1
    return random_split(ds, sizes, generator=torch.Generator().manual_seed(seed))


def build_client_datasets(base_seed: int, run: int, dataset_name: str = "mnist"):
    """
    Shared structured partitioning used by all algorithms.
    Returns (client_datasets, test_loader, num_classes).

    Clients are split into groups that each cover a contiguous range of classes,
    creating a naturally non-IID setting for CFL benchmarking.

    Supported datasets: "mnist", "cifar10", "cifar100".
    """
    cfg = _DATASET_CONFIGS[dataset_name]
    transform = cfg["transform"]

    train_ds = cfg["cls"](root="./data", download=True, train=True, transform=transform)
    test_ds  = cfg["cls"](root="./data", download=True, train=False, transform=transform)

    # CIFAR targets are a plain list; convert to tensor for masking
    targets = train_ds.targets
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    client_datasets = {}
    cid = 0
    for i, (lo, hi, n_clients) in enumerate(cfg["groups"]):
        mask = (targets >= lo) & (targets <= hi)
        subset = Subset(train_ds, torch.where(mask)[0].tolist())
        partitions = _split_into_n(subset, n_clients, seed=base_seed + run * (i + 2))
        for p in partitions:
            client_datasets[cid] = p
            cid += 1

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    return client_datasets, test_loader, cfg["num_classes"]


if __name__ == "__main__":
    base_seed = 2026
    for ds in ("cifar10",):  # ajouter "cifar10", "cifar100" pour tester sur ces datasets
        run_flhc_experiment(nb_runs=1, base_seed=base_seed, dataset=ds)
        run_hcfl_experiment(nb_runs=1, base_seed=base_seed, dataset=ds)
        run_lcfed_experiment(nb_runs=1, base_seed=base_seed, dataset=ds)
