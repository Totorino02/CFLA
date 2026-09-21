import os
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, utils


class FEMNIST(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.download = download
        self.download_link = "https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz"
        self.file_md5 = "a8a28afae0e007f1acb87e37919a21db"
        self.train = train
        self.root = root
        self.training_file = f"{self.root}/FEMNIST/processed/femnist_train.pt"
        self.test_file = f"{self.root}/FEMNIST/processed/femnist_test.pt"
        self.user_list = f"{self.root}/FEMNIST/processed/femnist_user_keys.pt"

        if not os.path.exists(
            f"{self.root}/FEMNIST/processed/femnist_test.pt"
        ) or not os.path.exists(f"{self.root}/FEMNIST/processed/femnist_train.pt"):
            if self.download:
                self.dataset_download()
            else:
                raise RuntimeError("Dataset not found, set parameter download=True to download")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        cache_file = data_file.replace(".pt", "_cached.pt")
        if os.path.exists(cache_file):
            cached = torch.load(cache_file, weights_only=True)
            self.data = cached["data"]
            self.targets = cached["targets"]
            self.users = cached["users"]
        else:
            data_targets_users = torch.load(data_file, weights_only=False)
            # Pre-reshape to [N, 1, 28, 28] once at load time to avoid per-sample PIL conversion
            self.data = torch.Tensor(data_targets_users[0]).reshape(-1, 1, 28, 28)
            self.targets = torch.Tensor(data_targets_users[1])
            self.users = data_targets_users[2]
            torch.save(
                {"data": self.data, "targets": self.targets, "users": self.users}, cache_file
            )
        self.user_ids = torch.load(self.user_list, weights_only=False)

    def __getitem__(self, index):
        img, target, user = self.data[index], int(self.targets[index]), self.users[index]
        # data is already a [1, 28, 28] float tensor — no PIL roundtrip needed
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, user

    def dataset_download(self):
        paths = [f"{self.root}/FEMNIST/raw/", f"{self.root}/FEMNIST/processed/"]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

        # download files
        filename = self.download_link.split("/")[-1]
        utils.download_and_extract_archive(
            self.download_link,
            download_root=f"{self.root}/FEMNIST/raw/",
            filename=filename,
            md5=self.file_md5,
        )

        files = ["femnist_train.pt", "femnist_test.pt", "femnist_user_keys.pt"]
        for file in files:
            # move to processed dir
            shutil.move(
                os.path.join(f"{self.root}/FEMNIST/raw/", file), f"{self.root}/FEMNIST/processed/"
            )


# ---------------------------------------------------------------------------
# Wrapper — strips the user field so standard (img, target) is returned
# ---------------------------------------------------------------------------


class _FEMNISTWrapper(Dataset):
    """Wraps FEMNIST to return (img, target) instead of (img, target, user)."""

    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target, _ = self.base[idx]
        return img, target


# ---------------------------------------------------------------------------
# Dataset builder — Dirichlet partitioning on 62 classes
# ---------------------------------------------------------------------------


def build_client_datasets_femnist(
    base_seed: int,
    run: int,
    n_clients: int = 50,
    alpha: float = 0.5,
    min_samples: int = 64,
    max_samples_per_client: int = None,
    data_root: str = "./data",
):
    """
    Partition FEMNIST training data into n_clients via Dirichlet(alpha) on 62 classes.

    FEMNIST: 671 k train samples, 77 k test samples, 62 classes
    (digits 0-9, lowercase a-z, uppercase A-Z).

    Returns
    -------
    client_datasets : dict[int, Subset]
    test_loader     : DataLoader
    num_classes     : int (62)
    """
    num_classes = 62
    # Data is pre-shaped as [1, 28, 28] tensors — ToTensor() not needed
    transform = transforms.Compose(
        [
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_femnist = FEMNIST(root=data_root, train=True, transform=transform, download=True)
    test_femnist = FEMNIST(root=data_root, train=False, transform=transform, download=True)

    train_ds = _FEMNISTWrapper(train_femnist)
    test_ds = _FEMNISTWrapper(test_femnist)

    targets = train_femnist.targets.long().numpy()  # shape [N]

    rng = np.random.default_rng(base_seed + run * 997)

    # ---- Dirichlet allocation (same logic as main_dirichlet.py) ----
    proportions = rng.dirichlet(alpha * np.ones(num_classes), size=n_clients)  # [n_clients, 62]

    class_indices = []
    for c in range(num_classes):
        idx = np.where(targets == c)[0].tolist()
        rng.shuffle(idx)
        class_indices.append(idx)

    client_indices = [[] for _ in range(n_clients)]
    for c in range(num_classes):
        idx = class_indices[c]
        n = len(idx)
        alloc = (proportions[:, c] / proportions[:, c].sum() * n).astype(int)
        alloc[-1] = n - alloc[:-1].sum()
        alloc = np.maximum(alloc, 0)
        offset = 0
        for k in range(n_clients):
            end = offset + alloc[k]
            client_indices[k].extend(idx[offset:end])
            offset = end

    for k in range(n_clients):
        rng.shuffle(client_indices[k])
        if max_samples_per_client is not None and len(client_indices[k]) > max_samples_per_client:
            client_indices[k] = client_indices[k][:max_samples_per_client]

    client_datasets = {}
    for cid, indices in enumerate(client_indices):
        if len(indices) >= min_samples:
            client_datasets[cid] = Subset(train_ds, indices)

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    return client_datasets, test_loader, num_classes
