"""
AG News dataset utilities for Federated Learning.

AG News: 4 classes (World=0, Sports=1, Business=2, Sci/Tech=3),
120 000 train samples, 7 600 test samples.

Loading strategy (in order of preference):
  1. Local CSV cache (data_root/AG_NEWS/{train,test}.csv)
  2. Direct download via requests (skips torchtext C extension)
"""

import csv
import os
import re
from collections import Counter

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# ---------------------------------------------------------------------------
# Raw data loading — bypasses torchtext C extension
# ---------------------------------------------------------------------------

_AG_NEWS_URLS = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}


def _load_ag_news_csv(split: str, data_root: str) -> list:
    """
    Load AG News from local CSV cache or download.
    Returns list of (label_0indexed, text) tuples.
    CSV format: class_index (1-4), title, description
    """
    cache_dir = os.path.join(data_root, "AG_NEWS")
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, f"{split}.csv")

    if not os.path.exists(local_path):
        print(f"Downloading AG News {split} set...")
        resp = requests.get(_AG_NEWS_URLS[split], timeout=60)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)

    rows = []
    with open(local_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            label = int(row[0]) - 1  # 1-indexed → 0-indexed
            text = row[1] + " " + row[2]  # title + description
            rows.append((label, text))
    return rows


# ---------------------------------------------------------------------------
# Tokenizer & vocabulary
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list:
    """Lowercase + strip non-alphanumeric, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def build_vocab(tokenized_texts: list, max_vocab: int = 25000) -> dict:
    """
    Build word→id mapping from a list of token lists.
    PAD=0, UNK=1, then most-frequent words from index 2.
    """
    counter = Counter(tok for tokens in tokenized_texts for tok in tokens)
    most_common = [w for w, _ in counter.most_common(max_vocab - 2)]
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, w in enumerate(most_common, start=2):
        vocab[w] = i
    return vocab


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class AGNewsDataset(Dataset):
    """
    AG News dataset as a PyTorch Dataset.

    Each item is (token_ids: LongTensor[max_len], label: LongTensor scalar).
    Labels are 0-indexed (original torchtext labels are 1-indexed).
    """

    def __init__(self, tokens: list, labels: list, vocab: dict, max_len: int = 128):
        self.tokens = tokens
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self._PAD = vocab["<pad>"]
        self._UNK = vocab["<unk>"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = [self.vocab.get(w, self._UNK) for w in self.tokens[idx][: self.max_len]]
        ids += [self._PAD] * (self.max_len - len(ids))
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Dataset builder — loads AG News, builds vocab, Dirichlet FL partitioning
# ---------------------------------------------------------------------------


def build_client_datasets_agnews(
    base_seed: int,
    run: int,
    n_clients: int = 40,
    alpha: float = 0.1,
    max_len: int = 128,
    max_vocab: int = 25000,
    min_samples: int = 64,
    data_root: str = "./data",
):
    """
    Load AG News (CSV download/cache), build vocab, and partition training data
    into n_clients via Dirichlet(alpha) on 4 classes.

    Returns
    -------
    client_datasets : dict[int, Subset]
    test_loader     : DataLoader
    num_classes     : int (4)
    vocab           : dict (word → id, needed to build the model's vocab_size)
    """
    num_classes = 4

    # ---- Load raw data (CSV download, no torchtext C extension needed) ----
    train_raw = _load_ag_news_csv("train", data_root)
    test_raw = _load_ag_news_csv("test", data_root)

    # ---- Tokenize ----
    train_tokens = [_tokenize(text) for _, text in train_raw]
    test_tokens = [_tokenize(text) for _, text in test_raw]
    train_labels = [lbl for lbl, _ in train_raw]
    test_labels = [lbl for lbl, _ in test_raw]

    # ---- Build vocabulary from training set only ----
    vocab = build_vocab(train_tokens, max_vocab=max_vocab)

    # ---- Full datasets ----
    train_ds = AGNewsDataset(train_tokens, train_labels, vocab, max_len=max_len)
    test_ds = AGNewsDataset(test_tokens, test_labels, vocab, max_len=max_len)

    # ---- Dirichlet partitioning on 4 classes ----
    targets = np.array(train_labels)
    rng = np.random.default_rng(base_seed + run * 997)

    proportions = rng.dirichlet(alpha * np.ones(num_classes), size=n_clients)

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

    client_datasets = {}
    for cid, indices in enumerate(client_indices):
        if len(indices) >= min_samples:
            client_datasets[cid] = Subset(train_ds, indices)

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    return client_datasets, test_loader, num_classes, vocab
