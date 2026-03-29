"""
FedGroup server — offline clustered FL (Tan et al., 2022)

Algorithm:
  1. Warm-up: standard FedAvg for initial_rounds
  2. Clustering (ONCE): all clients train one extra round, server collects
     full param vectors and clusters them by cosine similarity
     (AgglomerativeClustering, same approach as FLHC but on params, not updates)
  3. Personalization: per-cluster FedAvg for cluster_rounds
"""

import copy
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader
from tqdm import tqdm

from framework.server.serverbase import Server
from framework.client.client_fedgroup import ClientFedGroup
from framework.common.utils import average_state_dict, flatten_params


class ServerFedGroup(Server):
    """
    FedGroup server.
    Clustering is performed ONCE after warm-up, using cosine similarity of
    client full model parameter vectors.
    """

    def __init__(self, global_model, test_dataloader, args, **kwargs):
        super().__init__()
        self.global_model = global_model
        self.test_dataloader = test_dataloader
        self.args = args

        self.device = torch.device(args["device"])
        self.fraction = float(args.get("fraction", 0.2))
        self.initial_rounds = int(args.get("initial_rounds", 3))
        self.cluster_rounds = int(args.get("cluster_rounds", 50))
        self.distance_threshold = float(args.get("distance_threshold", 0.5))
        self.seed = int(args.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)

        self.clients: List[ClientFedGroup] = []
        self.clusters: Dict[int, List[ClientFedGroup]] = {}
        self.output_dir = args["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        # cluster assignment: R[cid] = cluster_id
        self.R: List[int] = []
        self.cluster_centers: List[Dict[str, torch.Tensor]] = []
        self.num_clusters: int = 0

    def set_clients(self, clients: List[ClientFedGroup]):
        self.clients = clients
        for c in self.clients:
            c.output_dir = self.output_dir
        self.R = [-1] * len(clients)
        with open(os.path.join(self.output_dir, "server_metrics.csv"), "w") as f:
            f.write("round,mean_acc,std_acc,mean_loss\n")

    def select_clients(self, pool: Optional[List[ClientFedGroup]] = None) -> List[ClientFedGroup]:
        if pool is None:
            pool = self.clients
        m = max(1, int(self.fraction * len(pool)))
        return list(self.rng.choice(pool, m, replace=False))

    # ------------------------------------------------------------------
    # Phase 1 — Warm-up (standard FedAvg)
    # ------------------------------------------------------------------

    def _fedavg_round(self, model: nn.Module, pool: List[ClientFedGroup], round_idx: int) -> float:
        """One FedAvg round on `pool`. Updates `model` in place. Returns mean loss."""
        selected = self.select_clients(pool)
        state_dicts, losses = [], []
        for c in selected:
            state, loss, _ = c.train(model, round=round_idx, save_metrics=False)
            state_dicts.append(state)
            losses.append(loss)
        averaged = average_state_dict(state_dicts)
        model.load_state_dict(averaged)
        return float(np.mean(losses)) if losses else 0.0

    def pre_learning(self):
        """Warm-up: initial_rounds of FedAvg on all clients."""
        for epoch in range(self.initial_rounds):
            self._fedavg_round(self.global_model, self.clients, round_idx=-(self.initial_rounds - epoch))

    # ------------------------------------------------------------------
    # Phase 2 — Clustering (ONCE, on full param vectors)
    # ------------------------------------------------------------------

    def _collect_params_and_cluster(self):
        """
        All clients train one round from global model; server collects full
        parameter vectors and clusters by cosine similarity.
        """
        param_vecs = []
        for c in self.clients:
            state, _, _ = c.train(self.global_model, round=self.initial_rounds, save_metrics=False)
            vec = torch.cat([v.flatten() for v in state.values()]).cpu().numpy()
            param_vecs.append(vec)

        param_matrix = np.stack(param_vecs)  # [N, P]

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(param_matrix)

        clusters: Dict[int, List[ClientFedGroup]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(self.clients[idx])
        return clusters

    # ------------------------------------------------------------------
    # Phase 3 — Per-cluster personalization (cluster FedAvg)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval_and_log(self, round_idx: int, selected_ids: set):
        accs, losses = [], []
        for c in self.clients:
            k = self.R[c.client_id]
            eval_model = type(self.global_model)().to(self.device)
            eval_model.load_state_dict(self.cluster_centers[k])
            acc, _, loss = c.evaluate(eval_model)
            accs.append(acc)
            losses.append(loss)
            if c.client_id not in selected_ids:
                with open(
                    os.path.join(self.output_dir, f"client_{c.client_id}", "metrics.csv"), "a"
                ) as f:
                    f.write(f"{round_idx},{loss},{acc},{acc},0,0\n")

        with open(os.path.join(self.output_dir, "server_metrics.csv"), "a") as f:
            f.write(
                f"{round_idx},{np.mean(accs):.6f},{np.std(accs):.6f},{np.mean(losses):.6f}\n"
            )

    def evaluate(self, model: nn.Module, dataloader: DataLoader, k: int = 1, return_loss: bool = False):
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        loss_fn = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                _, pred = outputs.max(dim=1)
                correct += (pred == targets).sum().item()
                total += targets.size(0)
                total_loss += loss.item() * targets.size(0)
        acc = correct / max(1, total)
        avg_loss = total_loss / max(1, total)
        return (acc, avg_loss) if return_loss else acc

    def train(self, verbose: bool = True):
        start = time.time()

        # Phase 1: warm-up
        print("FedGroup — warm-up...")
        self.pre_learning()

        # Phase 2: clustering
        print("FedGroup — clustering...")
        self.clusters = self._collect_params_and_cluster()
        self.num_clusters = len(self.clusters)
        print(f"  → {self.num_clusters} clusters formed")
        for cid_label, members in self.clusters.items():
            print(f"     Cluster {cid_label} [{len(members)}]: {[c.client_id for c in members]}")

        # Init cluster centers + assignment map
        self.cluster_centers = [
            copy.deepcopy(self.global_model.state_dict()) for _ in range(self.num_clusters)
        ]
        for label, members in self.clusters.items():
            for c in members:
                self.R[c.client_id] = label

        # Phase 3: per-cluster FedAvg
        print("FedGroup — personalization...")
        for _round in tqdm(range(self.cluster_rounds), unit="round", colour="green"):
            selected_ids: set = set()
            # Train each cluster independently
            for label, cluster_clients in self.clusters.items():
                cluster_model = type(self.global_model)().to(self.device)
                cluster_model.load_state_dict(self.cluster_centers[label])

                selected = self.select_clients(cluster_clients)
                for c in selected:
                    selected_ids.add(c.client_id)

                state_dicts, losses = [], []
                for c in selected:
                    state, loss, _ = c.train(cluster_model, round=_round, verbose=False)
                    state_dicts.append(state)
                    losses.append(loss)

                self.cluster_centers[label] = average_state_dict(state_dicts)

            self._eval_and_log(_round, selected_ids)

            if verbose and ((_round + 1) % self.args.get("log_every", 10) == 0):
                print(f"[Round {_round+1:04d}]")

        if verbose:
            print(f"FedGroup training done in {time.time() - start:.1f}s")

        return self.cluster_centers, self.clusters

    def aggregate(self, **kwargs):
        pass  # handled inside train()

    def get_params(self):
        return self.global_model.state_dict()
