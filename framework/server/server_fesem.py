"""
FeSEM server — Federated learning with Self-organizing Expectation Maximization

Online clustering at each round using negative L2 distance between client model
parameters and cluster centers:
    k*(i) = argmin_k  ||ω_i − Ω_k||²

Algorithm each round:
  1. Select clients
  2. Send Ω_{R[i]} to each selected client
  3. Client trains locally (standard SGD, no regularization)
  4. Reassign: R[i] = argmin_k ||ω_i_new − Ω_k||²
  5. Aggregate: Ω_k = FedAvg of updated params for clients in cluster k
"""

import copy
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from framework.server.serverbase import Server
from framework.client.client_fesem import ClientFeSEM
from framework.common.utils import average_state_dict, flatten_params


class ServerFeSEM(Server):

    def __init__(self, global_model, test_dataloader, args, **kwargs):
        super().__init__()
        self.global_model = global_model
        self.test_dataloader = test_dataloader
        self.args = args

        self.device = torch.device(args["device"])
        self.fraction = float(args.get("fraction", 0.2))
        self.rounds = int(args.get("rounds", 100))
        self.num_clusters = int(args.get("num_clusters", 10))
        self.seed = int(args.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)

        self.clients: List[ClientFeSEM] = []
        self.output_dir = args["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        # K cluster centers (full state_dicts)
        self.cluster_centers: List[Dict[str, torch.Tensor]] = []
        # Assignment vector: R[cid] = k
        self.R: List[int] = []

    def set_clients(self, clients: List[ClientFeSEM]):
        self.clients = clients
        for c in self.clients:
            c.output_dir = self.output_dir

        max_id = max(c.client_id for c in self.clients)
        # Random initialization
        rng_init = np.random.default_rng(self.seed)
        self.R = [int(rng_init.integers(0, self.num_clusters)) for _ in range(max_id + 1)]

        # Init K cluster centers as copies of global model
        self.cluster_centers = [
            copy.deepcopy(self.global_model.state_dict()) for _ in range(self.num_clusters)
        ]

        with open(os.path.join(self.output_dir, "server_metrics.csv"), "w") as f:
            f.write("round,mean_acc,std_acc,mean_loss\n")

    def select_clients(self) -> List[ClientFeSEM]:
        m = max(1, int(self.fraction * len(self.clients)))
        return list(self.rng.choice(self.clients, m, replace=False))

    @torch.no_grad()
    def _l2_distance(self, flat_vec: torch.Tensor, center_state: Dict[str, torch.Tensor]) -> float:
        """||ω_i − Ω_k||² in parameter space."""
        tmp = type(self.global_model)().to(self.device)
        tmp.load_state_dict(center_state, strict=True)
        center_vec = flatten_params(tmp, device=self.device)
        return float(((flat_vec - center_vec) ** 2).sum().item())

    @torch.no_grad()
    def _reassign(self, client: ClientFeSEM):
        """Assign client to closest cluster by L2 distance."""
        flat_vec = client.get_flat_params(device=self.device)
        dists = [self._l2_distance(flat_vec, self.cluster_centers[k]) for k in range(self.num_clusters)]
        self.R[client.client_id] = int(np.argmin(dists))

    @torch.no_grad()
    def _aggregate(self, client_states: Dict[int, Dict[str, torch.Tensor]]):
        """Per-cluster FedAvg of selected clients' state_dicts."""
        for k in range(self.num_clusters):
            members = [cid for cid, state in client_states.items() if self.R[cid] == k]
            if not members:
                continue
            self.cluster_centers[k] = average_state_dict(
                [client_states[cid] for cid in members]
            )

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

    def evaluate(self, model: nn.Module, dataloader: DataLoader, return_loss: bool = False):
        model.eval()
        correct, total, total_loss = 0, 0, 0.0
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

        for r in range(self.rounds):
            selected = self.select_clients()
            selected_ids = {c.client_id for c in selected}

            client_states: Dict[int, Dict[str, torch.Tensor]] = {}
            losses: List[float] = []

            for c in selected:
                k_star = self.R[c.client_id]
                # Build cluster model on device
                cluster_model = type(self.global_model)().to(self.device)
                cluster_model.load_state_dict(self.cluster_centers[k_star], strict=True)

                state, loss, _ = c.train(cluster_model, round=r, verbose=False)
                client_states[c.client_id] = state
                losses.append(loss)

            # Reassign based on updated params (E-step)
            for c in selected:
                self._reassign(c)

            # Aggregate per cluster (M-step)
            self._aggregate(client_states)

            self._eval_and_log(r, selected_ids)

            if verbose and ((r + 1) % self.args.get("log_every", 10) == 0):
                mean_loss = float(np.mean(losses)) if losses else 0.0
                counts = [
                    sum(1 for c in self.clients if self.R[c.client_id] == k)
                    for k in range(self.num_clusters)
                ]
                print(
                    f"[Round {r+1:04d}] "
                    f"selected={len(selected_ids)}/{len(self.clients)} "
                    f"mean loss(sel)={mean_loss:.4f} "
                    f"cluster_sizes={counts}"
                )

        if verbose:
            print(f"FeSEM training done in {time.time() - start:.1f}s")

        return self.cluster_centers, self.R

    def aggregate(self, **kwargs):
        pass  # handled inside train()

    def get_params(self):
        return self.global_model.state_dict()
