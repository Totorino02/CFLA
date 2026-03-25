# ============================================================
# LCFed (arXiv:2501.01850) — Client/Server architecture
#   - Two classes: ClientLCFed and ServerLCFed
#   - PyTorch, split model: embed (ϕ) + head (h)
#   - Server: global embedding Φ + per-cluster centers Ω_k
#   - Online clustering using low-rank PCA projection M
# ============================================================

from __future__ import annotations

import os
import copy
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from sklearn.cluster import KMeans

from framework.common.utils import flatten_params, pca_projection_matrix, cosine_sim, average_state_dict
from framework.client.client_lcfed import ClientLCFed

# ============================================================
# ServerLCFed
# ============================================================

class ServerLCFed:
    """
    LCFed server:
        - maintains:
            * global model (used as initialization / reference)
            * global embedding Φ (embed state_dict)
            * K cluster centers Ω_k (full state_dicts)
            * cluster assignment R_i for each client
            * low-rank PCA projection matrix M
        - each round:
            1) select clients
            2) send Φ and Ω_{R_i} + global_model init
            3) receive ω_i and z_i
            4) update assignments via cosine similarity in low-rank space
            5) aggregate Φ globally; aggregate Ω_k per-cluster
    """

    def __init__(self, global_model, test_dataloader, args, **kwargs):
        self.global_model = global_model
        self.test_dataloader = test_dataloader
        self.args = args

        self.device = torch.device(args["device"])
        self.fraction = float(args.get("fraction", 0.2))
        self.rounds = int(args.get("rounds", 100))

        # LCFed-specific
        self.num_clusters = int(args.get("num_clusters", 10))
        self.low_rank_dim = int(args.get("low_rank_dim", 50))
        self.pca_sample_clients = int(args.get("pca_sample_clients", 20))
        self.seed = int(args.get("seed", 0))
        self.rng = random.Random(self.seed)

        self.clients: List[ClientLCFed] = []
        self.output_dir = args["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        # State
        self.R: List[int] = []  # cluster assignment per client index (must align with client.client_id)
        self.cluster_centers: List[Dict[str, torch.Tensor]] = []
        self.global_phi: Dict[str, torch.Tensor] = {}
        self.M: Optional[torch.Tensor] = None  # [P, D]

        # optional metrics
        self.history = []

    def set_clients(self, clients: List[ClientLCFed]):
        self.clients = clients
        for c in self.clients:
            c.output_dir = self.output_dir

        # init assignments
        max_id = max(c.client_id for c in self.clients)
        # assume client_id are 0..N-1; if not, you can map ids -> index
        self.R = [self.rng.randrange(self.num_clusters) for _ in range(max_id + 1)]

        # init centers + global phi
        self.cluster_centers = [copy.deepcopy(self.global_model.state_dict()) for _ in range(self.num_clusters)]
        self.global_phi = copy.deepcopy(self.global_model.embed.state_dict())

        # Initialize server metrics CSV
        with open(os.path.join(self.output_dir, "server_metrics.csv"), "w") as f:
            f.write("round,mean_acc,std_acc,mean_loss\n")

    @torch.no_grad()
    def compute_M_and_broadcast(self):
        """
        Sample Sd clients, build PCA projection matrix M on their flattened ω vectors.
        Then broadcast M to all clients.
        """
        if not self.clients:
            raise RuntimeError("ServerLCFed: no clients set.")

        Sd = self.rng.sample(self.clients, k=min(self.pca_sample_clients, len(self.clients)))
        W = []
        for c in Sd:
            # if client has not trained yet, it may have no local_model: use global_model clone
            tmp = type(self.global_model)().cpu()
            tmp.load_state_dict(self.global_model.state_dict(), strict=True)
            W.append(flatten_params(tmp).cpu())
        W = torch.stack(W, dim=0)  # [|Sd|, P]

        M = pca_projection_matrix(W, D=self.low_rank_dim)  # [P, D]
        self.M = M.to(self.device)

        for c in self.clients:
            c.set_M(self.M)

    @torch.no_grad()
    def low_rank_of_center(self, center_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute z_k = vec(Ω_k) @ M
        """
        if self.M is None:
            raise RuntimeError("ServerLCFed: M is None.")
        tmp = type(self.global_model)().to(self.device)
        tmp.load_state_dict(center_state, strict=True)
        vec = flatten_params(tmp, device=self.device)
        return (vec @ self.M).detach()

    @torch.no_grad()
    def update_assignments(self, client_z: Dict[int, torch.Tensor]):
        """
        R_i = argmax_k cos(z_i, z_k_center)
        """
        centers_z = [self.low_rank_of_center(self.cluster_centers[k]).cpu() for k in range(self.num_clusters)]
        for cid, zi in client_z.items():
            zi = zi.cpu()
            sims = [cosine_sim(zi, centers_z[k]) for k in range(self.num_clusters)]
            best_k = int(max(range(self.num_clusters), key=lambda k: sims[k]))
            # assumes cid usable as index; otherwise map
            self.R[cid] = best_k

    @torch.no_grad()
    def aggregate(self, client_states: Dict[int, Dict[str, torch.Tensor]]):
        """
        - global Φ: average embedding states from all participating clients
        - Ω_k: average full model states within each cluster among participating clients
        """
        # Φ
        embed_dicts = []
        for _cid, st in client_states.items():
            # extract embed.* keys and strip prefix
            e = {k.replace("embed.", ""): v for k, v in st.items() if k.startswith("embed.")}
            embed_dicts.append(e)
        if embed_dicts:
            self.global_phi = average_state_dict(embed_dicts)

        # Ω_k
        for k in range(self.num_clusters):
            members = [cid for cid in client_states.keys() if self.R[cid] == k]
            if not members:
                continue
            self.cluster_centers[k] = average_state_dict([client_states[cid] for cid in members])

        # (Optional) keep global_model synced to Φ + average head or something
        # LCFed is primarily cluster-personalized; global_model used as init.
        # Here we update global_model's embedding to Φ for coherence.
        # Head stays as-is (global), but you can choose another policy.
        new_global = copy.deepcopy(self.global_model.state_dict())
        for k, v in self.global_phi.items():
            new_global[f"embed.{k}"] = v.clone()
        self.global_model.load_state_dict(new_global, strict=False)

    @torch.no_grad()
    def _eval_and_log(self, round_idx: int, selected_ids: set):
        """
        Single pass over all clients that:
          - logs one row per non-selected client in their metrics.csv
            (selected clients already logged their row inside train())
          - computes mean/std accuracy across ALL clients and writes to server_metrics.csv
        """
        accs, losses = [], []
        for c in self.clients:
            k = self.R[c.client_id]
            eval_model = type(self.global_model)().to(self.device)
            eval_model.load_state_dict(self.cluster_centers[k])
            acc, _, loss = c.evaluate(eval_model)
            accs.append(acc)
            losses.append(loss)
            if c.client_id not in selected_ids:
                with open(os.path.join(self.output_dir, f"client_{c.client_id}", "metrics.csv"), "a") as f:
                    f.write(f"{round_idx},{loss},{acc},{acc},0,0\n")

        with open(os.path.join(self.output_dir, "server_metrics.csv"), "a") as f:
            f.write(f"{round_idx},{np.mean(accs):.6f},{np.std(accs):.6f},{np.mean(losses):.6f}\n")

    def select_clients(self, clients_subset: Optional[List[ClientLCFed]] = None) -> List[ClientLCFed]:
        if clients_subset is None:
            clients_subset = self.clients
        m = max(1, int(self.fraction * len(clients_subset)))
        return list(self.rng.sample(clients_subset, m))

    def evaluate(self, model: nn.Module, dataloader: DataLoader, k: int = 1, return_loss: bool = False):
        model.eval()
        correct_top1 = 0
        correct_topk = 0
        total = 0
        last_loss = 0.0
        loss_fn = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                last_loss = float(loss.item())

                _, pred_top1 = outputs.max(dim=1)
                correct_top1 += (pred_top1 == targets).sum().item()

                _, pred_topk = outputs.topk(k, dim=1)
                correct_topk += pred_topk.eq(targets.view(-1, 1)).sum().item()

                total += targets.size(0)

        acc_top1 = correct_top1 / max(1, total)
        acc_topk = correct_topk / max(1, total)
        return (acc_top1, acc_topk, last_loss) if return_loss else (acc_top1, acc_topk)

    def train(self, verbose: bool = True):
        """
        LCFed training loop (online clustering + dual aggregation).
        Returns:
            - cluster_centers (Ω_k)
            - global_phi (Φ)
            - assignment vector R
        """
        if not self.clients:
            raise RuntimeError("ServerLCFed: set_clients must be called first.")

        start = time.time()
        self.compute_M_and_broadcast()

        for r in range(self.rounds):
            selected = self.select_clients()
            selected_ids = {c.client_id for c in selected}

            # On entraîne uniquement les clients sélectionnés
            client_states: Dict[int, Dict[str, torch.Tensor]] = {}
            client_z: Dict[int, torch.Tensor] = {}
            losses: List[float] = []

            for c in selected:
                cid = c.client_id
                k_star = self.R[cid]
                omega_k = self.cluster_centers[k_star]
                phi = self.global_phi

                st, z, loss, _energy = c.train(
                    global_model=self.global_model,
                    phi_global=phi,
                    omega_cluster=omega_k,
                    round=r,
                    verbose=False
                )

                client_states[cid] = st
                client_z[cid] = z
                losses.append(loss)

            # Online reassignment (uniquement sur selected)
            self.update_assignments(client_z)

            # Aggregate Φ and Ω_k (uniquement sur selected)
            self.aggregate(client_states)

            # Evaluate all clients, log per-client CSV + server_metrics.csv
            self._eval_and_log(r, selected_ids=selected_ids)

            # Optional server-side eval (global_model as reference)
            if verbose and ((r + 1) % self.args.get("log_every", 10) == 0):
                mean_loss = float(np.mean(losses)) if losses else 0.0
                g_acc1, _ = self.evaluate(self.global_model, self.test_dataloader, k=1, return_loss=False)

                counts = [0] * self.num_clusters
                for c in self.clients:
                    counts[self.R[c.client_id]] += 1

                print(
                    f"[Round {r+1:04d}] "
                    f"selected={len(selected_ids)}/{len(self.clients)} "
                    f"mean loss(sel)={mean_loss:.4f} "
                    f"global acc@1={g_acc1:.4f} "
                    f"cluster_sizes={counts}"
                )

        if verbose:
            dur = time.time() - start
            print(f"LCFed training done in {dur:.1f}s")

        return self.cluster_centers, self.global_phi, self.R
