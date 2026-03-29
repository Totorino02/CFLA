from collections import defaultdict
from datetime import datetime
import os
from sklearn.cluster import AgglomerativeClustering
import torch
import time
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Optional
import copy
from torch import nn
from torch.utils.data import DataLoader

from framework.server.serverbase import Server
from framework.client.client_hcfl import ClientHCFL
from framework.common.utils import flatten_params, average_state_dict


class ServerHCFL(Server):

    def __init__(self, global_model, test_dataloader, args, **kwargs):
        super().__init__()
        self.global_model = global_model.to(args["device"])
        self.test_dataloader = test_dataloader
        self.fraction = args["fraction"]
        self.device = args["device"]
        self.initial_rounds = args["initial_rounds"]
        self.cluster_rounds = args["cluster_rounds"]
        self.distance_threshold = args["distance_threshold"]
        self.clustering_metric = args["clustering_metric"]
        self.n_clusters = args.get("n_clusters", None)   # if set, overrides distance_threshold
        self.clusters = None
        self.clients : list[ClientHCFL]= []
        self.selected_clients : list[ClientHCFL] = []
        self.history = []
        self.specialized_models = dict()
        self.output_dir = args["output_dir"]
        self.args = args
        self.seed = args.get("seed", 0)
        self.rng = np.random.default_rng(self.seed)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # State
        self.R: List[int] = []  # cluster assignment per client index (must align with client.client_id)
        self.cluster_centers: List[Dict[str, torch.Tensor]] = []
        self.global_phi: Dict[str, torch.Tensor] = {}
        self.M: Optional[torch.Tensor] = None  # [P, D]

        # optional metrics
        self.history = []

    def set_clients(self, clients: List[ClientHCFL]):
        self.clients = clients
        for c in self.clients:
            c.output_dir = self.output_dir

        # Initialize R with -1 (unassigned) for all clients
        self.R = [-1] * len(clients)

        # Initialize server metrics CSV
        with open(os.path.join(self.output_dir, "server_metrics.csv"), "w") as f:
            f.write("round,mean_acc,std_acc,mean_loss\n")

    @torch.no_grad()
    def aggregate(self, client_states: Dict[int, Dict[str, torch.Tensor]]):
        """
        - global Φ: average embedding states from all participating clients
        - Ω_k: average full model states within each cluster among participating clients
        """
        # Φ ← Σ n_i·φ_i / Σ n_i  (weighted by local dataset size)
        client_size = {c.client_id: len(c.train_loader.dataset) for c in self.clients}
        embed_dicts, weights = [], []
        for cid, st in client_states.items():
            e = {k.replace("embed.", ""): v for k, v in st.items() if k.startswith("embed.")}
            embed_dicts.append(e)
            weights.append(client_size[cid])
        if embed_dicts:
            total = sum(weights)
            self.global_phi = {
                k: sum(w * d[k] for d, w in zip(embed_dicts, weights)) / total
                for k in embed_dicts[0]
            }

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

    def select_clients(self, clients_subset: Optional[List[ClientHCFL]] = None) -> List[ClientHCFL]:
        """Global selection — used during pre-learning (no clusters yet)."""
        if clients_subset is None:
            clients_subset = self.clients
        m = max(1, int(self.fraction * len(clients_subset)))
        return list(self.rng.choice(clients_subset, m, replace=False))

    def select_clients_per_cluster(self) -> List[ClientHCFL]:
        """
        Select `fraction` of clients independently within each cluster.
        Guarantees every cluster is represented at every round.
        """
        selected = []
        for cluster_clients in self.clusters.values():
            m = max(1, int(self.fraction * len(cluster_clients)))
            selected.extend(self.rng.choice(cluster_clients, m, replace=False))
        return selected

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

    def pre_learning(self):
        """
        Runs initial_rounds of FedAvg to warm up the global model, then collects
        embedding update vectors from all clients for hierarchical clustering.
        """
        # Warm-up rounds: standard FedAvg to get a meaningful global representation
        for epoch in range(self.initial_rounds):
            selected = self.select_clients()
            client_states = []
            for client in selected:
                state, _, _ = client.train(self.global_model, round=epoch, save_metrics=False)
                client_states.append(state)
            # Aggregate selected clients into global model
            averaged = average_state_dict(client_states)
            self.global_model.load_state_dict(averaged)

        # Collect embedding update vectors from ALL clients after warm-up
        updates = []
        for client in self.clients:
            _, _, _ = client.train(self.global_model, round=self.initial_rounds, save_metrics=False)
            update_vector = []
            for new_param, old_param in zip(
                client.local_model.embed.parameters(),
                self.global_model.embed.parameters(),
            ):
                update_vector.append((new_param.data - old_param.data).flatten())
            updates.append(torch.cat(update_vector).cpu().numpy())
        return np.array(updates)
    
    def hierarchical_clustering(self, updates):
        if self.n_clusters is not None:
            clustering = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                metric='euclidean',
                linkage='ward',
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric=self.clustering_metric,
                linkage='average',
            )

        labels = clustering.fit_predict(updates)
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(self.clients[idx])
        return clusters

    def train(self, verbose=False, **kwargs):
        """
        This method performs the training of the global model and the clustering of the clients.
        :param global_model: The global model to be trained
        :return: None
        """

        start = time.time()
        # Perform federated learning on all clients to get their update vectors
        updates = self.pre_learning()

        # Perform hierarchical clustering on the update vectors to get the clusters
        self.clusters = self.hierarchical_clustering(updates)

        print(f"Clusters formed: {len(self.clusters)}")
        for cluster_id, clients in self.clusters.items():
            print(f"Cluster {cluster_id} [{len(clients)}]: {[client.client_id for client in clients]}")

        self.num_clusters = len(self.clusters)

        # init centers + global phi
        self.cluster_centers = [copy.deepcopy(self.global_model.state_dict()) for _ in range(self.num_clusters)]
        self.global_phi = copy.deepcopy(self.global_model.embed.state_dict())

        # assigne R[cid] based on cluster membership
        for cluster_id, clients in self.clusters.items():
            for client in clients:
                self.R[client.client_id] = cluster_id

        for r in range(self.cluster_rounds):
            selected = self.select_clients_per_cluster()

            client_states: Dict[int, Dict[str, torch.Tensor]] = {}
            losses: List[float] = []

            for c in selected:
                cid = c.client_id
                k_star = self.R[cid]
                omega_k = self.cluster_centers[k_star]
                phi = self.global_phi

                st, loss, _energy = c.train(
                    global_model=self.global_model,
                    phi_global=phi,
                    omega_cluster=omega_k,
                    round=r,
                    verbose=False,
                    save_metrics=True,
                )

                client_states[cid] = st
                losses.append(loss)

            # Aggregate Φ and Ω_k (uniquement sur selected)
            self.aggregate(client_states)

            # Evaluate all clients, log per-client CSV + server_metrics.csv
            self._eval_and_log(r, selected_ids={c.client_id for c in selected})

            # Optional server-side eval (global_model as reference)
            if verbose and ((r + 1) % self.args.get("log_every", 10) == 0):
                mean_loss = float(np.mean(losses)) if losses else 0.0
                g_acc1, _ = self.evaluate(self.global_model, self.test_dataloader, k=1, return_loss=False)

                counts = [0] * self.num_clusters
                for c in self.clients:
                    counts[self.R[c.client_id]] += 1

                print(
                    f"[Round {r+1:04d}] "
                    f"selected={len(selected)}/{len(self.clients)} "
                    f"mean loss(sel)={mean_loss:.4f} "
                    f"global acc@1={g_acc1:.4f} "
                    f"cluster_sizes={counts}"
                )

        if verbose:
            dur = time.time() - start
            print(f"$HCFL training done in {dur:.1f}s")

        return self.cluster_centers, self.global_phi, self.R
    
    def get_params(self):
        return super().get_params()

