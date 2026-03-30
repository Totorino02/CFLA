import copy
import os
import time
from typing import Optional

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from torch import nn
from torch.utils.data import DataLoader

from framework.client.client_hcfl import ClientHCFL
from framework.common.utils import average_state_dict
from framework.server.serverbase import Server


class ServerHCFL(Server):
    def __init__(self, global_model, test_dataloader, args, **kwargs):
        super().__init__()
        self.global_model = global_model.to(args["device"])
        self.test_dataloader = test_dataloader
        self.fraction = args["fraction"]
        self.device = args["device"]
        self.initial_rounds = args["initial_rounds"]
        self.cluster_rounds = args["cluster_rounds"]
        self.distance_threshold = args.get("distance_threshold", 0.5)
        self.clustering_metric = args.get("clustering_metric", "cosine")
        self.n_clusters = args.get("n_clusters", None)  # if set, overrides distance_threshold
        # λ(t) = λ₀ / (1 + α·t)^p  — contrôle le mélange Ω_k ← (1-λ)·Avg + λ·Φ
        self.lambda_0 = float(args.get("lambda_0", 1.0))
        self.lambda_alpha = float(args.get("lambda_alpha", 0.1))
        self.lambda_p = float(args.get("lambda_p", 1.0))
        self.clusters = None
        self.clients: list[ClientHCFL] = []
        self.selected_clients: list[ClientHCFL] = []
        self.history = []
        self.specialized_models = dict()
        self.output_dir = args["output_dir"]
        self.args = args
        self.seed = args.get("seed", 0)
        self.rng = np.random.default_rng(self.seed)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # State
        self.R: list[
            int
        ] = []  # cluster assignment per client index (must align with client.client_id)
        self.cluster_centers: list[dict[str, torch.Tensor]] = []
        self.global_phi: dict[str, torch.Tensor] = {}
        self.M: Optional[torch.Tensor] = None  # [P, D]

        # optional metrics
        self.history = []

    def set_clients(self, clients: list[ClientHCFL]):
        self.clients = clients
        for c in self.clients:
            c.output_dir = self.output_dir

        # Initialize R with -1 (unassigned) for all clients
        self.R = [-1] * len(clients)

        # Initialize server metrics CSV
        with open(os.path.join(self.output_dir, "server_metrics.csv"), "w") as f:
            f.write("round,mean_acc,std_acc,mean_loss\n")

    @torch.no_grad()
    def aggregate(self, client_states: dict[int, dict[str, torch.Tensor]], round: int = 0):
        """
        - Φ ← Σ n_i·φ_i / Σ n_i  (moyenne pondérée des embeddings)
        - Ω_k ← (1-λ(t))·Avg({ω_i : i ∈ S_t ∩ C_k}) + λ(t)·Φ(t)
        """
        # λ(t) = λ₀ / (1 + α·t)^p
        lam = self.lambda_0 / (1.0 + self.lambda_alpha * round) ** self.lambda_p

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

        # Ω_k ← (1-λ)·Avg(ω_i pour i ∈ S_t ∩ C_k) + λ·Φ
        for k in range(self.num_clusters):
            members = [cid for cid in client_states.keys() if self.R[cid] == k]
            if not members:
                continue
            avg_k = average_state_dict([client_states[cid] for cid in members])
            # Mélange : parties embedding tirées vers Φ, reste inchangé
            blended = {}
            for param_key, avg_val in avg_k.items():
                if param_key.startswith("embed."):
                    phi_key = param_key[len("embed.") :]
                    blended[param_key] = (1.0 - lam) * avg_val + lam * self.global_phi[phi_key].to(
                        avg_val.device
                    )
                else:
                    blended[param_key] = avg_val
            self.cluster_centers[k] = blended

        # (Optional) keep global_model synced to Φ + average head or something
        # LCFed is primarily cluster-personalized; global_model used as init.
        # Here we update global_model's embedding to Φ for coherence.
        # Head stays as-is (global), but you can choose another policy.
        new_global = copy.deepcopy(self.global_model.state_dict())
        for k, v in self.global_phi.items():
            new_global[f"embed.{k}"] = v.clone()
        self.global_model.load_state_dict(new_global, strict=False)

    @torch.no_grad()
    def _omega_phi_dist(self, k: int) -> float:
        """‖Ω_k - Φ‖₂ sur les paramètres embedding uniquement."""
        dist = 0.0
        for param_key, omega_val in self.cluster_centers[k].items():
            if param_key.startswith("embed."):
                phi_key = param_key[len("embed.") :]
                if phi_key in self.global_phi:
                    dist += (omega_val.cpu() - self.global_phi[phi_key].cpu()).pow(2).sum().item()
        return dist**0.5

    @torch.no_grad()
    def _eval_and_log(self, round_idx: int, selected_ids: set):
        """
        - logs one row per non-selected client in their metrics.csv
        - computes mean/std accuracy across ALL clients → server_metrics.csv
        - computes per-cluster accuracy + ‖Ω_k - Φ‖ → cluster_metrics.csv
        """
        # Collect per-client and per-cluster accuracy
        cluster_accs: dict[int, list[float]] = {k: [] for k in range(self.num_clusters)}
        accs, losses = [], []

        for c in self.clients:
            k = self.R[c.client_id]
            eval_model = type(self.global_model)().to(self.device)
            eval_model.load_state_dict(self.cluster_centers[k])
            acc, _, loss = c.evaluate(eval_model)
            accs.append(acc)
            losses.append(loss)
            cluster_accs[k].append(acc)
            if c.client_id not in selected_ids:
                with open(
                    os.path.join(self.output_dir, f"client_{c.client_id}", "metrics.csv"), "a"
                ) as f:
                    f.write(f"{round_idx},{loss},{acc},{acc},0,0\n")

        with open(os.path.join(self.output_dir, "server_metrics.csv"), "a") as f:
            f.write(f"{round_idx},{np.mean(accs):.6f},{np.std(accs):.6f},{np.mean(losses):.6f}\n")

        # λ courant pour ce round
        lam = self.lambda_0 / (1.0 + self.lambda_alpha * round_idx) ** self.lambda_p

        with open(os.path.join(self.output_dir, "cluster_metrics.csv"), "a") as f:
            for k in range(self.num_clusters):
                n_k = len(self.clusters[k])
                mean_acc_k = float(np.mean(cluster_accs[k])) if cluster_accs[k] else 0.0
                dist_k = self._omega_phi_dist(k)
                f.write(f"{round_idx},{k},{n_k},{mean_acc_k:.6f},{dist_k:.6f},{lam:.6f}\n")

    def select_clients(self, clients_subset: Optional[list[ClientHCFL]] = None) -> list[ClientHCFL]:
        """Global selection — used during pre-learning (no clusters yet)."""
        if clients_subset is None:
            clients_subset = self.clients
        m = max(1, int(self.fraction * len(clients_subset)))
        return list(self.rng.choice(clients_subset, m, replace=False))

    def select_clients_per_cluster(self) -> list[ClientHCFL]:
        """
        Select `fraction` of clients independently within each cluster.
        Guarantees every cluster is represented at every round.
        """
        selected = []
        for cluster_clients in self.clusters.values():
            m = max(1, int(self.fraction * len(cluster_clients)))
            selected.extend(self.rng.choice(cluster_clients, m, replace=False))
        return selected

    def evaluate(
        self, model: nn.Module, dataloader: DataLoader, k: int = 1, return_loss: bool = False
    ):
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
                metric="euclidean",
                linkage="ward",
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric=self.clustering_metric,
                linkage="average",
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
            print(
                f"Cluster {cluster_id} [{len(clients)}]: {[client.client_id for client in clients]}"
            )

        self.num_clusters = len(self.clusters)

        # init centers + global phi
        self.cluster_centers = [
            copy.deepcopy(self.global_model.state_dict()) for _ in range(self.num_clusters)
        ]
        self.global_phi = copy.deepcopy(self.global_model.embed.state_dict())

        # assigne R[cid] based on cluster membership
        for cluster_id, clients in self.clusters.items():
            for client in clients:
                self.R[client.client_id] = cluster_id

        # cluster_metrics.csv — per-cluster accuracy + divergence Ω_k ↔ Φ
        with open(os.path.join(self.output_dir, "cluster_metrics.csv"), "w") as f:
            f.write("round,cluster,n_clients,mean_acc,omega_phi_dist,lambda\n")

        for r in range(self.cluster_rounds):
            selected = self.select_clients_per_cluster()

            client_states: dict[int, dict[str, torch.Tensor]] = {}
            losses: list[float] = []

            for c in selected:
                cid = c.client_id
                k_star = self.R[cid]
                omega_k = self.cluster_centers[k_star]

                st, loss, _energy = c.train(
                    global_model=self.global_model,
                    omega_cluster=omega_k,
                    verbose=False,
                    save_metrics=True,
                    round=r,
                )

                client_states[cid] = st
                losses.append(loss)

            # Aggregate Φ and Ω_k (uniquement sur selected)
            self.aggregate(client_states, round=r)

            # Evaluate all clients, log per-client CSV + server_metrics.csv
            self._eval_and_log(r, selected_ids={c.client_id for c in selected})

            # Optional server-side eval (global_model as reference)
            if verbose and ((r + 1) % self.args.get("log_every", 10) == 0):
                mean_loss = float(np.mean(losses)) if losses else 0.0
                g_acc1, _ = self.evaluate(
                    self.global_model, self.test_dataloader, k=1, return_loss=False
                )

                counts = [0] * self.num_clusters
                for c in self.clients:
                    counts[self.R[c.client_id]] += 1

                print(
                    f"[Round {r + 1:04d}] "
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
