"""
IFCA client — An Efficient Framework for Clustered Federated Learning
(Ghosh et al., NeurIPS 2020)

Each round the client receives ALL K cluster center models, evaluates
empirical loss on each, selects the cluster with the minimum loss,
trains from that model, and returns the updated state + selected cluster index.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

RAPL_ENERGY_UNITS = 1e6
NVML_NVIDIA_UNITS = 1e3


class ClientIFCA:
    """
    IFCA client:
      - Receives all K cluster center models from the server
      - Evaluates empirical loss on each: k* = argmin_k L_i(Ω_k)
      - Trains from Ω_{k*} using standard SGD (no regularization)
      - Returns (state_dict, k_star, loss, energy)
    """

    def __init__(self, client_id, dataset, args, output_dir, **kwargs):
        self.client_id = client_id
        self.args = args
        self.device = torch.device(args["device"])
        self.output_dir = output_dir

        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.local_epochs = args["local_epochs"]
        self.learning_rate = args["learning_rate"]
        self.batch_size = args["batch_size"]
        self.train_fraction = args.get("train_fraction", 0.8)
        self.monitor_energy = args.get("monitor_energy", False)

        train_size = int(len(dataset) * self.train_fraction)
        test_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(args.get("seed", 0) + client_id)
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], generator=generator
        )
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.local_model: Optional[nn.Module] = None

        os.makedirs(os.path.join(self.output_dir, f"client_{self.client_id}"), exist_ok=True)
        with open(os.path.join(self.output_dir, f"client_{self.client_id}", "metrics.csv"), "w") as f:
            f.write("round,loss,accuracy_before,accuracy_after,energy_consumed,energy_ratio\n")

    def evaluate(self, model: Optional[nn.Module] = None):
        if model is None:
            model = self.local_model
        model.eval()
        correct, total, test_loss = 0, 0, 0.0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                out = model(data)
                loss = self.criterion(out, target)
                _, pred = out.max(dim=1)
                total += target.size(0)
                correct += (pred == target).sum().item()
                test_loss = float(loss.item())
        return correct / max(1, total), correct, test_loss

    @torch.no_grad()
    def empirical_loss(self, model: nn.Module) -> float:
        """Average cross-entropy loss over the full local training set."""
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                out = model(data)
                total_loss += float(self.criterion(out, target).item())
                n_batches += 1
        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def get_full_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.local_model.state_dict().items()}

    def train(
        self,
        cluster_models: List[nn.Module],
        round: int = 0,
        verbose: bool = False,
        save_metrics: bool = True,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], int, float, dict]:
        """
        Selects best cluster by empirical loss, then trains from it.
        Returns:
          - full state_dict (cpu)
          - k_star (selected cluster index)
          - last training loss
          - energy_consumed dict
        """
        # Cluster selection: k* = argmin_k L_i(Ω_k)
        losses_per_cluster = [self.empirical_loss(m) for m in cluster_models]
        k_star = int(torch.tensor(losses_per_cluster).argmin().item())

        # Init from best cluster center
        best_model = cluster_models[k_star]
        self.local_model = type(best_model)().to(self.device)
        self.local_model.load_state_dict(best_model.state_dict(), strict=True)
        self.local_model.train()
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.learning_rate)

        acc_before, _, _ = self.evaluate(self.local_model)

        energy_consumed = {}
        energy_monitor = None
        if self.monitor_energy:
            from declearn.main.utils._energy_monitor import EnergyMonitor  # type: ignore
            energy_monitor = EnergyMonitor()
            energy_monitor.start()

        loss_val = 0.0
        for _epoch in range(self.local_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                out = self.local_model(data)
                loss = self.criterion(out, target)
                loss.backward()
                optimizer.step()
                loss_val = float(loss.item())

        if energy_monitor is not None:
            energy_consumed = energy_monitor.stop()

        acc_after, _, _ = self.evaluate(self.local_model)

        energy_ratio = 0.0
        if energy_consumed:
            client_energy = 0.0
            for k, e in energy_consumed.items():
                if e < 0:
                    e = 0
                e = e / (NVML_NVIDIA_UNITS if str(k).startswith("nvidia") else RAPL_ENERGY_UNITS)
                client_energy += e
            from framework.common.utils import flatten_params
            denom = float(flatten_params(self.local_model, device=self.device).norm().item()) + 1e-9
            energy_ratio = client_energy / denom

        if save_metrics:
            with open(os.path.join(self.output_dir, f"client_{self.client_id}", "metrics.csv"), "a") as f:
                e_pkg0 = energy_consumed.get("package_0", 0) if energy_consumed else 0
                f.write(f"{round},{loss_val},{acc_before},{acc_after},{e_pkg0},{energy_ratio}\n")

        if verbose:
            print(
                f"Client {self.client_id} | Round {round} | cluster={k_star} | "
                f"Loss {loss_val:.4f} | Acc {acc_before:.4f}->{acc_after:.4f}"
            )

        return self.get_full_state(), k_star, loss_val, energy_consumed
