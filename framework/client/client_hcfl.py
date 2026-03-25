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
from sklearn.cluster import KMeans
from torch.utils.data import random_split, DataLoader
from framework.client.clientbase import Client
from framework.common.utils import flatten_params




RAPL_ENERGY_UNITS = 1e6
NVML_NVIDIA_UNITS = 1e3

class ClientHCFL(Client):

    def __init__(self, client_id, dataset, args, output_dir, **kwargs):
        # You can inherit from your framework.client.clientbase.Client if you want;
        # kept standalone for clarity.
        self.client_id = client_id
        self.args = args
        self.device = torch.device(args["device"])
        self.output_dir = output_dir

        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.local_epochs = args["local_epochs"]
        self.learning_rate = args["learning_rate"]
        self.batch_size = args["batch_size"]
        self.train_fraction = args.get("train_fraction", 0.8)

        # LCFed-specific
        self.mu = float(args.get("mu", 2.0))     # cluster reg
        self.lam = float(args.get("lambda", 1.0)) # global embedding reg
        self.local_steps = int(args.get("local_steps", 0))  # if >0, overrides epochs loops
        self.monitor_energy = args.get("monitor_energy", False)

        # Split train/test
        train_size = int(len(dataset) * self.train_fraction)
        test_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(args.get("seed", 0) + client_id)
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Local model
        self.local_model: nn.Module = None

        # Low-rank projection matrix M (P x D) broadcast by server
        self.M: Optional[torch.Tensor] = None

        # IO
        os.makedirs(os.path.join(self.output_dir, f"client_{self.client_id}"), exist_ok=True)
        with open(os.path.join(self.output_dir, f"client_{self.client_id}", "metrics.csv"), "w") as f:
            f.write("round,loss,accuracy_before,accuracy_after,energy_consumed,energy_ratio\n")

    def set_M(self, M: torch.Tensor):
        self.M = M.detach().to(self.device)

    def evaluate(self, model: Optional[nn.Module] = None):
        if model is None:
            model = self.local_model
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
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
    def get_full_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.local_model.state_dict().items()}

    @torch.no_grad()
    def get_embed_state(self) -> Dict[str, torch.Tensor]:
        # IMPORTANT: returned without "embed." prefix (server keeps Φ as embed-only state_dict)
        return {k: v.detach().cpu() for k, v in self.local_model.embed.state_dict().items()}

    def train(
        self,
        global_model: nn.Module,
        phi_global: Dict[str, torch.Tensor] = None,
        omega_cluster: Dict[str, torch.Tensor] = None,
        verbose: bool = False,
        save_metrics: bool = True,
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float, dict]:
        """
        This method trains the local model on the local dataset.
        if phi_global and omega_cluster are provided, it performs the LCFed training with the additional regularization terms.
        if not, it performs standard local training (can be used for pre-learning rounds before clustering).
        Returns:
            - full local model state_dict (for server aggregation)
            - low-rank vector for clustering (z = M @ flatten(local_model))
            - last training loss (for logging)
            - energy consumed (optional, for logging)
        """
        # init local model from global model parameters (ω init)
        self.local_model = type(global_model)().to(self.device)
        self.local_model.load_state_dict(global_model.state_dict(), strict=True)
        self.local_model.train()

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.learning_rate)

        acc_before, _, _ = self.evaluate(self.local_model)

        # Energy monitoring (Linux/RAPL only)
        energy_consumed = {}
        energy_monitor = None
        if self.monitor_energy:
            from declearn.main.utils._energy_monitor import EnergyMonitor  # type: ignore
            energy_monitor = EnergyMonitor()
            energy_monitor.start()

        if phi_global is None or omega_cluster is None:

            # Standard local training (no clustering regularization)
            for epoch in range(self.local_epochs):
                for data, target in self.train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    out = self.local_model(data)
                    loss = self.criterion(out, target)
                    loss.backward()
                    optimizer.step()

            if energy_monitor is not None:
                energy_consumed = energy_monitor.stop()

            loss_val = float(loss.item())
            acc_after, _, _ = self.evaluate(self.local_model)

            if save_metrics:
                with open(os.path.join(self.output_dir, f"client_{self.client_id}", "metrics.csv"), "a") as f:
                    e_pkg0 = energy_consumed.get("package_0", 0) if energy_consumed else 0
                    f.write(f"{kwargs.get('round',0)},{loss_val},{acc_before},{acc_after},{e_pkg0},0.0\n")

            return self.get_full_state(), loss_val, energy_consumed


        # Prepare frozen refs Ω_{k*} and Φ on device
        omega_ref = type(global_model)().to(self.device)
        omega_ref.load_state_dict(omega_cluster, strict=True)
        omega_ref.eval()

        phi_ref = copy.deepcopy(self.local_model.embed).to(self.device)
        phi_ref.load_state_dict(phi_global, strict=True)
        phi_ref.eval()

        loss_val = 0.0

        # Training loops:
        # - if local_steps > 0: run fixed number of mini-batch updates (closer to some FL setups)
        # - else: epochs over dataloader
        if self.local_steps and self.local_steps > 0:
            it = iter(self.train_loader)
            for _ in range(self.local_steps):
                try:
                    data, target = next(it)
                except StopIteration:
                    it = iter(self.train_loader)
                    data, target = next(it)

                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                out = self.local_model(data)
                loss_sup = self.criterion(out, target)

                # μ/2 ||ω - Ω||^2 over all params
                l2_cluster = torch.zeros((), device=self.device)
                for p, q in zip(self.local_model.parameters(), omega_ref.parameters()):
                    l2_cluster = l2_cluster + (p - q).pow(2).sum()

                # λ/2 ||ϕ - Φ||^2 only over embedding params
                l2_global = torch.zeros((), device=self.device)
                for p, q in zip(self.local_model.embed.parameters(), phi_ref.parameters()):
                    l2_global = l2_global + (p - q).pow(2).sum()

                loss = loss_sup + 0.5 * self.mu * l2_cluster + 0.5 * self.lam * l2_global
                loss.backward()
                optimizer.step()
                loss_val = float(loss.item())
        else:
            for _epoch in range(self.local_epochs):
                for data, target in self.train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    out = self.local_model(data)
                    loss_sup = self.criterion(out, target)

                    l2_cluster = torch.zeros((), device=self.device)
                    for p, q in zip(self.local_model.parameters(), omega_ref.parameters()):
                        l2_cluster = l2_cluster + (p - q).pow(2).sum()

                    l2_global = torch.zeros((), device=self.device)
                    for p, q in zip(self.local_model.embed.parameters(), phi_ref.parameters()):
                        l2_global = l2_global + (p - q).pow(2).sum()

                    loss = loss_sup + 0.5 * self.mu * l2_cluster + 0.5 * self.lam * l2_global
                    loss.backward()
                    optimizer.step()
                    loss_val = float(loss.item())

        if energy_monitor is not None:
            energy_consumed = energy_monitor.stop()

        acc_after, _, _ = self.evaluate(self.local_model)
        
        vec = flatten_params(self.local_model, device=self.device)

        # Energy ratio (optional, similar to your FLHC)
        energy_ratio = 0.0
        if energy_consumed:
            client_energy = 0.0
            for k, e in energy_consumed.items():
                if e < 0:
                    e = 0
                if str(k).startswith("nvidia"):
                    e = e / NVML_NVIDIA_UNITS
                else:
                    e = e / RAPL_ENERGY_UNITS
                client_energy += e
            # use ||vec|| as proxy norm (or update norm if you compute update vector)
            denom = float(vec.norm().item()) + 1e-9
            energy_ratio = client_energy / denom

        if save_metrics:
            with open(os.path.join(self.output_dir, f"client_{self.client_id}", "metrics.csv"), "a") as f:
                e_pkg0 = energy_consumed.get("package_0", 0) if energy_consumed else 0
                f.write(f"{kwargs.get('round',0)},{loss_val},{acc_before},{acc_after},{e_pkg0},{energy_ratio}\n")

        if verbose:
            print(
                f"Client {self.client_id} | Round {kwargs.get('round',0)} | "
                f"Loss {loss_val:.4f} | Acc {acc_before:.4f}->{acc_after:.4f}"
            )

        return self.get_full_state(), loss_val, energy_consumed