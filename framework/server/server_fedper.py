"""
FedPer server — Federated Personalization (Collins et al., ICML 2021)

Maintains a global embedding Φ. Each round:
  1. Broadcast global model (embed Φ + global head) to selected clients
  2. Receive updated embed states from clients
  3. Average embed states → new Φ
  4. Clients keep their personalized heads (never aggregated)
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
from framework.client.client_fedper import ClientFedPer
from framework.common.utils import average_state_dict


class ServerFedPer(Server):
    """
    FedPer server.
    Only the embedding sub-model ϕ is globally aggregated.
    """

    def __init__(self, global_model, test_dataloader, args, **kwargs):
        super().__init__()
        self.global_model = global_model.to(args["device"])
        self.test_dataloader = test_dataloader
        self.args = args

        self.device = torch.device(args["device"])
        self.fraction = float(args.get("fraction", 0.2))
        self.rounds = int(args.get("rounds", 100))
        self.seed = int(args.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)

        self.clients: List[ClientFedPer] = []
        self.output_dir = args["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        self.global_phi: Dict[str, torch.Tensor] = {}

    def set_clients(self, clients: List[ClientFedPer]):
        self.clients = clients
        for c in self.clients:
            c.output_dir = self.output_dir
        self.global_phi = copy.deepcopy(self.global_model.embed.state_dict())
        with open(os.path.join(self.output_dir, "server_metrics.csv"), "w") as f:
            f.write("round,mean_acc,std_acc,mean_loss\n")

    def select_clients(self) -> List[ClientFedPer]:
        m = max(1, int(self.fraction * len(self.clients)))
        return list(self.rng.choice(self.clients, m, replace=False))

    @torch.no_grad()
    def _aggregate_embeds(self, embed_dicts: List[Dict[str, torch.Tensor]]):
        """Average embed states → global Φ, then update global_model."""
        self.global_phi = average_state_dict(embed_dicts)
        new_state = copy.deepcopy(self.global_model.state_dict())
        for k, v in self.global_phi.items():
            new_state[f"embed.{k}"] = v.clone()
        self.global_model.load_state_dict(new_state)

    @torch.no_grad()
    def _eval_and_log(self, round_idx: int, selected_ids: set):
        """
        Evaluate all clients using global embed + their local head.
        Non-selected clients get a passthrough row (acc_before == acc_after).
        """
        accs, losses = [], []
        for c in self.clients:
            eval_model = c.build_personalized_model(self.global_model)
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
        correct_top1 = 0
        total = 0
        total_loss = 0.0
        loss_fn = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                _, pred = outputs.max(dim=1)
                correct_top1 += (pred == targets).sum().item()
                total += targets.size(0)
                total_loss += loss.item() * targets.size(0)
        acc = correct_top1 / max(1, total)
        avg_loss = total_loss / max(1, total)
        return (acc, avg_loss) if return_loss else acc

    def train(self, verbose: bool = True):
        start = time.time()

        for r in range(self.rounds):
            selected = self.select_clients()
            selected_ids = {c.client_id for c in selected}

            embed_dicts, losses = [], []
            for c in selected:
                embed_state, loss, _ = c.train(
                    self.global_model, round=r, verbose=False
                )
                embed_dicts.append(embed_state)
                losses.append(loss)

            self._aggregate_embeds(embed_dicts)
            self._eval_and_log(r, selected_ids)

            if verbose and ((r + 1) % self.args.get("log_every", 10) == 0):
                mean_loss = float(np.mean(losses)) if losses else 0.0
                g_acc = self.evaluate(self.global_model, self.test_dataloader)
                print(
                    f"[Round {r+1:04d}] "
                    f"selected={len(selected_ids)}/{len(self.clients)} "
                    f"mean loss(sel)={mean_loss:.4f} "
                    f"global acc@1={g_acc:.4f}"
                )

        if verbose:
            print(f"FedPer training done in {time.time() - start:.1f}s")

    def aggregate(self, **kwargs):
        pass  # aggregation handled by _aggregate_embeds inside train()

    def get_params(self):
        return self.global_model.state_dict()
