from typing import Optional

import torch
from torch import nn


def convert_seconds_to_hhmmss(seconds):
    return (
        str(seconds // 3600) + ":" + str((seconds % 3600) // 60) + ":" + str(round(seconds % 60, 2))
    )


# -------------------------
# Helpers: flatten, average, etc.
# -------------------------


@torch.no_grad()
def flatten_params(module: nn.Module, device: Optional[torch.device] = None) -> torch.Tensor:
    vecs = []
    for p in module.parameters():
        t = p.detach()
        if device is not None:
            t = t.to(device)
        vecs.append(t.reshape(-1))
    return torch.cat(vecs, dim=0)


@torch.no_grad()
def average_state_dict(dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out = {k: v.detach().clone() for k, v in dicts[0].items()}
    for d in dicts[1:]:
        for k in out:
            out[k].add_(d[k])
    for k in out:
        out[k].mul_(1.0 / len(dicts))
    return out


@torch.no_grad()
def pca_projection_matrix(W: torch.Tensor, D: int) -> torch.Tensor:
    """
    W: [N, P] flattened full-model parameters
    returns M: [P, D] top-D principal directions (torch.pca_lowrank)
    """
    X = W - W.mean(dim=0, keepdim=True)
    q = min(D, X.shape[0], X.shape[1])
    _, _, V = torch.pca_lowrank(X, q=q, center=False)
    return V[:, :D].contiguous()


@torch.no_grad()
def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float(torch.dot(a, b) / denom)
