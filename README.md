# CFLA — Clustered Federated Learning Algorithms

A Python library implementing and benchmarking **Clustered Federated Learning (CFL)** algorithms. It includes reproductions of state-of-the-art methods from the literature alongside **HCFL**, an original algorithm developed as part of this work.

---

## Overview

**Federated Learning (FL)** enables training machine learning models on decentralized data without sharing it. Each client trains locally and only sends model updates to a central server.

**Clustered Federated Learning** extends FL by grouping similar clients and training one specialized model per cluster. This improves personalization when client data is heterogeneous (non-IID), which is the common case in practice.

CFLA provides a unified framework to implement, run, and compare CFL algorithms on standard benchmarks, with built-in support for energy consumption monitoring.

---

## Algorithms

### Baselines from the literature

| Algorithm | Clustering | Regularization | Reference |
|-----------|-----------|---------------|-----------|
| **FedAvg** | None — single global model | None | McMahan et al., AISTATS 2017 |
| **FedPer** | None — personalized head per client | None | Collins et al., ICML 2021 |
| **FLHC** | Offline — agglomerative on update vectors | None | Briggs et al., 2020 |
| **FedGroup** | Offline — agglomerative on model params (cosine) | None | Tan et al., 2022 |
| **MAD-MTOP** | Online — recursive bipartitioning on gradients | None | Sattler et al., 2019 |
| **FeSEM** | Online — L2 distance to cluster centers (EM) | None | Li et al., 2021 |
| **CGPFL** | Online — cosine similarity to cluster centers | `(μ/2)‖ω−Ω_k‖²` | Ref. [37] in LCFed paper |
| **IFCA** | Online — empirical risk (loss) per cluster | None | Ghosh et al., NeurIPS 2020 |
| **LCFed** | Online — PCA low-rank projection + cosine | `(μ/2)‖ω−Ω_k‖² + (λ/2)‖φ−Φ‖²` | Zhang et al., ICASSP 2025 |

### HCFL (original contribution)

HCFL combines **hierarchical clustering** on embedding update vectors with the **dual regularization** loss of LCFed, building on a split-model architecture (embedding `φ` + head `ω`):

```
L = L_sup + (μ/2)||ω_i − Ω_{k*}||² + (λ/2)||φ_i − Φ||²
```

- `Ω_{k*}`: center of the client's assigned cluster (pulls the full model toward its cluster)
- `Φ`: global embedding aggregated across all clients (keeps representations aligned globally)
- `μ` controls intra-cluster cohesion; `λ` controls global regularization strength

**Training procedure:**
1. **Pre-training** — short FedAvg warm-up to build a meaningful global representation
2. **Clustering** — agglomerative hierarchical clustering on embedding update vectors (cosine metric)
3. **Personalization** — per-cluster training with dual regularization; global `Φ` and per-cluster `Ω_k` are aggregated separately each round

The key idea: hierarchical clustering discovers the cluster structure automatically (no fixed K required), while the dual aggregation injects global knowledge into local training to avoid over-specialization.

---

## Project Structure

```
CFLA/
├── framework/
│   ├── client/
│   │   ├── client_flhc.py       # FLHC client
│   │   ├── client_hcfl.py       # HCFL client
│   │   ├── client_lcfed.py      # LCFed client
│   │   ├── client_fedper.py     # FedPer client
│   │   ├── client_fedgroup.py   # FedGroup client
│   │   ├── client_fesem.py      # FeSEM client
│   │   ├── client_cgpfl.py      # CGPFL client
│   │   ├── client_ifca.py       # IFCA client
│   │   └── client_madmtop.py    # MAD-MTOP client
│   ├── server/
│   │   ├── server_flhc.py
│   │   ├── server_hcfl.py
│   │   ├── serveur_lcfed.py
│   │   ├── server_fedper.py
│   │   ├── server_fedgroup.py
│   │   ├── server_fesem.py
│   │   ├── server_cgpfl.py
│   │   ├── server_ifca.py
│   │   ├── server_fedavg.py
│   │   └── server_madmtop.py
│   ├── models/
│   │   └── computer_vision.py   # LeNet5V1, SplitLeNet5V1, CNNCifar, SplitCNNCifar
│   └── common/
│       └── utils.py             # flatten_params, average_state_dict, pca_projection_matrix, cosine_sim
├── experiments/
│   └── scripts/
│       ├── main.py              # Shared dataset partitioning + experiment runner
│       ├── hcfl.py
│       ├── lcfed.py
│       ├── flhc.py
│       ├── fedper.py
│       ├── fedgroup.py
│       ├── fesem.py
│       ├── cgpfl.py
│       ├── ifca.py
│       └── plot_results.py      # Performance plots + comparison table
├── RESULTS/                     # Output directory (per-client CSVs + server_metrics.csv)
├── PLOTS/                       # Generated figures
├── pyproject.toml
└── requirements.dev.txt
```

---

## Installation

```bash
git clone https://github.com/<your-username>/CFLA.git
cd CFLA
pip install -e .
pip install -r requirements.dev.txt
```

### Energy monitoring (Linux only)

Energy tracking via RAPL (CPU) and NVML (GPU) is disabled by default (`monitor_energy: False`).
To enable it on Linux:

```bash
sudo chmod 444 /sys/class/powercap/intel-rapl:*/energy_uj
sudo chmod 444 /sys/class/powercap/intel-rapl:*:*/energy_uj
```

Then set `"monitor_energy": True` in the client args of your experiment script.

---

## Quick Start

Run a single algorithm directly:

```bash
python -m experiments.scripts.hcfl
python -m experiments.scripts.lcfed
python -m experiments.scripts.ifca
```

Or run all algorithms from `main.py`:

```python
# experiments/scripts/main.py
if __name__ == "__main__":
    base_seed = 2026
    for ds in ("mnist",):
        run_flhc_experiment(nb_runs=1, base_seed=base_seed, dataset=ds)
        run_hcfl_experiment(nb_runs=1, base_seed=base_seed, dataset=ds)
        run_lcfed_experiment(nb_runs=1, base_seed=base_seed, dataset=ds)
        run_fedper_experiment(nb_runs=1, base_seed=base_seed, dataset=ds)
        run_fedgroup_experiment(nb_runs=1, base_seed=base_seed, dataset=ds)
        run_fesem_experiment(nb_runs=1, base_seed=base_seed, dataset=ds)
        run_cgpfl_experiment(nb_runs=1, base_seed=base_seed, dataset=ds)
        run_ifca_experiment(nb_runs=1, base_seed=base_seed, dataset=ds)
```

### Generate plots and comparison table

```bash
python -m experiments.scripts.plot_results --results_dir ./RESULTS --output_dir ./PLOTS
```

Produces:
- `accuracy_curves.png` — mean accuracy per round with ±1σ band
- `loss_curves.png` — mean loss per round
- `accuracy_boxplot.png` — per-client accuracy distribution at the final round
- `convergence_speed.png` — rounds needed to reach a target accuracy
- `comparison_table.csv` — final metrics summary

---

## Data Partitioning

All experiments use a **structured non-IID partitioning**: classes are divided into contiguous groups, each group is assigned to a subset of clients. This creates natural heterogeneity across clients while remaining reproducible.

```python
# From experiments/scripts/main.py
"groups": [(0, 4, 18), (4, 7, 17), (7, 9, 15)]
# → 18 clients see only classes 0–4, 17 clients see classes 4–7, etc.
```

---

## Supported Datasets

| Dataset | Classes | Input | Clients (default) |
|---------|---------|-------|------------------|
| **MNIST** | 10 | 28×28 grayscale | 50 |
| **CIFAR-10** | 10 | 32×32 RGB | 50 |
| **CIFAR-100** | 100 | 32×32 RGB | 50 |

---

## Output Format

Each experiment writes:

```
RESULTS/result_{algo}_{dataset}_{timestamp}/
├── server_metrics.csv           # round, mean_acc, std_acc, mean_loss
└── client_{id}/
    └── metrics.csv              # round, loss, accuracy_before, accuracy_after, energy_consumed, energy_ratio
```

---

## Metrics

Each experiment tracks per client and per round:

| Metric | Description |
|--------|-------------|
| `accuracy_after` | Local test accuracy after the round's local update |
| `accuracy_before` | Local test accuracy before training (model received from server) |
| `loss` | Last mini-batch training loss |
| `energy_consumed` | CPU energy in joules (RAPL, Linux only) |
| `energy_ratio` | Energy per unit of model update norm |

`server_metrics.csv` aggregates `mean_acc`, `std_acc`, and `mean_loss` across all clients at each round — the primary metric for inter-algorithm comparison.

---

## References

- McMahan, B. et al. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS*. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)
- Sattler, F., Müller, K.-R., & Samek, W. (2019). Clustered federated learning: Model-agnostic distributed multi-task optimization under privacy constraints. [arXiv:1910.01991](https://arxiv.org/abs/1910.01991)
- Briggs, C., Fan, Z., & Andras, P. (2020). Federated learning with hierarchical clustering of local updates to improve training on non-IID data. [arXiv:2004.11791](https://arxiv.org/abs/2004.11791)
- Collins, L. et al. (2021). Exploiting shared representations for personalized federated learning. *ICML*. [arXiv:2102.07078](https://arxiv.org/abs/2102.07078)
- Ghosh, A. et al. (2020). An efficient framework for clustered federated learning. *NeurIPS*. [arXiv:2006.04088](https://arxiv.org/abs/2006.04088)
- Tan, Y. et al. (2022). Towards personalized federated learning. *IEEE TNNLS*. [arXiv:2103.00710](https://arxiv.org/abs/2103.00710)
- Zhang, Y. et al. (2025). LCFed: An efficient clustered federated learning framework for heterogeneous data. *ICASSP*. [arXiv:2501.01850](https://arxiv.org/abs/2501.01850)

---

## Author

Antoine Hounsi — [hounsi.madouvi.etu@univ-lille.fr](mailto:hounsi.madouvi.etu@univ-lille.fr)

Master's research project, Université de Lille.
