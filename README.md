# CFLA — Clustered Federated Learning Algorithms

A Python library implementing and benchmarking **Clustered Federated Learning (CFL)** algorithms. It includes reproductions of state-of-the-art methods from the literature alongside **HCFL**, an original algorithm that automatically discovers the number of clusters without requiring it as a hyperparameter.

> **Paper:** *HCFL: Hierarchical Clustered Federated Learning with Automatic Cluster Discovery* — Antoine Hounsi.

---

## Overview

**Federated Learning (FL)** enables training machine learning models on decentralized data without sharing it. Each client trains locally and only sends model updates to a central server.

**Clustered Federated Learning** extends FL by grouping similar clients and training one specialized model per cluster. This improves personalization when client data is heterogeneous (non-IID), which is the common case in practice.

CFLA provides a unified framework to implement, run, and compare CFL algorithms on standard benchmarks, with built-in support for energy consumption monitoring.

---

## Algorithms

### Baselines from the literature

| Algorithm | Clustering strategy | Regularization | Reference |
|-----------|-------------------|----------------|-----------|
| **FedAvg** | None — single global model | None | McMahan et al., AISTATS 2017 |
| **FLHC** | Offline — agglomerative on gradient update vectors | None | Briggs et al., 2020 |
| **FedGroup** | Offline — agglomerative on model params (cosine) | None | Tan et al., 2022 |
| **FeSEM** | Online — L2 distance to cluster centers (EM) | None | Li et al., 2021 |
| **CGPFL** | Online — cosine similarity to cluster centers | `(μ/2)‖ω−Ω_k‖²` | Liu et al. |
| **IFCA** | Online — empirical risk (loss) per cluster | None | Ghosh et al., NeurIPS 2020 |
| **LCFed** | Online — PCA low-rank projection + cosine | `(μ/2)‖ω−Ω_k‖² + (λ/2)‖φ−Φ‖²` | Zhang et al., ICASSP 2025 |

### HCFL (original contribution)

HCFL automatically discovers the number of clusters $K^*$ via agglomerative hierarchical clustering on client embedding update vectors, then trains cluster-specific models using a FedProx-style objective toward the cluster center:

```
L_i(ω) = L_sup(ω; D_i) + (μ/2) ‖ω - Ω_k‖²
```

Inter-cluster knowledge sharing is handled **server-side** via a scheduled blending of cluster models toward the global embedding Φ:

```
Ω_k(t) ← (1 - λ(t)) · Avg({ω_i : i ∈ S_t ∩ C_k}) + λ(t) · Φ(t)
λ(t)    = λ₀ / (1 + α·t)^p
```

This decouples two roles: **μ** controls client-to-cluster fidelity (local), **λ(t)** controls inter-cluster sharing (server-side, decaying).

**Training procedure:**
1. **Pre-training** (`R_pre` rounds) — FedAvg warm-up to build a meaningful global representation
2. **Cluster discovery** — each client computes δᵢ = φᵢ - φ⁽⁰⁾ after one local step; agglomerative clustering on {δᵢ} determines K* automatically
3. **CFL phase** (`T` rounds) — per-cluster training with server-side embedding blending; clusters specialize progressively as λ(t) → 0

---

## Results

Evaluated on MNIST, FEMNIST, and AG News with N=50 clients.

| Method | MNIST Acc ± Std | FEMNIST Acc ± Std | AG News Acc ± Std |
|--------|----------------|-------------------|-------------------|
| IFCA† | 97.3 ± 0.5 | 11.4 ± 15.5 | 81.5 ± 19.2 |
| FLHC† | 97.1 ± 0.4 | **35.2 ± 15.1** | 79.1 ± 22.6 |
| LCFed† | 82.4 ± 10.5 | 6.9 ± 16.3 | 27.4 ± 30.5 |
| FeSEM | 90.4 ± 12.2 | 12.8 ± 19.2 | 79.4 ± 22.3 |
| FedGroup | 91.5 ± 2.1 | 5.2 ± 11.2 | 23.0 ± 25.6 |
| CGPFL | 50.7 ± 7.9 | 12.7 ± 20.5 | 27.9 ± 26.9 |
| **HCFL (ours)** | **94.3 ± 0.7** | **32.3 ± 16.7** | **82.0 ± 18.3** |

†: requires K a priori. **Bold**: best no-K method. **Bold + best overall**: AG News.

HCFL is the best no-K method on all three benchmarks, and achieves the best overall accuracy on AG News — surpassing oracle-K methods without any prior knowledge of K.

---

## Project Structure

```
CFLA/
├── framework/
│   ├── client/
│   │   ├── clientbase.py        # Abstract Client base class
│   │   ├── client_hcfl.py       # HCFL client
│   │   ├── client_flhc.py       # FLHC client
│   │   ├── client_lcfed.py      # LCFed client
│   │   ├── client_fesem.py      # FeSEM client
│   │   ├── client_cgpfl.py      # CGPFL client
│   │   └── client_ifca.py       # IFCA client
│   ├── server/
│   │   ├── serverbase.py        # Abstract Server base class
│   │   ├── server_hcfl.py       # HCFL server
│   │   ├── server_flhc.py       # FLHC server
│   │   ├── serveur_lcfed.py     # LCFed server
│   │   ├── server_fesem.py      # FeSEM server
│   │   ├── server_cgpfl.py      # CGPFL server
│   │   └── server_ifca.py       # IFCA server
│   ├── models/
│   │   ├── computer_vision.py   # LeNet-5, SplitLeNet5, CNN variants
│   │   └── nlp_models.py        # DistilBERT-based text encoder
│   └── common/
│       └── utils.py             # flatten_params, average_state_dict, cosine_sim
├── datasets/
│   ├── femnist.py               # FEMNIST loader + Dirichlet partitioning
│   └── ag_news.py               # AG News loader + Dirichlet partitioning
├── experiments/
│   └── scripts/
│       ├── run_all_mnist.py     # Run all algorithms on MNIST
│       ├── run_all_cifar-10.py  # Run all algorithms on CIFAR-10
│       ├── run_all_femnist.py   # Run all algorithms on FEMNIST
│       ├── run_all_agnews.py    # Run all algorithms on AG News
│       ├── hcfl.py
│       ├── lcfed.py
│       ├── flhc.py
│       ├── fedgroup.py
│       ├── fesem.py
│       ├── cgpfl.py
│       ├── ifca.py
│       └── plot_results.py      # Performance plots + comparison table
├── pyproject.toml
└── requirements.dev.txt
```

---

## Installation

```bash
git clone https://github.com/Totorino02/CFLA.git
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

Run HCFL on MNIST:

```bash
python -m experiments.scripts.hcfl
```

Run all algorithms on a specific dataset:

```bash
python -m experiments.scripts.run_all_mnist
python -m experiments.scripts.run_all_femnist
python -m experiments.scripts.run_all_cifar-10
python -m experiments.scripts.run_all_agnews
```

### Generate plots and comparison table

```bash
python -m experiments.scripts.plot_results --results_dir ./RESULTS/my_run --output_dir ./PLOTS/my_run
```

Produces:
- `accuracy_curves.png` — mean accuracy per round with ±1σ band
- `loss_curves.png` — mean loss per round
- `accuracy_boxplot.png` — per-client accuracy distribution at the final round
- `convergence_speed.png` — rounds needed to reach a target accuracy
- `comparison_table.csv` — final metrics summary

---

## Supported Datasets

| Dataset | Classes | Input | Partitioning |
|---------|---------|-------|--------------|
| **MNIST** | 10 | 28×28 grayscale | Structured non-IID (disjoint class groups) |
| **CIFAR-10** | 10 | 32×32 RGB | Structured non-IID |
| **FEMNIST** | 62 | 28×28 grayscale | Dirichlet(α) |
| **AG News** | 4 | Text | Dirichlet(α) |

---

## Output Format

Each experiment writes results to a local `RESULTS/` directory (not tracked in git):

```
result_{algo}_{dataset}_{timestamp}/
├── server_metrics.csv           # round, mean_acc, std_acc, mean_loss
└── client_{id}/
    └── metrics.csv              # round, loss, accuracy_before, accuracy_after, energy_consumed, energy_ratio
```

---

## References

- McMahan, B. et al. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS*. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)
- Briggs, C., Fan, Z., & Andras, P. (2020). Federated learning with hierarchical clustering of local updates to improve training on non-IID data. [arXiv:2004.11791](https://arxiv.org/abs/2004.11791)
- Ghosh, A. et al. (2020). An efficient framework for clustered federated learning. *NeurIPS*. [arXiv:2006.04088](https://arxiv.org/abs/2006.04088)
- Tan, Y. et al. (2022). Towards personalized federated learning. *IEEE TNNLS*. [arXiv:2103.00710](https://arxiv.org/abs/2103.00710)
- Li, X. et al. (2021). FeSEM: Federated learning via expectation maximization. *IEEE TPAMI*.
- Zhang, Y. et al. (2025). LCFed: An efficient clustered federated learning framework for heterogeneous data. *ICASSP*. [arXiv:2501.01850](https://arxiv.org/abs/2501.01850)
- Zhang, Y. et al. (2015). Character-level convolutional networks for text classification. *NeurIPS*. [arXiv:1509.01626](https://arxiv.org/abs/1509.01626)
- Caldas, S. et al. (2019). LEAF: A benchmark for federated settings. [arXiv:1812.01097](https://arxiv.org/abs/1812.01097)

---

## Author

Antoine Hounsi — [antoinehounsi3@gmail.com](mailto:antoinehounsi3@gmail.com)

Master's research project, Université de Lille.
