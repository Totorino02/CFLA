"""
Enregistrement des directions de mise à jour pour HCFL.

Tout est calculé côté serveur à partir des state_dicts déjà disponibles :
  - Δ_i        = ω_i - Ω_k(avant)              mise à jour du client i
  - g_k        = Avg_k - Ω_k(avant)            mise à jour locale du cluster k
  - s_k        = Ω_k(après) - Avg_k            déplacement dû au partage (= λ(Φ - Avg_k) sur embed)
  - c_{j→k}    = λ · w_j · (Φ_j - Avg_k)       contribution du cluster j au partage reçu par k
                 (Σ_j c_{j→k} = s_k exactement)

  Chaque contribution se décompose exactement (sur embed) en trois termes :
      c_{j→k} = know_{j→k} + align_{j→k} + damp_{j→k}
      know_{j→k}  = λ w_j (Φ_j - Ω_j(avant))   ce que j a appris ce round  -> la « connaissance »
      align_{j→k} = λ w_j (Ω_j - Ω_k)(avant)    rapprochement des centres
      damp_{j→k}  = -λ w_j g_k                  part de sa propre mise à jour que k perd
  Le cosinus brut cos(c_{j→k}, g_k) est biaisé négativement par damp ; la mesure de
  transfert à lire est cos(know_{j→k}, g_k) (même round et round suivant).

Les métriques scalaires sont exactes (vecteurs complets). Les vecteurs eux-mêmes ne
sont stockés que sous forme de count-sketch (dimension réduite, préserve
approximativement normes et produits scalaires) pour les visualisations hors ligne.

Fichiers produits dans <output_dir>/grad_logs/ :
  meta.json                  clés, groupes de couches, config
  clustering.npz             signal utilisé pour le clustering + labels
  grad_metrics.csv           1 ligne par (round, cluster)
  grad_transfer.csv          1 ligne par (round, source, cible)
  grad_layers.csv            1 ligne par (round, couche)
  <phase>_<round>.npz        matrices, coordonnées du cône, sketches
"""

import csv
import json
import os
from collections import OrderedDict

import numpy as np

EPS = 1e-12


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach()
        if hasattr(x, "float"):
            x = x.float()
        x = x.cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _cos(a, b):
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > EPS else float("nan")


def _nanmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def _cos_matrix(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    Vn = V / np.maximum(n, EPS)
    return Vn @ Vn.T


def _mean_offdiag(M):
    k = M.shape[0]
    if k < 2:
        return float("nan")
    return float((M.sum() - np.trace(M)) / (k * (k - 1)))


class GradRecorder:
    def __init__(
        self,
        output_dir,
        named_param_sizes,
        embed_prefix="embed.",
        sketch_dim=1024,
        seed=0,
        deferred_transfer=True,
        config=None,
    ):
        """
        named_param_sizes : liste de (nom, numel) dans l'ordre de model.named_parameters().
        Seuls les paramètres entraînables sont suivis (les buffers BN, etc. sont ignorés).
        """
        self.dir = os.path.join(output_dir, "grad_logs")
        os.makedirs(self.dir, exist_ok=True)
        self.embed_prefix = embed_prefix
        self.deferred_transfer = deferred_transfer

        self.keys = [n for n, _ in named_param_sizes]
        sizes = [int(s) for _, s in named_param_sizes]
        self.d = int(sum(sizes))
        self.slices = OrderedDict()
        off = 0
        for k, s in zip(self.keys, sizes):
            self.slices[k] = slice(off, off + s)
            off += s

        self.embed_mask = np.zeros(self.d, dtype=bool)
        for k in self.keys:
            if k.startswith(embed_prefix):
                self.embed_mask[self.slices[k]] = True
        self.embed_keys = [k for k in self.keys if k.startswith(embed_prefix)]

        # Groupes de couches : "embed.conv1.weight" + "embed.conv1.bias" -> "embed.conv1"
        self.groups = OrderedDict()
        for k in self.keys:
            g = k.rsplit(".", 1)[0] if "." in k else k
            self.groups.setdefault(g, []).append(self.slices[k])

        # Count-sketch : chaque coordonnée -> un seau aléatoire avec un signe aléatoire
        rng = np.random.default_rng(seed)
        self.sketch_dim = int(sketch_dim)
        self.sk_idx = rng.integers(0, self.sketch_dim, self.d)
        self.sk_sign = (rng.integers(0, 2, self.d) * 2 - 1).astype(np.float32)

        self._pending = None  # contributions du round précédent (transfert différé)

        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(
                {
                    "param_keys": self.keys,
                    "embed_keys": self.embed_keys,
                    "groups": list(self.groups.keys()),
                    "d": self.d,
                    "d_embed": int(self.embed_mask.sum()),
                    "sketch_dim": self.sketch_dim,
                    "config": config or {},
                },
                f,
                indent=1,
            )

        self._csv(
            "grad_metrics.csv",
            [
                "phase",
                "round",
                "cluster",
                "n_selected",
                "lambda",
                "norm_local",
                "norm_local_embed",
                "norm_local_head",
                "norm_share",
                "cos_share_local",
                "cos_local_mean_all",
                "cos_local_mean_embed",
                "cos_local_mean_head",
                "shared_norm",
                "specific_norm",
                "intra_client_cos",
                "client_cos_to_cluster",
                "cfl_ratio",
                "mean_client_norm",
                "max_client_norm",
                "mean_g_sup",
                "mean_g_prox",
                "mean_cos_sup_prox",
            ],
            new=True,
        )
        self._csv(
            "grad_transfer.csv",
            [
                "round",
                "source",
                "target",
                "lambda",
                "weight",
                "norm_contrib",
                "norm_know",
                "norm_align",
                "norm_damp",
                "cos_contrib_same",
                "cos_know_same",
                "cos_know_next",
            ],
            new=True,
        )
        self._csv(
            "grad_layers.csv",
            [
                "phase",
                "round",
                "layer",
                "mean_inter_cluster_cos",
                "mean_norm",
            ],
            new=True,
        )

    # ------------------------------------------------------------------ utils
    def _csv(self, name, row, new=False):
        with open(os.path.join(self.dir, name), "w" if new else "a", newline="") as f:
            csv.writer(f).writerow(row)

    def flat(self, state):
        return np.concatenate([_to_np(state[k]).reshape(-1) for k in self.keys])

    def flat_embed(self, state):
        return np.concatenate([_to_np(state[k]).reshape(-1) for k in self.embed_keys])

    def sketch(self, V):
        V = np.atleast_2d(V)
        out = np.zeros((V.shape[0], self.sketch_dim), dtype=np.float32)
        for r in range(V.shape[0]):
            out[r] = np.bincount(
                self.sk_idx, weights=V[r] * self.sk_sign, minlength=self.sketch_dim
            )
        return out

    def _layer_stats(self, G):
        """G : (K, d). Renvoie cos inter-cluster [L,K,K] et normes moyennes [L]."""
        mats, norms = [], []
        for _, sls in self.groups.items():
            sub = np.concatenate([G[:, s] for s in sls], axis=1)
            mats.append(_cos_matrix(sub))
            norms.append(float(np.linalg.norm(sub, axis=1).mean()))
        return np.stack(mats), np.array(norms)

    @staticmethod
    def _stats_arrays(ids, client_stats):
        keys = ["g_sup_norm", "g_prox_norm", "cos_sup_prox", "g_total_norm", "steps"]
        out = {}
        for k in keys:
            out["client_" + k] = np.array(
                [float((client_stats or {}).get(c, {}).get(k, np.nan)) for c in ids]
            )
        return out

    # ------------------------------------------------------------ warm-up
    def record_warmup(self, round_idx, ref_state, client_states, client_stats=None, phase="warmup"):
        """Mises à jour Δ_i = ω_i - global pendant FedAvg (pas encore de clusters)."""
        ids = sorted(client_states)
        ref = self.flat(ref_state)
        D = np.stack([self.flat(client_states[c]) - ref for c in ids])
        norms = np.linalg.norm(D, axis=1)
        C = _cos_matrix(D)
        mean_upd = D.mean(0)
        cfl = float(np.linalg.norm(mean_upd) / (norms.max() + EPS))
        st = self._stats_arrays(ids, client_stats)
        self._csv(
            "grad_metrics.csv",
            [
                phase,
                round_idx,
                -1,
                len(ids),
                0.0,
                np.linalg.norm(mean_upd),
                np.linalg.norm(mean_upd[self.embed_mask]),
                np.linalg.norm(mean_upd[~self.embed_mask]),
                0.0,
                "",
                "",
                "",
                "",
                "",
                "",
                _mean_offdiag(C),
                "",
                cfl,
                norms.mean(),
                norms.max(),
                _nanmean(st["client_g_sup_norm"]),
                "",
                "",
            ],
        )
        np.savez_compressed(
            os.path.join(self.dir, f"{phase}_{round_idx:04d}.npz"),
            phase=phase,
            round=round_idx,
            client_ids=np.array(ids),
            client_norms=norms,
            client_cos=C,
            client_cos_embed=_cos_matrix(D[:, self.embed_mask]),
            client_sketch=self.sketch(D),
            **st,
        )

    def record_clustering(self, updates, labels, signal):
        U = np.asarray(updates, dtype=np.float32)
        Uc = U - U.mean(0, keepdims=True)
        _, S, Vt = np.linalg.svd(Uc, full_matrices=False)
        pca2 = Uc @ Vt[:2].T
        np.savez_compressed(
            os.path.join(self.dir, "clustering.npz"),
            labels=np.asarray(labels),
            cos=_cos_matrix(U),
            norms=np.linalg.norm(U, axis=1),
            pca2=pca2,
            explained=(S[:2] ** 2) / max(float((S**2).sum()), EPS),
            signal=signal,
        )

    # ---------------------------------------------------------- clustered
    def record_cluster_round(
        self,
        round_idx,
        lam,
        R,
        centers_before,
        centers_after,
        avg_states,
        phi_parts,
        client_states,
        client_stats=None,
    ):
        """
        centers_before / centers_after : listes de state_dicts Ω_k avant / après agrégation
        avg_states  : {k: Avg_k}  (moyenne des clients sélectionnés du cluster, avant mélange)
        phi_parts   : {j: (w_j, Φ_j)}  avec Φ = Σ_j w_j Φ_j  (Φ_j : state_dict embed, clés "embed.*")
        """
        active = sorted(avg_states)
        K = len(active)
        pos = {k: i for i, k in enumerate(active)}
        em = self.embed_mask

        before = {k: self.flat(centers_before[k]) for k in active}
        A = {k: self.flat(avg_states[k]) for k in active}
        G = np.stack([A[k] - before[k] for k in active])  # g_k
        S = np.stack([self.flat(centers_after[k]) - A[k] for k in active])  # s_k

        # Contributions c_{j→k} (espace embed) et leur décomposition
        phi = {j: (float(w), self.flat_embed(p)) for j, (w, p) in phi_parts.items()}
        contrib, know, align = {}, {}, {}
        for k in active:
            Ae = A[k][em]
            for j in phi:
                w, pj = phi[j]
                contrib[(j, k)] = lam * w * (pj - Ae)
                know[(j, k)] = lam * w * (pj - before[j][em])
                align[(j, k)] = lam * w * (before[j][em] - before[k][em])
            err = np.linalg.norm(sum(contrib[(j, k)] for j in phi) - S[pos[k]][em])
            if err > 1e-3 * (np.linalg.norm(S[pos[k]]) + 1e-6):
                print(f"[GradRecorder] attention : Σ contributions ≠ s_{k} (écart {err:.2e})")

        # Clients
        ids = sorted(client_states)
        D = np.stack([self.flat(client_states[c]) - before[R[c]] for c in ids])
        cl = np.array([R[c] for c in ids])
        cnorm = np.linalg.norm(D, axis=1)
        Ccl = _cos_matrix(D)
        st = self._stats_arrays(ids, client_stats)

        Gbar = G.mean(0)
        u = Gbar / (np.linalg.norm(Gbar) + EPS)

        # Transfert différé : cos(know_{j→k}^{r-1}, g_k^r)
        if self._pending is not None:
            for row, kvec, k in self._pending:
                nxt = _cos(kvec, G[pos[k]][em]) if k in pos else float("nan")
                self._csv("grad_transfer.csv", row + [nxt])
            self._pending = None

        pending = []
        T_same = np.full((K, K), np.nan)  # cos(know_{j→k}, g_k)   [cible, source]
        T_raw = np.full((K, K), np.nan)  # cos(c_{j→k}, g_k)
        for (j, k), vec in contrib.items():
            if j == k or j not in pos:
                continue
            gk = G[pos[k]][em]
            c_know = _cos(know[(j, k)], gk)
            c_raw = _cos(vec, gk)
            T_same[pos[k], pos[j]] = c_know
            T_raw[pos[k], pos[j]] = c_raw
            w = phi[j][0]
            row = [
                round_idx,
                j,
                k,
                lam,
                w,
                np.linalg.norm(vec),
                np.linalg.norm(know[(j, k)]),
                np.linalg.norm(align[(j, k)]),
                lam * w * np.linalg.norm(gk),
                c_raw,
                c_know,
            ]
            pending.append((row, know[(j, k)], k))
        if self.deferred_transfer:
            self._pending = pending
        else:
            for row, _, _ in pending:
                self._csv("grad_transfer.csv", row + [""])

        # Métriques par cluster
        Gbar_e, Gbar_h = Gbar[em], Gbar[~em]
        for k in active:
            g, s = G[pos[k]], S[pos[k]]
            h = float(g @ u)
            spec = float(np.sqrt(max(g @ g - h * h, 0.0)))
            m = cl == k
            Dk = D[m]
            Ck = Ccl[np.ix_(m, m)]
            to_cluster = _nanmean([_cos(x, g) for x in Dk]) if m.any() else np.nan
            cfl = np.linalg.norm(Dk.mean(0)) / (cnorm[m].max() + EPS) if m.any() else np.nan
            self._csv(
                "grad_metrics.csv",
                [
                    "cluster",
                    round_idx,
                    k,
                    int(m.sum()),
                    lam,
                    np.linalg.norm(g),
                    np.linalg.norm(g[em]),
                    np.linalg.norm(g[~em]),
                    np.linalg.norm(s),
                    _cos(s[em], g[em]),
                    _cos(g, Gbar),
                    _cos(g[em], Gbar_e),
                    _cos(g[~em], Gbar_h),
                    h,
                    spec,
                    _mean_offdiag(Ck),
                    to_cluster,
                    cfl,
                    cnorm[m].mean(),
                    cnorm[m].max(),
                    _nanmean(st["client_g_sup_norm"][m]),
                    _nanmean(st["client_g_prox_norm"][m]),
                    _nanmean(st["client_cos_sup_prox"][m]),
                ],
            )

        # Couches
        Lcos, Lnorm = self._layer_stats(G)
        for (name, _), M, n in zip(self.groups.items(), Lcos, Lnorm):
            self._csv("grad_layers.csv", ["cluster", round_idx, name, _mean_offdiag(M), n])

        # Repère du cône (exact) : axe vertical = Ḡ, plan = composantes spécifiques
        heights = G @ u
        Rk = G - np.outer(heights, u)
        lam_e, V = np.linalg.eigh(Rk @ Rk.T)
        order = np.argsort(lam_e)[::-1][:2]
        dirs = np.zeros((2, self.d), dtype=np.float32)
        for i, o in enumerate(order):
            if lam_e[o] > EPS:
                dirs[i] = (Rk.T @ V[:, o]) / np.sqrt(lam_e[o])
        xy = Rk @ dirs.T
        # Orientation stable : 1er cluster à azimut 0, 2e cluster en y > 0
        a = np.arctan2(xy[0, 1], xy[0, 0])
        Q = np.array([[np.cos(-a), -np.sin(-a)], [np.sin(-a), np.cos(-a)]])
        xy, dirs = xy @ Q.T, Q @ dirs
        if K > 1 and xy[1, 1] < 0:
            xy[:, 1] *= -1
            dirs[1] *= -1

        def cone(Vs):
            Vs = np.atleast_2d(Vs)
            hh = Vs @ u
            pp = Vs @ dirs.T
            res = np.sqrt(np.maximum((Vs * Vs).sum(1) - hh**2 - (pp**2).sum(1), 0.0))
            return np.column_stack([pp, hh, res])  # x, y, hauteur, norme hors repère

        def lift(v):
            full = np.zeros(self.d, dtype=np.float32)
            full[em] = v
            return full

        contrib_cone = np.full((K, K, 4), np.nan)
        know_cone = np.full((K, K, 4), np.nan)
        for (j, k), vec in contrib.items():
            if j in pos:
                contrib_cone[pos[k], pos[j]] = cone(lift(vec))[0]
                know_cone[pos[k], pos[j]] = cone(lift(know[(j, k)]))[0]

        np.savez_compressed(
            os.path.join(self.dir, f"cluster_{round_idx:04d}.npz"),
            phase="cluster",
            round=round_idx,
            lam=lam,
            clusters=np.array(active),
            cluster_cos=_cos_matrix(G),
            cluster_cos_embed=_cos_matrix(G[:, em]),
            cluster_cos_head=_cos_matrix(G[:, ~em]) if (~em).any() else np.zeros((K, K)),
            layer_cos=Lcos,
            layer_names=np.array(list(self.groups.keys())),
            layer_norms=Lnorm,
            transfer_know_same=T_same,
            transfer_raw_same=T_raw,
            cone_clusters=np.column_stack([xy, heights]),
            cone_share=cone(S),
            cone_clients=cone(D),
            cone_contrib=contrib_cone,
            cone_know=know_cone,
            client_ids=np.array(ids),
            client_clusters=cl,
            client_norms=cnorm,
            client_cos=Ccl,
            sketch_local=self.sketch(G),
            sketch_share=self.sketch(S),
            client_sketch=self.sketch(D),
            **st,
        )

    def finalize(self):
        """Écrit les transferts du dernier round (sans cos_next_round)."""
        if self._pending is not None:
            for row, _, _ in self._pending:
                self._csv("grad_transfer.csv", row + [""])
            self._pending = None
