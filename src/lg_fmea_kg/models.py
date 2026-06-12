"""Feature fusion for the four ablation modes and the evaluation routine.

The four configurations compared in the paper are:

* STAT          - statistical features only (data-only baseline);
* STAT_KG       - statistical features plus static FMEA graph embeddings
                  and cosine similarities (an ungrounded knowledge-graph
                  baseline in the spirit of FKGCN, Lyu et al.);
* STAT_LG       - statistical features plus Landau-Ginzburg features;
* STAT_LG_KG    - statistical + LG features + physics-grounded FMEA graph
                  embeddings gated by LG-to-mode similarity (proposed).

`evaluate` performs a stratified split, class-balanced resampling, a
two-layer MLP, and reports macro precision / recall / F1 plus the F1 on a
designated rare class. A scarcity option drops a fraction of the rare
class from training to probe robustness under label scarcity.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .kg import FMEAGraph


def _cosine(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    v_n = v / (np.linalg.norm(v) + 1e-9)
    M_n = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return (M_n @ v_n).astype(np.float32)


def _row_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (A_n @ B_n.T).astype(np.float32)


def make_features(blocks: List[dict], mode: str, kg: FMEAGraph):
    """Concatenate the feature views selected by `mode` across all blocks."""
    X_parts, y_parts = [], []
    for b in blocks:
        n = b["stat"].shape[0]
        feats = [b["stat"]]
        if mode == "STAT":
            pass
        elif mode == "STAT_KG":
            fm_emb = kg.static_failure_mode_embeddings()
            P = np.random.default_rng(7).normal(
                0, 1, size=(b["stat"].shape[1], kg.dim)
            ).astype(np.float32)
            sims = _row_cosine(b["stat"] @ P, fm_emb)
            feats.append(np.tile(fm_emb.reshape(1, -1), (n, 1)))
            feats.append(sims)
        elif mode == "STAT_LG":
            feats.append(b["lg"])
        elif mode == "STAT_LG_KG":
            feats.append(b["lg"])
            grounded = kg.lg_grounded_embeddings(b["soft_zone"])
            P_lg = np.random.default_rng(7).normal(
                0, 1, size=(b["lg"].shape[1], kg.dim)
            ).astype(np.float32)
            lg_proj = b["lg"] @ P_lg
            sims = np.zeros((n, len(kg.FAILURE_MODES)), dtype=np.float32)
            for i in range(n):
                sims[i] = _cosine(lg_proj[i], grounded[i])
            gated = (grounded * sims[:, :, None]).reshape(n, -1)
            feats.append(gated)
            feats.append(sims)
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        X_parts.append(np.concatenate(feats, axis=1))
        y_parts.append(b["y"])
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def evaluate(
    mode: str,
    kg: FMEAGraph,
    blocks: List[dict],
    seed: int = 0,
    drop_rare_frac: Optional[float] = None,
    rare_class: int = 2,
) -> dict:
    """Train and score one configuration with a fixed random seed."""
    X, y = make_features(blocks, mode, kg)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    rng = np.random.default_rng(seed + 1000)

    if drop_rare_frac is not None:
        rare_idx = np.where(ytr == rare_class)[0]
        if len(rare_idx) > 0:
            keep_n = max(int(drop_rare_frac * len(rare_idx)), 1)
            keep = rng.choice(rare_idx, size=keep_n, replace=False)
            drop = np.setdiff1d(rare_idx, keep)
            m = np.ones(len(ytr), dtype=bool)
            m[drop] = False
            Xtr = Xtr[m]
            ytr = ytr[m]

    classes, counts = np.unique(ytr, return_counts=True)
    target = counts.max()
    Xb_parts, yb_parts = [], []
    for c, ct in zip(classes, counts):
        idx = np.where(ytr == c)[0]
        sel = rng.choice(idx, size=target, replace=(ct < target))
        Xb_parts.append(Xtr[sel])
        yb_parts.append(ytr[sel])
    Xtr_b = np.concatenate(Xb_parts)
    ytr_b = np.concatenate(yb_parts)
    perm = rng.permutation(len(ytr_b))
    Xtr_b, ytr_b = Xtr_b[perm], ytr_b[perm]

    sc = StandardScaler().fit(Xtr_b)
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=400,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=15,
    )
    clf.fit(sc.transform(Xtr_b), ytr_b)
    yp = clf.predict(sc.transform(Xte))

    return {
        "mode": mode,
        "n_feats": X.shape[1],
        "precision": precision_score(yte, yp, average="macro", zero_division=0),
        "recall": recall_score(yte, yp, average="macro", zero_division=0),
        "f1": f1_score(yte, yp, average="macro", zero_division=0),
        "f1_rare": f1_score(yte == rare_class, yp == rare_class, zero_division=0),
        "yte": yte,
        "yp": yp,
    }
