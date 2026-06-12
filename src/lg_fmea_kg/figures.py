"""Figure generation for the paper.

`trajectory_figure` renders the LG order-parameter trajectory, the
fault-mode cross-attention over the bearing lifetime, and the zone
classification for one bearing block. `scarcity_figure` renders the
rare-class F1 against the fraction of available rare-class labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .kg import FMEAGraph
from .models import _cosine


def trajectory_figure(blocks: List[dict], kg: FMEAGraph, out_path: Path,
                      block_index: int = 0, dpi: int = 130):
    import matplotlib.pyplot as plt

    b = blocks[block_index]
    n = b["stat"].shape[0]
    grounded = kg.lg_grounded_embeddings(b["soft_zone"])
    P_lg = np.random.default_rng(7).normal(
        0, 1, size=(b["lg"].shape[1], kg.dim)
    ).astype(np.float32)
    lg_proj = b["lg"] @ P_lg
    sims = np.zeros((n, len(kg.FAILURE_MODES)), dtype=np.float32)
    for i in range(n):
        sims[i] = _cosine(lg_proj[i], grounded[i])
    T = 0.3
    e = np.exp((sims - sims.max(axis=1, keepdims=True)) / T)
    attn = e / e.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    t = np.arange(n)

    axes[0].plot(t, b["lg"][:, 3], lw=1.5, color="#222")
    eh, ec = b["lg_obj"].eps_h, b["lg_obj"].eps_c
    axes[0].axhline(eh, ls="--", color="#ffb74d", alpha=0.6, label=fr"$\epsilon_h$={eh}")
    axes[0].axhline(ec, ls="--", color="#00838f", alpha=0.6, label=fr"$\epsilon_c$={ec}")
    axes[0].set_ylabel(r"$|\psi(t)|$")
    axes[0].set_title(f"LG order parameter trajectory ({b['fault']})")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(alpha=0.3)

    colors = ["#2ca02c", "#1f77b4", "#e6a817", "#00838f"]
    gt_map = {"healthy": 0, "inner_race": 1, "outer_race": 2,
              "cage": 3, "rolling_element": 3}
    gt_idx = gt_map.get(b["fault"], -1)
    labels = ["healthy", "inner_race", "outer_race", "cage/RE"]
    labels = [l + " (GT)" if i == gt_idx else l for i, l in enumerate(labels)]
    axes[1].stackplot(t, attn.T, labels=labels, colors=colors, alpha=0.85)
    axes[1].set_ylabel("cross-attention weight")
    axes[1].set_title("Fault-mode cross-attention over the bearing's lifetime")
    axes[1].legend(loc="upper left", fontsize=9, ncol=4)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)

    zone_colors = {0: "#2ca02c", 1: "#ffb74d", 2: "#00838f"}
    zone_names = {0: "HEALTHY", 1: "BUFFER", 2: "CRITICAL"}
    zones = b["lg_obj"].zones(b["lg"])
    for z, c in zone_colors.items():
        mask = zones == z
        axes[2].fill_between(t, 0, 1, where=mask, color=c, alpha=0.55,
                             label=zone_names[z])
    axes[2].set_xlabel("snapshot index (minutes)")
    axes[2].set_yticks([])
    axes[2].set_title("LG zone classification")
    axes[2].legend(loc="upper left", fontsize=9, ncol=3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def scarcity_figure(fractions: List[float], modes: List[str],
                    results_scarcity: Dict[str, dict], out_path: Path,
                    dpi: int = 130):
    import matplotlib.pyplot as plt

    styles = {
        "STAT": {"c": "#888888", "marker": "o", "label": "STAT (data only)"},
        "STAT_KG": {"c": "#1f77b4", "marker": "s", "label": "STAT + FKGCN (Lyu et al.)"},
        "STAT_LG": {"c": "#e6a817", "marker": "^", "label": "STAT + LG"},
        "STAT_LG_KG": {"c": "#5e3c99", "marker": "D",
                       "label": "STAT + LG + grounded KG (proposed)", "lw": 2.6},
    }
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    fr = np.array(fractions) * 100
    for m in modes:
        s = styles[m]
        mean = np.array(results_scarcity[m]["f1_rare"])
        sd = np.array(results_scarcity[m]["f1_rare_sd"])
        ax.errorbar(fr, mean, yerr=sd, color=s["c"], marker=s["marker"],
                    label=s["label"], capsize=3, lw=s.get("lw", 1.6), markersize=7)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 25, 50, 100])
    ax.set_xticklabels(["5%", "10%", "25%", "50%", "100%"])
    ax.invert_xaxis()
    ax.set_xlabel("Fraction of training labels available for rare class (outer_race)")
    ax.set_ylabel("F1 on rare class (test)")
    ax.set_title("Performance under rare-class scarcity (XJTU-SY real)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
