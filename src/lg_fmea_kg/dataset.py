"""Assemble per-bearing feature blocks and integer labels.

For each bearing run the snapshots are loaded once and turned into a
dictionary holding the statistical features, the Landau-Ginzburg features,
the soft-zone membership and the integer labels. Healthy snapshots (LG
zone 0) are labelled 0 regardless of the bearing's eventual fault mode;
the remaining snapshots take the bearing's fault class.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path

import numpy as np

from .config import FAULT_TO_CLASS
from .data import make_runs, build_lifetime
from .features import stat_feature_matrix, damage_indicator_matrix
from .lg import LGOrderParameter, lg_to_soft_zones


def assign_labels(zones: np.ndarray, fault_mode: str) -> np.ndarray:
    """Healthy zones -> class 0, everything else -> the bearing's fault class."""
    cls = FAULT_TO_CLASS[fault_mode]
    return np.where(zones == 0, 0, cls).astype(np.int64)


def build_dataset(
    bearing_paths: Dict[str, Path],
    subset: str = "wc2",
    lg_kwargs: Optional[dict] = None,
    verbose: bool = True,
) -> List[dict]:
    """Return one feature block per bearing in the chosen subset."""
    lg_kwargs = lg_kwargs or {}
    runs = make_runs(bearing_paths, subset=subset)
    blocks = []
    for r in runs:
        if verbose:
            print(f"  loading {r.path.name} ({r.fault_mode})...", end=" ", flush=True)
        snaps = build_lifetime(r)
        stat_X = stat_feature_matrix(snaps)
        dmg = damage_indicator_matrix(snaps)
        healthy_end = max(int(0.30 * snaps.shape[0]), 5)
        lg = LGOrderParameter(**lg_kwargs).fit_healthy(dmg, slice(0, healthy_end))
        lg_feats = lg.transform(dmg)
        zones = lg.zones(lg_feats)
        soft = lg_to_soft_zones(lg_feats, lg.eps_h, lg.eps_c)
        y = assign_labels(zones, r.fault_mode)
        blocks.append(
            {
                "stat": stat_X,
                "lg": lg_feats,
                "soft_zone": soft,
                "y": y,
                "fault": r.fault_mode,
                "lg_obj": lg,
                "raw_dmg": dmg,
            }
        )
        del snaps
        if verbose:
            print(f"N={stat_X.shape[0]}")
    return blocks
