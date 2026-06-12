"""Locate XJTU-SY bearing folders on disk and load their snapshots.

The original Colab notebook mounted Google Drive, extracted the multi-part
`.rar` archives and built a global bearing -> path mapping. This module
performs the same discovery against any local directory, so the pipeline
runs identically on a workstation, a cluster node or a CI runner without
any Colab-specific dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import BEARING_LABELS, N_SAMPLES

_BEARING_PAT = re.compile(r"bearing[ _-]?(\d)[ _-]?(\d)", re.IGNORECASE)


@dataclass
class BearingRun:
    """A single run-to-failure recording for one bearing."""

    fault_mode: str
    n_snapshots: int
    path: Optional[Path] = None
    rng_seed: int = 0
    channel: int = 0


def _has_data_files(p: Path, min_files: int = 5) -> bool:
    """True if a folder directly contains at least `min_files` snapshot files."""
    try:
        files = list(p.iterdir())
    except (PermissionError, OSError):
        return False
    csv_like = [
        f
        for f in files
        if f.is_file() and (f.suffix.lower() == ".csv" or f.stem.isdigit())
    ]
    return len(csv_like) >= min_files


def _descend_to_data(p: Path, max_descents: int = 3) -> Path:
    """Follow single-child directories until the snapshot files are reached."""
    cur = p
    for _ in range(max_descents):
        if _has_data_files(cur):
            return cur
        try:
            subs = [x for x in cur.iterdir() if x.is_dir()]
        except (PermissionError, OSError):
            return cur
        if len(subs) == 1:
            cur = subs[0]
        else:
            break
    return cur


def discover_bearings(
    root: Path | str,
    manual_override: Optional[Dict[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Scan `root` and return a mapping {canonical_name: folder_with_snapshots}.

    Canonical names follow the `BearingX_Y` convention. Folders are matched
    by name regardless of separators (e.g. "Bearing2_1", "bearing 2-1") and
    the search descends through single wrapper directories to reach the
    folder that directly holds the per-minute snapshot files.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    bearing_paths: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        m = _BEARING_PAT.search(p.name)
        if not m:
            continue
        canonical = f"Bearing{m.group(1)}_{m.group(2)}"
        if canonical not in BEARING_LABELS:
            continue
        actual = _descend_to_data(p)
        if _has_data_files(actual):
            bearing_paths.setdefault(canonical, actual)

    for k, v in (manual_override or {}).items():
        actual = _descend_to_data(Path(v))
        if _has_data_files(actual):
            bearing_paths[k] = actual
            if verbose:
                print(f"  manual override applied for {k}: {actual}")
        elif verbose:
            print(f"  WARNING: override path for {k} has no data files: {v}")

    if verbose:
        print(f"Detected {len(bearing_paths)} of {len(BEARING_LABELS)} bearings.")
        wc2 = sum(
            1
            for k in BEARING_LABELS
            if k.startswith("Bearing2_") and k in bearing_paths
        )
        print(f"  WC2 bearings (main experiment): {wc2} of 5")
    return bearing_paths


def make_runs(bearing_paths: Dict[str, Path], subset: str = "wc2") -> List[BearingRun]:
    """Build the list of BearingRun objects for a working-condition subset.

    `subset` is one of: "wc1", "wc2", "wc3", "all", or "fast" (the first two
    detected WC2 bearings, used for the sanity check).
    """
    if subset == "wc1":
        keys = [k for k in BEARING_LABELS if k.startswith("Bearing1_")]
    elif subset == "wc2":
        keys = [k for k in BEARING_LABELS if k.startswith("Bearing2_")]
    elif subset == "wc3":
        keys = [k for k in BEARING_LABELS if k.startswith("Bearing3_")]
    elif subset == "all":
        keys = list(BEARING_LABELS.keys())
    elif subset == "fast":
        keys = [
            k
            for k in BEARING_LABELS
            if k.startswith("Bearing2_") and k in bearing_paths
        ][:2]
    else:
        raise ValueError(f"unknown subset: {subset!r}")

    runs: List[BearingRun] = []
    for i, k in enumerate(keys):
        if k not in bearing_paths:
            print(f"  WARNING: {k} not detected, skipping")
            continue
        fault, n = BEARING_LABELS[k]
        runs.append(BearingRun(fault, n, bearing_paths[k], rng_seed=i + 1))
    return runs


def _list_data_files(folder: Path) -> List[Path]:
    candidates = []
    for f in folder.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() == ".csv":
            candidates.append(f)
        elif not f.suffix and f.stem.isdigit():
            candidates.append(f)
    try:
        return sorted(candidates, key=lambda p: int(p.stem))
    except ValueError:
        return sorted(candidates)


def _load_snapshot(path: Path, channel: int = 0) -> np.ndarray:
    """Load one 1.28 s snapshot, returning a fixed-length float32 vector."""
    try:
        df = pd.read_csv(path)
        if df.shape[0] != N_SAMPLES:
            df = pd.read_csv(path, header=None)
    except Exception:
        df = pd.read_csv(path, header=None)
    arr = df.iloc[:, channel].to_numpy(dtype=np.float32)
    if arr.shape[0] != N_SAMPLES:
        arr = (
            arr[:N_SAMPLES]
            if arr.shape[0] > N_SAMPLES
            else np.pad(arr, (0, N_SAMPLES - arr.shape[0]))
        )
    return arr


def build_lifetime(run: BearingRun) -> np.ndarray:
    """Stack every snapshot of a bearing into an (n_snapshots, N_SAMPLES) array."""
    if run.path is None or not run.path.exists():
        raise FileNotFoundError(f"Path does not exist: {run.path}")
    files = _list_data_files(run.path)
    if not files:
        raise FileNotFoundError(
            f"No CSV/data files in {run.path}. "
            f"Contents: {[p.name for p in list(run.path.iterdir())[:5]]}"
        )
    return np.stack([_load_snapshot(p, run.channel) for p in files])
