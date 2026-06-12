"""Synthetic run-to-failure data generator for smoke testing.

This produces a tiny directory tree that mimics the XJTU-SY folder layout
(`BearingX_Y/<minute>.csv`) so the full pipeline can be exercised end to
end on a laptop or a CI runner without downloading the real 5 GB dataset.

The signals are NOT physically meaningful and must never be reported as
results. They exist only to verify that the code runs and the shapes line
up. Use the real XJTU-SY dataset for any reported numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import N_SAMPLES, FS

# A small WC2 subset is enough to drive the four-way ablation.
_SYNTH_BEARINGS = {
    "Bearing2_1": ("inner_race", 24),
    "Bearing2_2": ("outer_race", 24),
    "Bearing2_3": ("cage", 24),
    "Bearing2_4": ("outer_race", 24),
    "Bearing2_5": ("outer_race", 24),
}


def _synthetic_snapshot(t_frac: float, fault: str, rng: np.random.Generator) -> np.ndarray:
    """A noisy sinusoid whose impulsiveness grows with the lifetime fraction."""
    t = np.arange(N_SAMPLES) / FS
    base_freq = {"inner_race": 162.0, "outer_race": 107.0,
                 "cage": 41.0, "rolling_element": 138.0}.get(fault, 100.0)
    severity = max(0.0, t_frac - 0.4) / 0.6  # healthy for first 40% of life
    carrier = 0.05 * np.sin(2 * np.pi * 35.0 * t)
    impulses = severity * np.sin(2 * np.pi * base_freq * t) * (
        1 + 3 * severity * np.abs(np.sin(2 * np.pi * 5 * t))
    )
    noise = rng.normal(0, 0.05 + 0.15 * severity, size=N_SAMPLES)
    x = carrier + impulses + noise
    return x.astype(np.float32)


def generate(root: Path | str, seed: int = 0, n_per_bearing: int | None = None) -> Path:
    """Write a synthetic dataset under `root` and return the path."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    for name, (fault, n_default) in _SYNTH_BEARINGS.items():
        n = n_per_bearing or n_default
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for minute in range(1, n + 1):
            x = _synthetic_snapshot(minute / n, fault, rng)
            ch2 = _synthetic_snapshot(minute / n, fault, rng)
            arr = np.stack([x, ch2], axis=1)
            np.savetxt(folder / f"{minute}.csv", arr, delimiter=",",
                       header="Horizontal_vibration,Vertical_vibration",
                       comments="")
    return root


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/synthetic")
    generate(target)
    print(f"Synthetic dataset written to {target}")
