"""Vibration features extracted from each snapshot.

Two families are produced:

* statistical descriptors (RMS, peak, kurtosis, skewness, ...), used by
  every configuration as the data-only baseline; and
* physics-motivated damage indicators (envelope-band energy and spectral
  kurtosis) that feed the Landau-Ginzburg order parameter.
"""

from __future__ import annotations

import numpy as np
from scipy import signal, stats

from .config import FS


def stat_features(x: np.ndarray) -> list:
    """Seven time-domain statistical descriptors of a single snapshot."""
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    return [
        rms,
        peak,
        stats.kurtosis(x, fisher=True),
        stats.skew(x),
        np.std(x),
        peak / (rms + 1e-9),
        np.ptp(x),
    ]


def stat_feature_matrix(snaps: np.ndarray) -> np.ndarray:
    """Stack statistical features over all snapshots: (n, 7) float32."""
    return np.array(
        [stat_features(snaps[i]) for i in range(snaps.shape[0])],
        dtype=np.float32,
    )


def envelope_band_energy(x: np.ndarray, fs: int = FS, lo: float = 500.0,
                         hi: float = 5000.0) -> float:
    """Variance of the Hilbert envelope inside a resonance band [lo, hi] Hz."""
    sos = signal.butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    xf = signal.sosfiltfilt(sos, x)
    env = np.abs(signal.hilbert(xf))
    return float(np.var(env))


def spectral_kurtosis(x: np.ndarray, fs: int = FS) -> float:
    """Kurtosis of the Welch power spectral density (impulsiveness proxy)."""
    f, Pxx = signal.welch(x, fs=fs, nperseg=2048)
    return float(stats.kurtosis(Pxx[5:], fisher=True))


def damage_indicator_matrix(snaps: np.ndarray) -> np.ndarray:
    """Per-snapshot physics damage indicators: (n, 2) float32.

    Column 0 is the envelope-band energy, column 1 the spectral kurtosis.
    """
    out = np.zeros((snaps.shape[0], 2), dtype=np.float32)
    for i in range(snaps.shape[0]):
        out[i, 0] = envelope_band_energy(snaps[i])
        out[i, 1] = spectral_kurtosis(snaps[i])
    return out
