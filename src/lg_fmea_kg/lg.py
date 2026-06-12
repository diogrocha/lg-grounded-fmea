"""Landau-Ginzburg order parameter layer.

The physics damage indicators are reduced to a scalar damage coordinate
`d(t)`, which drives a Landau-Ginzburg control parameter `a(t) = a0 -
alpha * d`. Below the transition (a < 0) a non-zero order parameter
amplitude |psi| emerges, |psi|^2 = -a / (2 b), giving a smooth physical
readout of degradation severity. Thresholds eps_h and eps_c partition the
trajectory into HEALTHY / BUFFER / CRITICAL zones, and a smooth (soft)
version of that partition is used to ground the FMEA knowledge graph.
"""

from __future__ import annotations

import numpy as np


class LGOrderParameter:
    """Maps physics damage indicators to a Landau-Ginzburg order parameter."""

    def __init__(self, a0: float = 1.0, alpha: float = 0.07, b: float = 1.0,
                 eps_h: float = 0.5, eps_c: float = 2.5):
        self.a0, self.alpha, self.b = a0, alpha, b
        self.eps_h, self.eps_c = eps_h, eps_c
        self._mu = None
        self._sd = None

    def fit_healthy(self, dmg: np.ndarray, healthy_idx) -> "LGOrderParameter":
        """Calibrate the healthy reference from an early-life slice."""
        ref = dmg[healthy_idx]
        self._mu = ref.mean(axis=0)
        self._sd = ref.std(axis=0) + 1e-9
        return self

    def transform(self, dmg: np.ndarray) -> np.ndarray:
        """Return (n, 4) features: [psi, a, d, psi].

        The damage coordinate `d` is a log-domain standardised deviation
        from the healthy reference, clipped at zero and smoothed with a
        5-point moving average.
        """
        z = (
            np.log1p(np.clip(dmg, 0, None))
            - np.log1p(np.clip(self._mu, 0, None))
        ) / (np.log1p(self._sd) + 1e-9)
        d_raw = np.clip(z.mean(axis=1), 0.0, None)
        d = (
            np.convolve(d_raw, np.ones(5) / 5, mode="same")
            if len(d_raw) >= 5
            else d_raw
        )
        a = self.a0 - self.alpha * d
        psi2 = np.where(a < 0, -a / (2 * self.b), 0.0)
        psi = np.sqrt(psi2)
        return np.stack([psi, a, d, psi], axis=1).astype(np.float32)

    def zones(self, lg_feats: np.ndarray) -> np.ndarray:
        """Hard zone labels: 0 HEALTHY, 1 BUFFER, 2 CRITICAL."""
        amp = lg_feats[:, 3]
        z = np.zeros(amp.shape[0], dtype=np.int64)
        z[amp >= self.eps_h] = 1
        z[amp >= self.eps_c] = 2
        return z


def lg_to_soft_zones(lg_feats: np.ndarray, eps_h: float, eps_c: float) -> np.ndarray:
    """Smooth (probabilistic) HEALTHY / BUFFER / CRITICAL membership: (n, 3)."""
    amp = lg_feats[:, 3]
    t_h = 1 / (1 + np.exp(-(amp - eps_h) * 6.0))
    t_c = 1 / (1 + np.exp(-(amp - eps_c) * 4.0))
    p_h = 1 - t_h
    p_c = t_c
    p_b = np.clip(t_h - t_c, 0, 1)
    s = np.stack([p_h, p_b, p_c], axis=1)
    return (s / (s.sum(axis=1, keepdims=True) + 1e-9)).astype(np.float32)
