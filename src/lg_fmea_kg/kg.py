"""FMEA knowledge graph and its embeddings.

Nodes encode the bearing equipment, its four failure modes, and discrete
severity / frequency attributes drawn from a Failure Mode and Effects
Analysis. `static_failure_mode_embeddings` is a fixed one-hop propagation
of the symbolic graph. `lg_grounded_embeddings` instead injects the
physics-derived soft-zone membership into the severity / frequency nodes
before propagation, so the symbolic prior is continuously modulated by the
Landau-Ginzburg readout.
"""

from __future__ import annotations

import numpy as np


class FMEAGraph:
    SEV_NODES = ["sev_low", "sev_med", "sev_hi"]
    FREQ_NODES = ["frq_low", "frq_med", "frq_hi"]
    FAILURE_MODES = ["healthy", "inner_race", "outer_race", "cage_or_re"]

    def __init__(self, dim: int = 8, seed: int = 0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.nodes = (
            ["equipment_bearing"]
            + self.FAILURE_MODES
            + self.SEV_NODES
            + self.FREQ_NODES
        )
        self.idx = {n: i for i, n in enumerate(self.nodes)}

        E = self.rng.normal(0, 0.2, size=(len(self.nodes), dim)).astype(np.float32)
        for k, fm in enumerate(self.FAILURE_MODES):
            E[self.idx[fm], k] = 1.5
        for k, s in enumerate(self.SEV_NODES):
            E[self.idx[s], 4] = (k - 1) * 1.0
        for k, f in enumerate(self.FREQ_NODES):
            E[self.idx[f], 5] = (k - 1) * 1.0
        self.E = E

        self.adj = np.zeros((len(self.nodes), len(self.nodes)), dtype=np.float32)
        for fm in self.FAILURE_MODES:
            self._link("equipment_bearing", fm)
        self._link("inner_race", "sev_med")
        self._link("inner_race", "frq_med")
        self._link("outer_race", "sev_hi")
        self._link("outer_race", "frq_hi")
        self._link("cage_or_re", "sev_hi")
        self._link("cage_or_re", "frq_low")
        self._link("healthy", "sev_low")
        self._link("healthy", "frq_low")

    def _link(self, a: str, b: str) -> None:
        ia, ib = self.idx[a], self.idx[b]
        self.adj[ia, ib] = 1.0
        self.adj[ib, ia] = 1.0

    def static_failure_mode_embeddings(self) -> np.ndarray:
        """One-hop propagated embeddings for the four failure modes."""
        deg = self.adj.sum(axis=1, keepdims=True) + 1e-9
        H = (self.adj / deg) @ self.E + self.E
        return H[[self.idx[fm] for fm in self.FAILURE_MODES]]

    def lg_grounded_embeddings(self, soft: np.ndarray) -> np.ndarray:
        """Per-snapshot failure-mode embeddings grounded by soft LG zones.

        `soft` is the (n, 3) HEALTHY/BUFFER/CRITICAL membership. The three
        severity and three frequency node embeddings are replaced by a
        soft-zone-weighted combination of their basis vectors before the
        one-hop propagation, yielding an (n, n_modes, dim) tensor.
        """
        n = soft.shape[0]
        out = np.zeros((n, len(self.FAILURE_MODES), self.dim), dtype=np.float32)
        E = self.E.copy()
        sev_basis = E[[self.idx[s] for s in self.SEV_NODES]]
        frq_basis = E[[self.idx[f] for f in self.FREQ_NODES]]
        deg = self.adj.sum(axis=1, keepdims=True) + 1e-9
        norm_adj = self.adj / deg
        for i in range(n):
            E_i = E.copy()
            sev_i = soft[i] @ sev_basis
            frq_i = soft[i] @ frq_basis
            for s in self.SEV_NODES:
                E_i[self.idx[s]] = sev_i
            for f in self.FREQ_NODES:
                E_i[self.idx[f]] = frq_i
            H = norm_adj @ E_i + E_i
            out[i] = H[[self.idx[fm] for fm in self.FAILURE_MODES]]
        return out
