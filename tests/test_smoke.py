"""End-to-end smoke test on synthetic data.

Verifies that discovery, dataset construction, the FMEA graph and all four
ablation modes run and return sane shapes. Not a correctness test of the
science; it guards against import and shape regressions.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lg_fmea_kg import (
    FMEAGraph,
    build_dataset,
    discover_bearings,
    evaluate,
)
from lg_fmea_kg.synth import generate

MODES = ["STAT", "STAT_KG", "STAT_LG", "STAT_LG_KG"]


def _build_blocks(tmp):
    generate(tmp, n_per_bearing=20)
    paths = discover_bearings(tmp, verbose=False)
    assert len(paths) == 5, paths
    return build_dataset(paths, subset="wc2", verbose=False)


def test_discovery_and_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        blocks = _build_blocks(tmp)
        assert len(blocks) == 5
        for b in blocks:
            assert b["stat"].shape[1] == 7
            assert b["lg"].shape[1] == 4
            assert b["soft_zone"].shape[1] == 3
            assert b["y"].shape[0] == b["stat"].shape[0]


def test_all_modes_run():
    with tempfile.TemporaryDirectory() as tmp:
        blocks = _build_blocks(tmp)
        kg = FMEAGraph(dim=8, seed=0)
        for mode in MODES:
            res = evaluate(mode, kg, blocks, seed=0)
            assert 0.0 <= res["f1"] <= 1.0
            assert 0.0 <= res["f1_rare"] <= 1.0
            assert res["n_feats"] > 0


def test_scarcity_option():
    with tempfile.TemporaryDirectory() as tmp:
        blocks = _build_blocks(tmp)
        kg = FMEAGraph(dim=8, seed=0)
        res = evaluate("STAT_LG_KG", kg, blocks, seed=0,
                       drop_rare_frac=0.1, rare_class=2)
        assert np.isfinite(res["f1"])
