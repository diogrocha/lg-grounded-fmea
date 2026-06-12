"""LG-grounded FMEA-KG for hybrid fault detection on XJTU-SY bearings.

Reference implementation for the paper:

    "Grounding Symbolic Failure-Mode Knowledge with Physics:
     A Landau-Ginzburg Layer for Hybrid FMEA-Aware Fault Detection"

The package decouples the scientific pipeline from any notebook or Colab
plumbing so that the experiments can be reproduced locally from a single
command line entry point. See the project README for instructions.
"""

from .config import FS, SNAPSHOT_SEC, N_SAMPLES, BEARING_LABELS, FAULT_TO_CLASS
from .data import BearingRun, discover_bearings, make_runs, build_lifetime
from .features import (
    stat_features,
    stat_feature_matrix,
    damage_indicator_matrix,
    envelope_band_energy,
    spectral_kurtosis,
)
from .lg import LGOrderParameter, lg_to_soft_zones
from .kg import FMEAGraph
from .dataset import assign_labels, build_dataset
from .models import make_features, evaluate

__version__ = "1.0.0"

__all__ = [
    "FS",
    "SNAPSHOT_SEC",
    "N_SAMPLES",
    "BEARING_LABELS",
    "FAULT_TO_CLASS",
    "BearingRun",
    "discover_bearings",
    "make_runs",
    "build_lifetime",
    "stat_features",
    "stat_feature_matrix",
    "damage_indicator_matrix",
    "envelope_band_energy",
    "spectral_kurtosis",
    "LGOrderParameter",
    "lg_to_soft_zones",
    "FMEAGraph",
    "assign_labels",
    "build_dataset",
    "make_features",
    "evaluate",
    "__version__",
]
