# LG-FMEA-KG

Reference implementation and reproducibility code for the paper:

> **Grounding Symbolic Failure-Mode Knowledge with Physics:
> A Landau-Ginzburg Layer for Hybrid FMEA-Aware Fault Detection**
> Diogo Rocha, Rui Pinto, Gil Goncalves. *Work in progress, 2026.*

The method couples a physics-based **Landau-Ginzburg (LG) order parameter**,
which turns vibration damage indicators into a smooth degradation readout,
with a symbolic **FMEA knowledge graph**. The LG readout continuously grounds
the graph's severity and frequency nodes, so the symbolic failure-mode prior
is modulated by the physical state of the bearing rather than being static.
The code reproduces the four-way ablation, the rare-class scarcity stress
test, the cross-condition generalisation study and the paper figures on the
public **XJTU-SY** run-to-failure bearing dataset.

> Status: this repository accompanies a work-in-progress submission. Venue,
> DOI and final numbers will be added on acceptance.

## Method at a glance

The pipeline compares four configurations of increasing structure:

| Configuration | Description |
|---------------|-------------|
| `STAT`        | Statistical vibration features only (data-only baseline). |
| `STAT_KG`     | Statistical features plus static FMEA graph embeddings (ungrounded knowledge-graph baseline, in the spirit of FKGCN). |
| `STAT_LG`     | Statistical features plus Landau-Ginzburg features. |
| `STAT_LG_KG`  | Statistical and LG features plus physics-grounded FMEA embeddings gated by LG-to-mode similarity (**proposed**). |

Per snapshot the pipeline computes statistical descriptors and two
physics damage indicators (envelope-band energy and spectral kurtosis).
The damage indicators drive the LG control parameter `a(t) = a0 - alpha * d(t)`;
below the transition a non-zero amplitude `|psi|` emerges and is thresholded
into HEALTHY / BUFFER / CRITICAL zones. A smooth version of those zones grounds
the FMEA graph used by the proposed configuration.

## Installation

Python 3.9 or newer.

```bash
git clone https://github.com/your-username/lg-fmea-kg.git
cd lg-fmea-kg
pip install -r requirements.txt
# optional, to import the package from anywhere:
pip install -e .
```

## Quick start (no dataset needed)

A synthetic smoke test runs the entire pipeline end to end in under a minute.
The synthetic signals are not physically meaningful and exist only to verify
that the code executes and the shapes are consistent. Do not report these
numbers.

```bash
python run.py --demo
```

## Reproducing the paper

1. Obtain the XJTU-SY dataset and extract it. See [`data/README.md`](data/README.md).
2. Run the experiments, pointing at the extracted folder:

```bash
# everything: main ablation, scarcity, cross-condition, figures
python run.py --data-dir /path/to/XJTU-SY --experiment all

# or one experiment at a time
python run.py --data-dir /path/to/XJTU-SY --experiment main
python run.py --data-dir /path/to/XJTU-SY --experiment scarcity
python run.py --data-dir /path/to/XJTU-SY --experiment cross
python run.py --data-dir /path/to/XJTU-SY --experiment figures
```

Results are written to `results/results_real.json`. The figures experiment
also writes `results/fig_trajectory.png` and `results/fig_scarcity.png`.

### Useful options

| Flag | Default | Meaning |
|------|---------|---------|
| `--subset` | `wc2` | Working-condition subset for the main run (`wc1`, `wc2`, `wc3`, `all`). |
| `--seeds`  | `5`   | Random seeds averaged per configuration. |
| `--eps-h` / `--eps-c` | calibrated | Override the LG healthy / critical thresholds. |
| `--out`    | `results` | Output directory. |

### Calibrating the LG thresholds

The default thresholds (`eps_h=0.5`, `eps_c=2.5`) target roughly 50 to 70
percent HEALTHY, 10 to 25 percent BUFFER and 15 to 30 percent CRITICAL
snapshots. If a working condition is dominated by one zone, override the
thresholds, for example `--eps-h 0.3 --eps-c 1.5`.

## Programmatic use

```python
from lg_fmea_kg import discover_bearings, build_dataset, FMEAGraph, evaluate

paths  = discover_bearings("/path/to/XJTU-SY")
blocks = build_dataset(paths, subset="wc2")
kg     = FMEAGraph(dim=8, seed=0)

res = evaluate("STAT_LG_KG", kg, blocks, seed=0)
print(res["f1"], res["f1_rare"])
```

## Repository layout

```
lg-fmea-kg/
  run.py                     One-command experiment runner (CLI).
  src/lg_fmea_kg/
    config.py                Acquisition constants and bearing ground-truth labels.
    data.py                  Dataset discovery and snapshot loading (no Colab needed).
    features.py              Statistical and physics vibration features.
    lg.py                    Landau-Ginzburg order parameter and soft zones.
    kg.py                    FMEA knowledge graph and grounded embeddings.
    dataset.py               Per-bearing feature blocks and labelling.
    models.py                Feature fusion for the four modes and evaluation.
    figures.py               Trajectory and scarcity figure generation.
    synth.py                 Synthetic data generator for the smoke test.
  notebooks/                 Colab notebook (mounts Drive, extracts rar, runs).
  data/                      Dataset instructions (data itself is not committed).
  results/                   Experiment outputs (git-ignored).
  tests/                     Smoke test.
```

## Reproducibility notes

- All randomness is seeded. Per-configuration scores are averaged over
  `--seeds` runs and reported with standard deviation.
- Training uses class-balanced resampling and a two-layer MLP with early
  stopping. The scarcity test removes a controlled fraction of the rare
  class (`outer_race`) from training only.
- The exact package versions used are pinned in `requirements.txt`. Minor
  differences in BLAS or scikit-learn versions can shift the last decimal.

## Dataset and references

XJTU-SY run-to-failure bearing dataset:

> B. Wang, Y. Lei, N. Li, N. Li, "A Hybrid Prognostics Approach for
> Estimating Remaining Useful Life of Rolling Element Bearings,"
> *IEEE Transactions on Reliability*, 69(1), 401-412, 2020.

## Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)).
The BibTeX below is a placeholder to update on acceptance:

```bibtex
@inproceedings{rocha2026lgfmeakg,
  title     = {Grounding Symbolic Failure-Mode Knowledge with Physics:
               A Landau-Ginzburg Layer for Hybrid FMEA-Aware Fault Detection},
  author    = {Rocha, Diogo and Pinto, Rui and Goncalves, Gil},
  year      = {2026},
  note      = {Work in progress}
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
