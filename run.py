#!/usr/bin/env python3
"""Reproduce the XJTU-SY experiments from a single command line.

Examples
--------
Smoke test on synthetic data (no download required)::

    python run.py --demo

Full reproduction on the real dataset::

    python run.py --data-dir /path/to/XJTU-SY --experiment all

Individual experiments::

    python run.py --data-dir /path/to/XJTU-SY --experiment main
    python run.py --data-dir /path/to/XJTU-SY --experiment scarcity
    python run.py --data-dir /path/to/XJTU-SY --experiment figures
    python run.py --data-dir /path/to/XJTU-SY --experiment cross

The script writes ``results/results_real.json`` and, for the figures
experiment, ``results/fig_trajectory.png`` and ``results/fig_scarcity.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lg_fmea_kg import (  # noqa: E402
    FMEAGraph,
    build_dataset,
    discover_bearings,
    evaluate,
)
from lg_fmea_kg.config import CLASS_NAMES  # noqa: E402
from lg_fmea_kg.figures import trajectory_figure, scarcity_figure  # noqa: E402
from lg_fmea_kg.models import make_features  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402

MODES = ["STAT", "STAT_KG", "STAT_LG", "STAT_LG_KG"]


def run_main(blocks, kg, seeds):
    print("\n" + "=" * 70)
    print(f"{'Configuration':<20}{'#feats':>8}{'Precision':>12}{'Recall':>10}{'F1':>14}")
    print("=" * 70)
    results_main = {}
    for mode in MODES:
        rs = [evaluate(mode, kg, blocks, seed=s) for s in range(seeds)]
        p = np.mean([r["precision"] for r in rs])
        rc = np.mean([r["recall"] for r in rs])
        f = np.mean([r["f1"] for r in rs])
        sd = np.std([r["f1"] for r in rs])
        results_main[mode] = {
            "precision": float(p), "recall": float(rc),
            "f1": float(f), "f1_sd": float(sd),
            "n_feats": int(rs[0]["n_feats"]),
        }
        print(f"{mode:<20}{rs[0]['n_feats']:>8}{p:>12.4f}{rc:>10.4f}{f:>9.4f}±{sd:.3f}")

    print("\nPer-class breakdown of the proposed configuration (STAT_LG_KG):")
    rs = [evaluate("STAT_LG_KG", kg, blocks, seed=s) for s in range(seeds)]
    yte_all = np.concatenate([r["yte"] for r in rs])
    yp_all = np.concatenate([r["yp"] for r in rs])
    print(classification_report(yte_all, yp_all, target_names=CLASS_NAMES,
                                digits=3, zero_division=0))
    return results_main


def run_scarcity(blocks, kg, seeds, fractions):
    results = {m: {"f1_rare": [], "f1_rare_sd": [],
                   "f1_macro": [], "f1_macro_sd": []} for m in MODES}
    for frac in fractions:
        print(f"\n--- scarcity fraction = {frac:.2f} ---")
        for m in MODES:
            rs = [evaluate(m, kg, blocks, seed=s, drop_rare_frac=frac, rare_class=2)
                  for s in range(seeds)]
            f1r = [r["f1_rare"] for r in rs]
            f1m = [r["f1"] for r in rs]
            results[m]["f1_rare"].append(float(np.mean(f1r)))
            results[m]["f1_rare_sd"].append(float(np.std(f1r)))
            results[m]["f1_macro"].append(float(np.mean(f1m)))
            results[m]["f1_macro_sd"].append(float(np.std(f1m)))
            print(f"  {m:<14} F1_rare={np.mean(f1r):.3f}±{np.std(f1r):.3f}  "
                  f"F1_macro={np.mean(f1m):.3f}±{np.std(f1m):.3f}")
    return results


def run_cross(bearing_paths, blocks, kg, lg_kwargs, seeds):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    have_wc1 = any(k.startswith("Bearing1_") for k in bearing_paths)
    have_wc3 = any(k.startswith("Bearing3_") for k in bearing_paths)
    print(f"WC1 available: {have_wc1}   WC3 available: {have_wc3}")
    if not (have_wc1 or have_wc3):
        print("Skipping cross-condition (no WC1 or WC3 data).")
        return None

    blocks_wc1 = build_dataset(bearing_paths, subset="wc1", lg_kwargs=lg_kwargs) if have_wc1 else None
    blocks_wc3 = build_dataset(bearing_paths, subset="wc3", lg_kwargs=lg_kwargs) if have_wc3 else None

    def cross_eval(train_blocks, test_blocks, mode, seed=0):
        Xtr, ytr = make_features(train_blocks, mode, kg)
        Xte, yte = make_features(test_blocks, mode, kg)
        rng = np.random.default_rng(seed + 2000)
        classes, counts = np.unique(ytr, return_counts=True)
        target = counts.max()
        Xb, yb = [], []
        for c, ct in zip(classes, counts):
            idx = np.where(ytr == c)[0]
            sel = rng.choice(idx, size=target, replace=(ct < target))
            Xb.append(Xtr[sel]); yb.append(ytr[sel])
        Xtr_b = np.concatenate(Xb); ytr_b = np.concatenate(yb)
        perm = rng.permutation(len(ytr_b))
        Xtr_b, ytr_b = Xtr_b[perm], ytr_b[perm]
        sc = StandardScaler().fit(Xtr_b)
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                            random_state=seed, early_stopping=True,
                            n_iter_no_change=15)
        clf.fit(sc.transform(Xtr_b), ytr_b)
        yp = clf.predict(sc.transform(Xte))
        return f1_score(yte, yp, average="macro", zero_division=0)

    print("\nCross-condition generalisation (train WC2)")
    results_cross = {}
    for m in MODES:
        d = {}
        row = f"{m:<14}"
        if blocks_wc1:
            v = float(np.mean([cross_eval(blocks, blocks_wc1, m, s) for s in range(seeds)]))
            d["to_wc1"] = v; row += f"  ->WC1 {v:.4f}"
        if blocks_wc3:
            v = float(np.mean([cross_eval(blocks, blocks_wc3, m, s) for s in range(seeds)]))
            d["to_wc3"] = v; row += f"  ->WC3 {v:.4f}"
        results_cross[m] = d
        print(row)
    return results_cross


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Root folder containing the XJTU-SY BearingX_Y directories.")
    ap.add_argument("--demo", action="store_true",
                    help="Generate and use a small synthetic dataset (smoke test).")
    ap.add_argument("--experiment", choices=["main", "scarcity", "cross", "figures", "all"],
                    default="all")
    ap.add_argument("--subset", default="wc2", help="Working-condition subset for the main run.")
    ap.add_argument("--seeds", type=int, default=5, help="Number of random seeds per configuration.")
    ap.add_argument("--out", type=str, default="results", help="Output directory.")
    ap.add_argument("--eps-h", type=float, default=None, help="Override LG healthy threshold.")
    ap.add_argument("--eps-c", type=float, default=None, help="Override LG critical threshold.")
    args = ap.parse_args()

    lg_kwargs = {}
    if args.eps_h is not None:
        lg_kwargs["eps_h"] = args.eps_h
    if args.eps_c is not None:
        lg_kwargs["eps_c"] = args.eps_c

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp = None
    if args.demo:
        from lg_fmea_kg.synth import generate
        tmp = tempfile.mkdtemp(prefix="xjtu_synth_")
        print(f"Generating synthetic dataset in {tmp} (smoke test only, not real results)...")
        generate(tmp)
        data_dir = tmp
        if args.seeds > 3:
            args.seeds = 3  # the synthetic set is tiny
    elif args.data_dir:
        data_dir = args.data_dir
    else:
        ap.error("provide --data-dir /path/to/XJTU-SY or use --demo")

    bearing_paths = discover_bearings(data_dir)
    if not bearing_paths:
        ap.error(f"no bearing folders found under {data_dir}")

    t0 = time.time()
    print(f"\nBuilding dataset for subset '{args.subset}'...")
    blocks = build_dataset(bearing_paths, subset=args.subset, lg_kwargs=lg_kwargs)
    print(f"Dataset built in {(time.time() - t0) / 60:.1f} min")

    print("\nLabel distribution per bearing:")
    for b in blocks:
        cls, cnt = np.unique(b["y"], return_counts=True)
        cls_str = ", ".join(f"y={c}:{n}" for c, n in zip(cls, cnt))
        print(f"  {b['fault']:>16s}  N={b['y'].shape[0]:4d}  ({cls_str})")

    kg = FMEAGraph(dim=8, seed=0)
    print(f"\nFMEA graph: {len(kg.nodes)} nodes, "
          f"{int(kg.adj.sum() / 2)} undirected edges, dim={kg.dim}")

    out = {"subset": args.subset, "lg_kwargs": lg_kwargs,
           "n_bearings_detected": len(bearing_paths), "demo": bool(args.demo)}

    fractions = [1.0, 0.5, 0.25, 0.10, 0.05]

    if args.experiment in ("main", "all"):
        out["main"] = run_main(blocks, kg, args.seeds)

    if args.experiment in ("scarcity", "all"):
        sc = run_scarcity(blocks, kg, args.seeds, fractions)
        out["scarcity"] = {"fractions": fractions, "modes": MODES, "results": sc}

    if args.experiment in ("cross", "all"):
        out["cross_condition"] = run_cross(bearing_paths, blocks, kg, lg_kwargs, args.seeds)

    if args.experiment in ("figures", "all"):
        print("\nRendering figures...")
        trajectory_figure(blocks, kg, out_dir / "fig_trajectory.png")
        if "scarcity" not in out:
            sc = run_scarcity(blocks, kg, args.seeds, fractions)
            out["scarcity"] = {"fractions": fractions, "modes": MODES, "results": sc}
        scarcity_figure(fractions, MODES, out["scarcity"]["results"],
                        out_dir / "fig_scarcity.png")
        print(f"  wrote {out_dir / 'fig_trajectory.png'}")
        print(f"  wrote {out_dir / 'fig_scarcity.png'}")

    res_path = out_dir / "results_real.json"
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nResults written to {res_path}")

    if tmp:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
