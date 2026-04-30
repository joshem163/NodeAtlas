"""
NSA Benchmark Runner
====================
Computes NSA scores (FSS/TSS/LSS) and baseline model accuracies
on real benchmark datasets. Outputs a single CSV with all results.

Models: MLP, LP, GCN, H2GCN, GPS, SGFormer, APPNP, SAGE, SGC

NSA paper settings (Section 4.1 / Appendix A.3):
  - NSA scores: normalized macro-accuracy, computed once per dataset
    (3 sampling seeds, 5-fold stratified CV)
  - Model performance: micro-accuracy on stratified 60/20/20 split
  - 10 runs (seeds 42-51) for model evaluation
  - Grid search: lr in {0.01,0.005}, wd in {0,5e-4,5e-3}, dropout in {0.3,0.5}
  - 500 epochs, patience 50

Usage:
    python run_baselines.py
    python run_baselines.py --datasets cora cornell
    python run_baselines.py --models MLP GCN H2GCN --runs 5
    python run_baselines.py --device cuda:0 --out results.csv
    python run_baselines.py --nsa-only          # NSA scores only, skip models
    python run_baselines.py --models-only       # models only, skip NSA
"""

import argparse
import csv
import os
import time
import warnings

import numpy as np
import torch
from torch_geometric.data import Data

from benchmarks import (
    ALL_DATASETS, DATASET_INFO, dataset_category,
    load_dataset as load_unified,
)
from descriptors import NodeDiagnostic
from models import (
    MODEL_NAMES, HIDDEN, GPS_RWSE_DIM,
    LR_GRID, WD_GRID, DROPOUT_GRID, LP_ALPHA_GRID,
    set_seed, stratified_split,
    build_adj, compute_rwse, h2gcn_precompute_adj, row_normalize_features,
    grid_search_neural, grid_search_lp,
)

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ #
#  Data loading
# ------------------------------------------------------------------ #

def load_data(name):
    """Load dataset -> (PyG Data, numpy arrays, num_classes)."""
    edge_index, X, y, num_classes = load_unified(name)
    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
    )
    return data, edge_index, X, y, num_classes


# ------------------------------------------------------------------ #
#  Main experiment loop
# ------------------------------------------------------------------ #

def run_experiment(
    datasets=None, models=None, runs=10, start_seed=42,
    out_csv="results.csv", device="cpu", hidden=HIDDEN,
    verbose=True,
    run_nsa=True, run_models=True,
    # NSA hyperparameters
    nsa_r=32, nsa_T=4, nsa_K=3, nsa_n_splits=5, nsa_n_seeds=3,
):
    if datasets is None:
        datasets = ALL_DATASETS
    if models is None:
        models = MODEL_NAMES

    sep = "=" * 80

    print(sep)
    print(f"  NSA Benchmark Runner")
    print(f"  NSA: {'ON' if run_nsa else 'OFF'}  "
          f"Models: {', '.join(models) if run_models else 'OFF'}")
    print(f"  datasets={len(datasets)}  runs={runs}  "
          f"seeds={start_seed}-{start_seed + runs - 1}")
    if run_models:
        total_grid = len(LR_GRID) * len(WD_GRID) * len(DROPOUT_GRID)
        print(f"  grid per neural model: {total_grid} combos  |  "
              f"LP alphas: {len(LP_ALPHA_GRID)}")
    print(f"  hidden={hidden}  device={device}")
    print(f"  NSA metric: normalized macro-accuracy  "
          f"Model metric: micro-accuracy")
    print(sep)

    # ---- CSV fields ----
    csv_fields = ["dataset", "category", "N", "E", "d", "C",
                  "h_edge", "h_node"]
    if run_nsa:
        csv_fields += ["FSS", "TSS", "LSS",
                       "FSS_std", "TSS_std", "LSS_std",
                       "nsa_time_s"]
    if run_models:
        for m in models:
            csv_fields += [f"{m}_mean", f"{m}_std"]

    all_results = []
    csv_opened = False

    for d_idx, ds in enumerate(datasets, 1):
        print(f"\n{'-' * 60}")
        print(f"[{d_idx}/{len(datasets)}] {ds}")

        try:
            data, ei_np, X_np, y_np, num_cls = load_data(ds)
        except Exception as e:
            print(f"  !! Load failed: {e}")
            continue

        N = data.num_nodes
        E = data.edge_index.size(1)
        d = data.x.size(1)
        cat = dataset_category(ds)
        info = DATASET_INFO.get(ds.lower().replace('_', '-'))
        print(f"  N={N:,}  E={E:,}  d={d}  C={num_cls}  cat={cat}")

        row = {
            "dataset": ds, "category": cat,
            "N": N, "E": E, "d": d, "C": num_cls,
            "h_edge": round(info[4], 3) if info else np.nan,
            "h_node": round(info[5], 3) if info else np.nan,
        }

        # ---- NSA scores (once per dataset, using seed=start_seed split) ----
        if run_nsa:
            set_seed(start_seed)
            tr_nsa, va_nsa, te_nsa = stratified_split(
                data.y.cpu(), seed=start_seed)
            train_mask = np.zeros(N, dtype=bool)
            train_mask[tr_nsa.numpy()] = True

            try:
                t0_nsa = time.time()
                diag = NodeDiagnostic(
                    r=nsa_r, T=nsa_T, K=nsa_K,
                    m=(64, 64, 64), n_splits=nsa_n_splits,
                    n_seeds=nsa_n_seeds, metric='macro')
                res = diag.fit(ei_np, X_np, y_np, train_mask,
                               verbose=verbose)
                nsa_time = time.time() - t0_nsa

                row["FSS"] = round(res['FSS'], 4)
                row["TSS"] = round(res['TSS'], 4)
                row["LSS"] = round(res['LSS'], 4)
                row["FSS_std"] = round(res['FSS_std'], 4)
                row["TSS_std"] = round(res['TSS_std'], 4)
                row["LSS_std"] = round(res['LSS_std'], 4)
                row["nsa_time_s"] = round(nsa_time, 1)

                print(f"  NSA: FSS={res['FSS']:.3f}  "
                      f"TSS={res['TSS']:.3f}  "
                      f"LSS={res['LSS']:.3f}  "
                      f"({nsa_time:.1f}s)")
            except Exception as e:
                print(f"  NSA FAILED: {e}")
                for k in ("FSS", "TSS", "LSS", "FSS_std", "TSS_std",
                          "LSS_std", "nsa_time_s"):
                    row[k] = np.nan

        # ---- Precompute model structures (once per dataset) ----
        precomp = {}
        if run_models:
            adj = build_adj(data.edge_index, N)
            if "GPS" in models:
                t0 = time.time()
                precomp["rwse"] = compute_rwse(adj, N, walk_length=GPS_RWSE_DIM)
                if verbose:
                    print(f"  RWSE done ({time.time() - t0:.1f}s)")
            if "H2GCN" in models:
                t0 = time.time()
                precomp["a1"], precomp["a2"] = h2gcn_precompute_adj(
                    data.edge_index, N, device=device)
                if verbose:
                    print(f"  H2GCN adj done ({time.time() - t0:.1f}s)")
            if any(m in models for m in ("GCN", "H2GCN", "SAGE")):
                precomp["x_normed"] = row_normalize_features(data.x)

        # ---- Model runs (micro-accuracy, multiple seeds) ----
        model_runs = {m: [] for m in models}
        if run_models:
            t0_models = time.time()
            for run in range(runs):
                seed = start_seed + run
                set_seed(seed)
                tr, va, te = stratified_split(data.y.cpu(), seed=seed)

                for mname in models:
                    try:
                        if mname == "LP":
                            sc, hp = grid_search_lp(
                                data, tr, va, te, num_cls, "acc")
                        else:
                            sc, hp = grid_search_neural(
                                mname, data, tr, va, te, num_cls,
                                precomp, "acc", device, hidden)
                        model_runs[mname].append(sc)
                        if verbose:
                            print(f"    run {run} [{mname}]: "
                                  f"{sc * 100:.2f}  hp={hp}")
                    except Exception as e:
                        print(f"    run {run} [{mname}] FAILED: {e}")

            elapsed_models = time.time() - t0_models

            for mname in models:
                vals = [v for v in model_runs[mname] if not np.isnan(v)]
                if vals:
                    mn = np.mean(vals) * 100
                    sd = np.std(vals) * 100
                    row[f"{mname}_mean"] = round(mn, 2)
                    row[f"{mname}_std"] = round(sd, 2)
                    print(f"  [{mname}] {mn:.2f} +/- {sd:.2f}  "
                          f"({len(vals)} runs)")
                else:
                    row[f"{mname}_mean"] = np.nan
                    row[f"{mname}_std"] = np.nan
                    print(f"  [{mname}] ALL FAILED")

            print(f"  Model time: {elapsed_models:.1f}s")

        all_results.append(row)

        # ---- Incremental CSV save ----
        if out_csv:
            mode = "w" if not csv_opened else "a"
            with open(out_csv, mode, newline="") as f:
                w = csv.DictWriter(f, fieldnames=csv_fields)
                if not csv_opened:
                    w.writeheader()
                    csv_opened = True
                w.writerow({k: row.get(k, "") for k in csv_fields})

    # ---- Summary table ----
    _print_summary(all_results, models, run_nsa, run_models)

    if out_csv and all_results:
        print(f"\n  Results saved: {os.path.abspath(out_csv)}")

    return all_results


def _print_summary(results, models, run_nsa, run_models):
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  SUMMARY")
    print(sep)

    hdr = f"  {'Dataset':<22} {'Cat':<15}"
    if run_nsa:
        hdr += f"  {'FSS':>6} {'TSS':>6} {'LSS':>6}"
    if run_models:
        for m in models:
            hdr += f"  {m:>9}"
    print(hdr)
    print(f"  {'-' * 22} {'-' * 15}" +
          ("  " + "-" * 22 if run_nsa else "") +
          ("  " + "-" * 9 * len(models) if run_models else ""))

    prev = None
    for r in results:
        if r["category"] != prev:
            if prev is not None:
                print()
            prev = r["category"]
        line = f"  {r['dataset']:<22} {r['category']:<15}"
        if run_nsa:
            for key in ("FSS", "TSS", "LSS"):
                v = r.get(key, np.nan)
                line += f"  {v:>6.3f}" if not np.isnan(v) else f"  {'N/A':>6}"
        if run_models:
            for m in models:
                mn = r.get(f"{m}_mean", np.nan)
                if np.isnan(mn):
                    line += f"  {'N/A':>9}"
                else:
                    line += f"  {mn:>9.2f}"
        print(line)
    print(sep)


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def main():
    p = argparse.ArgumentParser(
        description="NSA benchmark runner: NSA scores + baseline models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Datasets to run (default: all 18)")
    p.add_argument("--models", nargs="+", default=None,
                   choices=MODEL_NAMES,
                   help="Models to run (default: all 9)")
    p.add_argument("--runs", type=int, default=10,
                   help="Number of runs per dataset")
    p.add_argument("--start_seed", type=int, default=42)
    p.add_argument("--hidden", type=int, default=HIDDEN)
    p.add_argument("--out", default="results.csv",
                   help="Output CSV path")
    p.add_argument("--device", default="cpu")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--nsa-only", action="store_true",
                   help="Compute NSA scores only, skip models")
    p.add_argument("--models-only", action="store_true",
                   help="Run models only, skip NSA scores")
    a = p.parse_args()

    dev = a.device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable -> using CPU")
        dev = "cpu"

    do_nsa = not a.models_only
    do_models = not a.nsa_only

    run_experiment(
        datasets=a.datasets,
        models=a.models,
        runs=a.runs,
        start_seed=a.start_seed,
        out_csv=a.out,
        device=dev,
        hidden=a.hidden,
        verbose=not a.quiet,
        run_nsa=do_nsa,
        run_models=do_models,
    )


if __name__ == "__main__":
    main()
