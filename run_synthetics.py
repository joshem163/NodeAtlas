"""
Synthetic Experiment Runner
============================
Experiments for the NSA paper:

  E1: Synthetic regime validation (Section 4.2, Table 1/8)
      24 controlled datasets (8 corners + 16 sweep of the 2x2x2 F/T/L cube).
      Verify NSA selectivity, calibration, compositionality, and scope.
      N=1500, C=3, model performance in micro-accuracy (mean over 10 seeds).

  E2: Real benchmark ranking prediction (Section 4.3)
      Compute (FSS, TSS, LSS) on training nodes; predict winning model
      family; compare against observed rankings over all 9 baselines.

  E3: Quantitative ranking prediction (Section 4.4, Table 4)
      Top-1 family prediction accuracy and regret over
      the full 18-dataset real benchmark suite.
"""

import numpy as np
from collections import Counter
from typing import Dict, Tuple, Optional

from descriptors import NodeDiagnostic
from synthetics import (generate_all_synthetic,
                         ALL_SYNTHETIC_DATASETS, EXPECTED_DOMINANT)
from models import run_all_models

import torch


# =============================================================
# UTILITIES
# =============================================================

def make_masks(num_nodes: int,
               y: np.ndarray,
               train_ratio: float = 0.6,
               val_ratio: float = 0.2,
               seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 60/20/20 split aligned with NSA paper Section 4.1."""
    rng = np.random.RandomState(seed)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask   = np.zeros(num_nodes, dtype=bool)
    test_mask  = np.zeros(num_nodes, dtype=bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        idx = rng.permutation(idx)
        n_tr  = int(len(idx) * train_ratio)
        n_val = int(len(idx) * val_ratio)
        train_mask[idx[:n_tr]]            = True
        val_mask[idx[n_tr:n_tr + n_val]] = True
        test_mask[idx[n_tr + n_val:]]    = True
    return train_mask, val_mask, test_mask


def run_nsa(edge_index: np.ndarray,
            X: np.ndarray,
            y: np.ndarray,
            train_mask: np.ndarray,
            verbose: bool = True) -> Dict:
    """Run NSA diagnostic (normalized macro-accuracy)."""
    diag = NodeDiagnostic(r=32, T=4, K=3, n_seeds=3, metric='macro')
    return diag.fit(edge_index, X, y, train_mask, verbose=verbose)


def print_results_table(results: Dict):
    """Print a compact results table for a dict of per-dataset results."""
    models = ['MLP', 'LP', 'APPNP', 'GCN', 'SAGE',
              'H2GCN', 'SGC', 'GPS', 'SGFormer']
    first = next(iter(results.values()), {})
    avail = [m for m in models
             if not np.isnan(first.get('models', {}).get(m, float('nan')))]

    header = f"{'Dataset':<20} {'FSS':>5} {'TSS':>5} {'LSS':>5}  "
    header += "  ".join(f"{m:>9}" for m in avail) + f"  {'Winner':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        d = res['diagnostic']
        m = res['models']
        valid = {k: v for k, v in m.items() if not np.isnan(v)}
        winner = max(valid, key=valid.get) if valid else '?'
        row = (f"{name:<20} {d['FSS']:>5.3f} {d['TSS']:>5.3f} "
               f"{d['LSS']:>5.3f}  ")
        row += "  ".join(f"{m.get(k, float('nan')):>9.3f}" for k in avail)
        row += f"  {winner:>9}"
        print(row)
    print("=" * len(header))


# =============================================================
# E1: SYNTHETIC REGIME VALIDATION
# =============================================================

def experiment_synthetic_regimes(seed: int = 42,
                                  num_nodes: int = 1500,
                                  num_classes: int = 3,
                                  n_runs: int = 10) -> Dict:
    """
    E1: Verify NSA selectivity, calibration, and scope on all 24
    controlled synthetic datasets (8 corners + 16 sweep).

    NSA scores: normalized macro-accuracy (0=chance, 1=perfect).
    Model performance: micro-accuracy, mean +/- std over n_runs seeds.
    """
    from models import MODEL_NAMES

    print("\n" + "=" * 60)
    print("E1: Synthetic Regime Validation (24 datasets: 8 corners + 16 sweep)")
    print(f"    n_runs={n_runs}, seeds={seed}-{seed + n_runs - 1}")
    print("=" * 60)

    datasets = generate_all_synthetic(
        num_nodes=num_nodes, num_classes=num_classes, seed=seed)

    all_results = {}
    for name, (edge_index, X, y) in datasets.items():
        n = X.shape[0]
        print(f"\n--- {name} (expected: {EXPECTED_DOMINANT[name]}) ---")

        # NSA diagnostic (once per dataset, using first seed's split)
        train_mask_nsa, _, _ = make_masks(n, y, seed=seed)
        diag = run_nsa(edge_index, X, y, train_mask_nsa)
        fss, tss, lss = diag['FSS'], diag['TSS'], diag['LSS']
        print(f"  NSA: FSS={fss:.3f}  TSS={tss:.3f}  LSS={lss:.3f}")

        # Observed dominant channels
        margin = 0.05
        active = []
        if fss > margin: active.append('F')
        if tss > margin: active.append('T')
        if lss > margin: active.append('L')
        observed = ''.join(active) if active else 'none'
        match = 'OK' if observed == EXPECTED_DOMINANT[name] else 'MISMATCH'
        print(f"  Observed: {observed}  Expected: "
              f"{EXPECTED_DOMINANT[name]}  [{match}]")

        # All 9 models x n_runs seeds (micro-accuracy)
        model_runs = {m: [] for m in MODEL_NAMES}
        for run in range(n_runs):
            s = seed + run
            tr_m, va_m, te_m = make_masks(n, y, seed=s)
            accs = run_all_models(
                edge_index, X, y, tr_m, va_m, te_m,
                seed=s, metric='acc')
            for m, a in accs.items():
                model_runs[m].append(a)
            print(f"    run {run} (seed={s}) done")

        # Aggregate mean +/- std
        model_mean = {}
        model_std = {}
        for m in MODEL_NAMES:
            vals = [v for v in model_runs[m] if not np.isnan(v)]
            if vals:
                model_mean[m] = float(np.mean(vals))
                model_std[m] = float(np.std(vals))
            else:
                model_mean[m] = float('nan')
                model_std[m] = float('nan')

        print(f"  Model scores (micro-acc, {n_runs} seeds):")
        for m in MODEL_NAMES:
            mn = model_mean[m]
            sd = model_std[m]
            if not np.isnan(mn):
                print(f"    {m}: {mn*100:.1f} +/- {sd*100:.1f}")

        all_results[name] = {
            'diagnostic': {'FSS': fss, 'TSS': tss, 'LSS': lss},
            'models': model_mean,
            'models_std': model_std,
            'expected': EXPECTED_DOMINANT[name],
            'observed': observed,
        }

    print_results_table(all_results)

    n_correct = sum(
        1 for r in all_results.values()
        if r['observed'] == r['expected']
    )
    print(f"\n  Diagnostic regime accuracy: {n_correct}/{len(all_results)} "
          f"datasets matched")
    return all_results


# =============================================================
# E2: REAL BENCHMARK RANKING PREDICTION
# =============================================================

def experiment_real_benchmarks(dataset_names: Optional[list] = None,
                                 seed: int = 42) -> Dict:
    """
    E2: Compute (FSS, TSS, LSS) on training nodes; predict winning model
    family; compare against observed rankings over all 9 baselines
    on the full 18-dataset suite.
    """
    from benchmarks import ALL_DATASETS, load_dataset

    print("\n" + "=" * 60)
    print("E2: Real Benchmark Ranking Prediction")
    print("=" * 60)

    if dataset_names is None:
        dataset_names = ALL_DATASETS

    all_results = {}
    for name in dataset_names:
        print(f"\n--- {name.upper()} ---")
        try:
            edge_index, X, y, num_classes = load_dataset(name)
            N = X.shape[0]
            train_mask, val_mask, test_mask = make_masks(N, y, seed=seed)
            print(f"  N={N}, d={X.shape[1]}, C={num_classes}, "
                  f"E={edge_index.shape[1]//2}")

            # NSA diagnostic
            diag = run_nsa(edge_index, X, y, train_mask)
            print(f"  NSA: FSS={diag['FSS']:.3f}  TSS={diag['TSS']:.3f}  "
                  f"LSS={diag['LSS']:.3f}")

            predicted = predict_family(diag['FSS'], diag['TSS'], diag['LSS'])
            print(f"  Predicted dominant family: {predicted}")

            # All 9 models (micro-accuracy)
            models = run_all_models(
                edge_index, X, y, train_mask, val_mask, test_mask,
                seed=seed, metric='acc')

            valid = {k: v for k, v in models.items() if not np.isnan(v)}
            observed = max(valid, key=valid.get) if valid else '?'
            print(f"  Observed winner: {observed}")
            print(f"  Model accs: " +
                  ", ".join(f"{k}={v:.3f}" for k, v in valid.items()))

            all_results[name] = {
                'diagnostic': {'FSS': diag['FSS'],
                                'TSS': diag['TSS'],
                                'LSS': diag['LSS']},
                'predicted':  predicted,
                'observed':   observed,
                'models':     models,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print_results_table(all_results)
    return all_results


# =============================================================
# E3: RANKING PREDICTION (Table 4)
# =============================================================

# Representative models per family (Table 4 in paper).
# "With GT": 4 families — MLP (F), APPNP (L), H2GCN (GNN), SGFormer (GT)
# "Classical": 3 families — MLP (F), APPNP (L), H2GCN (GNN)
FAMILY_MAP_GT = {
    'F': 'MLP', 'L': 'APPNP', 'GNN': 'H2GCN', 'GT': 'SGFormer',
}
FAMILY_MAP_CLASSICAL = {
    'F': 'MLP', 'L': 'APPNP', 'GNN': 'H2GCN',
}

# NSA zone -> family prediction mapping
# F-skewed -> F, L-skewed -> L, T-skewed -> GNN (classical) or GT (with GT)
ZONE_TO_FAMILY_GT = {'F': 'F', 'L': 'L', 'T': 'GT'}
ZONE_TO_FAMILY_CLASSICAL = {'F': 'F', 'L': 'L', 'T': 'GNN'}


def predict_family(fss: float, tss: float, lss: float,
                   epsilon: float = 0.10) -> str:
    """NSA zone-based family prediction (Appendix E.1).

    Activity threshold delta=0.05, margin threshold epsilon=0.10.
    Returns singleton zone ('F', 'L', 'T') if leading score exceeds
    next-highest active score by at least epsilon; otherwise 'mixed'.
    """
    delta = 0.05
    scores = {'F': fss, 'L': lss, 'T': tss}
    active = {k: v for k, v in scores.items() if v > delta}
    if not active:
        return 'desert'
    sorted_vals = sorted(active.values(), reverse=True)
    if len(sorted_vals) < 2 or sorted_vals[0] - sorted_vals[1] >= epsilon:
        return max(active, key=active.get)
    return 'mixed'


def _oracle_family(mean_accs: Dict[str, float],
                   family_map: Dict[str, str]) -> Tuple[str, float]:
    """Find the oracle (best) family given mean model accuracies."""
    best_fam, best_acc = None, -1.0
    for fam, model in family_map.items():
        acc = mean_accs.get(model, float('nan'))
        if not np.isnan(acc) and acc > best_acc:
            best_acc, best_fam = acc, fam
    return best_fam, best_acc


def _balanced_top1(per_dataset: Dict, family_map: Dict[str, str],
                   zone_to_family: Dict[str, str]) -> float:
    """Balanced Top-1 accuracy: average per-family recall (Table 4).

    For each oracle family f, compute recall = (# correctly predicted as f)
    / (# datasets whose oracle is f). Average over families with >= 1 dataset.
    """
    # Count per oracle family
    family_correct: Dict[str, int] = {}
    family_total: Dict[str, int] = {}

    for name, v in per_dataset.items():
        zone = v['predicted_zone']
        if zone in ('mixed', 'desert'):
            continue
        oracle_fam = v.get('oracle_family')
        if oracle_fam is None:
            continue
        family_total[oracle_fam] = family_total.get(oracle_fam, 0) + 1
        pred_fam = zone_to_family.get(zone)
        if pred_fam == oracle_fam:
            family_correct[oracle_fam] = family_correct.get(oracle_fam, 0) + 1

    recalls = []
    for fam in family_total:
        recalls.append(family_correct.get(fam, 0) / family_total[fam])
    return float(np.mean(recalls)) if recalls else float('nan')


def _regret(per_dataset: Dict, family_map: Dict[str, str],
            zone_to_family: Dict[str, str]) -> float:
    """Mean regret: oracle_acc - predicted_acc, averaged over non-mixed."""
    gaps = []
    for name, v in per_dataset.items():
        zone = v['predicted_zone']
        if zone in ('mixed', 'desert'):
            continue
        pred_fam = zone_to_family.get(zone)
        if pred_fam is None:
            continue
        pred_model = family_map.get(pred_fam)
        if pred_model is None:
            continue
        pred_acc = v['mean_accs'].get(pred_model, float('nan'))
        oracle_acc = v.get('oracle_acc', float('nan'))
        if not np.isnan(pred_acc) and not np.isnan(oracle_acc):
            gaps.append(oracle_acc - pred_acc)
    return float(np.mean(gaps)) * 100 if gaps else float('nan')


def bootstrap_ci(values: list, n_boot: int = 1000,
                 ci: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    vals = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if len(vals) == 0:
        return float('nan'), float('nan')
    boot_means = [rng.choice(vals, size=len(vals), replace=True).mean()
                  for _ in range(n_boot)]
    lo = np.percentile(boot_means, 100 * (1 - ci) / 2)
    hi = np.percentile(boot_means, 100 * (1 + ci) / 2)
    return float(lo), float(hi)


def experiment_ranking_prediction(dataset_names: Optional[list] = None,
                                   epsilon: float = 0.10,
                                   n_seeds: int = 10,
                                   seed: int = 42) -> Dict:
    """
    E3: Quantitative ranking prediction -- Table 4.

    Two settings:
      "With GT": 4 families MLP(F), APPNP(L), H2GCN(GNN), SGFormer(GT)
      "Classical": 3 families MLP(F), APPNP(L), H2GCN(GNN)

    Metrics:
      Top-1 = balanced accuracy (average per-family recall)
      Regret = mean accuracy gap from oracle (%)
    """
    from benchmarks import ALL_DATASETS, load_dataset

    print("\n" + "=" * 60)
    print("E3: Ranking Prediction (Table 4)")
    print(f"    epsilon={epsilon}, n_seeds={n_seeds}")
    print("=" * 60)

    if dataset_names is None:
        dataset_names = ALL_DATASETS

    per_dataset = {}

    for name in dataset_names:
        print(f"\n--- {name.upper()} ---")
        try:
            edge_index, X, y, _ = load_dataset(name)
            N = X.shape[0]

            # NSA diagnostic (once per dataset)
            train_mask_d, _, _ = make_masks(N, y, seed=seed)
            diag = run_nsa(edge_index, X, y, train_mask_d, verbose=False)
            fss, tss, lss = diag['FSS'], diag['TSS'], diag['LSS']
            zone = predict_family(fss, tss, lss, epsilon)
            print(f"  NSA: FSS={fss:.3f}  TSS={tss:.3f}  LSS={lss:.3f}  "
                  f"-> zone: {zone}")

            # Model accuracies averaged over n_seeds (micro-accuracy)
            seed_accs: Dict[str, list] = {}
            for s in range(n_seeds):
                s_seed = seed + s
                tr_m, va_m, te_m = make_masks(N, y, seed=s_seed)
                accs = run_all_models(
                    edge_index, X, y, tr_m, va_m, te_m,
                    seed=s_seed, metric='acc')
                for m, a in accs.items():
                    seed_accs.setdefault(m, []).append(a)

            mean_accs = {m: float(np.nanmean(v)) for m, v in seed_accs.items()}

            # Oracle for both settings
            oracle_gt, oracle_gt_acc = _oracle_family(mean_accs, FAMILY_MAP_GT)
            oracle_cl, oracle_cl_acc = _oracle_family(mean_accs, FAMILY_MAP_CLASSICAL)

            print(f"  Oracle (GT): {oracle_gt}={oracle_gt_acc:.3f}  "
                  f"Oracle (Classical): {oracle_cl}={oracle_cl_acc:.3f}")

            per_dataset[name] = {
                'FSS': fss, 'TSS': tss, 'LSS': lss,
                'predicted_zone': zone,
                'oracle_family_gt': oracle_gt,
                'oracle_acc_gt': oracle_gt_acc,
                'oracle_family_cl': oracle_cl,
                'oracle_acc_cl': oracle_cl_acc,
                'mean_accs': mean_accs,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

    # ---- Compute metrics for both settings ----
    # Attach oracle info for balanced top-1 computation
    for name, v in per_dataset.items():
        # With GT
        v['oracle_family'] = v['oracle_family_gt']
        v['oracle_acc'] = v['oracle_acc_gt']
    bal_top1_gt = _balanced_top1(per_dataset, FAMILY_MAP_GT, ZONE_TO_FAMILY_GT)
    regret_gt = _regret(per_dataset, FAMILY_MAP_GT, ZONE_TO_FAMILY_GT)

    # Classical
    for name, v in per_dataset.items():
        v['oracle_family'] = v['oracle_family_cl']
        v['oracle_acc'] = v['oracle_acc_cl']
    bal_top1_cl = _balanced_top1(per_dataset, FAMILY_MAP_CLASSICAL,
                                 ZONE_TO_FAMILY_CLASSICAL)
    regret_cl = _regret(per_dataset, FAMILY_MAP_CLASSICAL,
                         ZONE_TO_FAMILY_CLASSICAL)

    # ---- Summary (Table 4 format) ----
    n_non_mixed = sum(1 for v in per_dataset.values()
                      if v['predicted_zone'] not in ('mixed', 'desert'))
    n_mixed = len(per_dataset) - n_non_mixed

    print("\n" + "=" * 60)
    print("TABLE 4: NSA GUIDANCE ON MODEL SELECTION")
    print("=" * 60)
    print(f"  Datasets: {len(per_dataset)} total, "
          f"{n_non_mixed} non-mixed, {n_mixed} mixed/desert")
    print()
    print(f"  {'':20s}  {'With GT':>12s}  {'Classical':>12s}")
    print(f"  {'':20s}  {'(chance=25%)':>12s}  {'(chance=33.3%)':>14s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}")
    print(f"  {'Top-1 (balanced)':<20s}  {bal_top1_gt*100:>11.1f}%  "
          f"{bal_top1_cl*100:>11.1f}%")
    print(f"  {'Regret':<20s}  {regret_gt:>11.2f}%  "
          f"{regret_cl:>11.2f}%")

    return {
        'per_dataset': per_dataset,
        'with_gt': {'top1_balanced': bal_top1_gt, 'regret': regret_gt},
        'classical': {'top1_balanced': bal_top1_cl, 'regret': regret_cl},
    }


# =============================================================
# MAIN
# =============================================================

if __name__ == '__main__':
    from benchmarks import ALL_DATASETS

    print("Node Signal Atlas (NSA)")
    print("Characterizing Node Classification Beyond Homophily")
    print("=" * 60)

    # E1: Synthetic regime validation (24 datasets: 8 corners + 16 sweep, 10 seeds)
    e1 = experiment_synthetic_regimes(seed=42, n_runs=10)

    # E2: Real benchmark ranking prediction (18 datasets)
    e2 = experiment_real_benchmarks(
        dataset_names=ALL_DATASETS, seed=42)

    # E3: Quantitative ranking prediction (Top-1 + Regret)
    e3 = experiment_ranking_prediction(
        dataset_names=ALL_DATASETS,
        epsilon=0.10, n_seeds=10, seed=42)
