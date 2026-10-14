"""Refit the Ridge selectors with the complementary representations.

Produces Table 4 (architecture selection) and Table 13 (paired
complementarity tests) of the paper.

The v5 revision frames the result as complementarity rather than competition:
Drops Meta-5 from the main table and adds NSA+Homophily-1 and NSA+Homophily-4.

nsa_v5.py is NOT modified. This rebuilds the same corpora (fresh final seeds
1010/2020) so the architecture accuracies come straight from the cache -- no
GNN is retrained -- then refits the meta-predictors exactly as
evaluate_predictors() does, with the extra representations added.

Corpus features are cached to npz, so further representation variants are free.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsa_v5 as v5

RES = Path(__file__).resolve().parent / "artifacts"
NPZ = RES / "v5_final_features.npz"

cfg = v5.ExperimentConfig()
cfg.cache_dir = str(Path(__file__).resolve().parent / "cache")
archs = list(v5.active_architectures(cfg))
mechs = list(v5.NATIVE_MECHANISMS)

if NPZ.exists():
    z = np.load(NPZ, allow_pickle=True)
    D = {k: z[k] for k in z.files}
    print(f"loaded cached features from {NPZ}")
else:
    cache = v5.PerformanceCache(
        Path(cfg.cache_dir) /
        f"architecture_perf_{v5.GENERATOR_VERSION}_{v5.MODEL_BANK_VERSION}.json")
    n_before = len(cache.db)
    print(f"cache holds {n_before} entries before the rebuild", flush=True)
    print("rebuilding FINAL train corpus (seed 1010) ...", flush=True)
    tr = v5.build_synthetic_corpus("train", cfg.train_problems_per_mechanism, cfg,
                                   cache, seed=1010, train_models=True,
                                   compute_descriptors=True)
    print("rebuilding FINAL test corpus (seed 2020) ...", flush=True)
    te = v5.build_synthetic_corpus("test", cfg.test_problems_per_mechanism, cfg,
                                   cache, seed=2020, train_models=True,
                                   compute_descriptors=True)
    n_after = len(cache.db)
    print(f"cache holds {n_after} entries after -- {n_after - n_before} new "
          f"(0 means nothing was retrained)")
    D = {}
    for tag, items in (("tr", tr), ("te", te)):
        D[f"{tag}_nsa"] = v5.nsa_matrix(items)
        D[f"{tag}_hom4"] = v5.homophily_matrix(items)
        D[f"{tag}_nodeh"] = v5.node_homophily_matrix(items)
        D[f"{tag}_meta5"] = np.vstack([x.conventional[list(v5._META5_INDICES)]
                                       for x in items])
        D[f"{tag}_perf"] = v5.performance_matrix(items)
        D[f"{tag}_mech"] = np.array([x.mechanism for x in items])
    np.savez(NPZ, **D)
    print(f"wrote {NPZ}")

Ytr, Yte = D["tr_perf"], D["te_perf"]

REPS = {
    "Node Homophily":       (lambda t: D[f"{t}_nodeh"], 1),
    "Homophily-4":          (lambda t: D[f"{t}_hom4"], 4),
    "NSA-3":                (lambda t: D[f"{t}_nsa"], 3),
    "NSA + Node Homophily": (lambda t: np.hstack([D[f"{t}_nsa"], D[f"{t}_nodeh"]]), 4),
    "NSA + Homophily-4":    (lambda t: np.hstack([D[f"{t}_nsa"], D[f"{t}_hom4"]]), 7),
    "Homophily-4 + Meta-5": (lambda t: np.hstack([D[f"{t}_hom4"], D[f"{t}_meta5"]]), 9),
    "NSA + Meta-5":         (lambda t: np.hstack([D[f"{t}_nsa"], D[f"{t}_meta5"]]), 8),
}

gm = np.nanmean(Ytr, axis=0)
mrows, preds = {}, {}
for name, (fn, dim) in REPS.items():
    P = v5.PerformancePredictor(cfg.meta_ridge_alpha).fit(fn("tr"), Ytr)
    Xte = fn("te")
    pr = np.vstack([P.predict(Xte[i]) for i in range(len(Xte))])
    preds[name] = pr
    mrows[name] = [v5.selection_metrics(Yte[i], pr[i]) for i in range(len(Yte))]
mrows["[base] global mean"] = [v5.selection_metrics(Yte[i], gm) for i in range(len(Yte))]

MAIN = ["Node Homophily", "Homophily-4", "NSA-3",
        "NSA + Node Homophily", "NSA + Homophily-4"]
APPX = ["Homophily-4 + Meta-5", "NSA + Meta-5"]

print("\n" + "=" * 92)
print("MAIN TABLE -- 90 held-out problems, fresh final seeds, regret is primary")
print("=" * 92)
print(f"{'Representation':24s}{'dims':>5s}{'regret':>17s}{'Spearman':>15s}"
      f"{'Top-1':>14s}{'Top-3':>14s}")
print("-" * 92)


def row(name, dim=""):
    m = v5.metrics_mean_std(mrows[name])
    g = lambda k, p=4: f"{m[k]['mean']:.{p}f}±{m[k]['std']:.{p}f}"
    print(f"{name:24s}{str(dim):>5s}{g('regret'):>17s}{g('spearman',3):>15s}"
          f"{g('top1',3):>14s}{g('top3',3):>14s}")


for n in MAIN:
    row(n, REPS[n][1])
print("-" * 92)
row("[base] global mean")
print("-" * 92)
print("appendix (Meta-5), dropped from the main story:")
for n in APPX:
    row(n, REPS[n][1])

print("\n" + "=" * 92)
print("PRE-SPECIFIED COMPLEMENTARITY TESTS (paired Wilcoxon, left regret < right)")
print("=" * 92)


def reg(name):
    return np.array([m.regret for m in mrows[name]])


tests = [
    ("nsa_nodeh_vs_nodeh", "NSA + Node Homophily", "Node Homophily"),
    ("nsa_hom4_vs_hom4", "NSA + Homophily-4", "Homophily-4"),
    ("nsa_hom4_vs_nsa", "NSA + Homophily-4", "NSA-3"),
]
out_tests = {}
for tag, a, b in tests:
    try:
        p = float(wilcoxon(reg(a), reg(b), alternative="less").pvalue)
    except ValueError:
        p = float("nan")
    out_tests[tag] = p
    star = "  *significant" if p == p and p < 0.05 else ""
    print(f"  {a:22s} < {b:22s}  p={p:.4f}{star}")
print("\n  (the main complementarity test is the second row)")

print("\n" + "=" * 92)
print("PER-REGIME regret")
print("=" * 92)
show = MAIN + ["[base] global mean"]
print(f"{'regime':26s}" + "".join(f"{n[:11]:>13s}" for n in show))
me = D["te_mech"]
for mech in mechs:
    ids = [i for i, x in enumerate(me) if x == mech]
    vals = [np.mean([mrows[n][i].regret for i in ids]) for n in show]
    b = int(np.argmin(vals[:5]))
    print(f"{mech:26s}" + "".join(
        (f"{v:>12.4f}*" if i == b else f"{v:>13.4f}") for i, v in enumerate(vals)))

print("\n" + "=" * 92)
print("SENSITIVITY -- excluding neighbor_feature_context (75 problems, post hoc)")
print("=" * 92)
keep = np.array([m != "neighbor_feature_context" for m in me])
print(f"{'Representation':24s}{'all 90':>12s}{'excl. 75':>12s}")
for n in MAIN:
    r = reg(n)
    print(f"{n:24s}{r.mean():>12.4f}{r[keep].mean():>12.4f}")
print()
for tag, a, b in tests:
    ra, rb = reg(a)[keep], reg(b)[keep]
    try:
        p = float(wilcoxon(ra, rb, alternative="less").pvalue)
    except ValueError:
        p = float("nan")
    star = "  *significant" if p == p and p < 0.05 else ""
    print(f"  {a:22s} < {b:22s}  p={p:.4f}{star}")

summary = {"main": {n: v5.metrics_mean_std(mrows[n]) for n in MAIN + APPX},
           "global_mean": v5.metrics_mean_std(mrows["[base] global mean"]),
           "regret_tests": out_tests}
(RES / "v5_complementarity.json").write_text(json.dumps(summary, indent=2))
print(f"\nwrote {RES / 'v5_complementarity.json'}")
