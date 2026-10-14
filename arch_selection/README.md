# Architecture Triage (Section 4, Appendix E)

The cross-generator benchmark used to test whether NSA's coordinates carry
architecture-selection information beyond homophily statistics. It is separate
from the real-benchmark atlas: no dataset here is real, and the architecture
bank is a different, smaller one.

```
arch_selection/
├── nsa_v5.py            # the frozen benchmark: generators, architectures, protocol
├── complementarity.py   # refits the Ridge selectors -> Tables 4 and 13
└── artifacts/           # the exact outputs behind the reported numbers
```

## What the benchmark is

- **6 signal regimes** — `feature_signal`, `smooth_diffusion`,
  `structured_heterophily`, `structural_role`, `neighbor_feature_context`,
  `mixed_signal`. A regime says how predictive information enters the problem;
  it does **not** name a target architecture. Winners are always empirical.
- **6 architectures** — MLP, GCN, GraphSAGE, APPNP, H2GCN, GIN. Chosen because
  they expose more distinct signal-access mechanisms than the flexible
  generalists (GPS, SGFormer) used on the real benchmarks.
- **Cross-generator split** — train and test problems come from *different*
  graph-generation procedures with the same underlying mechanism, so the
  selector is evaluated cold on an unseen generator family.
- 270 problems (180 train / 90 test), `n = 512`, 3 balanced classes, node split
  stratified 60/20/20.

## Pilot → freeze → final

`nsa_v5.py` enforces the discipline in code:

| Stage | Flag | Purpose |
|---|---|---|
| Audit | `--audit-only` | generator and NSA sanity checks (NaNs, saturation, drift) |
| Pilot | `--pilot-only` | calibrate on **architecture performance alone** — winner diversity, selection headroom, spread |
| Final | (default) | fresh seeds, the frozen specification, descriptor comparison |

The seven acceptance criteria were fixed before any descriptor was evaluated.
Six passed on the 36-problem pilot; the one miss (fixed Top-3 hit rate 0.833
against a 0.80 target) was left uncorrected rather than tuning the generators
after seeing the result.

## Running it

```bash
# 1. Build the corpus and train the architecture bank (slow; results are cached)
python nsa_v5.py --out artifacts/final --cache-dir cache

# 2. Refit the selectors -> Tables 4 and 13
python complementarity.py
```

Step 2 reads `artifacts/v5_final_features.npz` when it exists, so the
representation comparison reruns in seconds without touching a GNN. Delete that
file to rebuild the features from the architecture cache.

`complementarity.py` imports `nsa_v5` and does not modify it; the extra
representations are fitted with the benchmark's own `PerformancePredictor`,
`selection_metrics`, and `metrics_mean_std`.

## Artifacts

| File | Contents |
|---|---|
| `nsa_interface_benchmark_v5_final.json` | the full final run (90 held-out problems) |
| `v5_complementarity.json` | the metrics behind Tables 4 and 13 |
| `v5_final_features.npz` | cached corpus features |
| `nsa_v5_pilot_headroom.json` | the pilot and its seven criteria |
| `generator_audit_v5.json` | stage-1 generator audit |
| `final_config.json` | the frozen configuration of the final run |

## Headline numbers

Oracle accuracy 0.701; best constant architecture (GIN) 0.630, leaving 7.18 pp
of selection headroom. All six architectures win somewhere: GIN 42.2%,
H2GCN 23.3%, SAGE 21.1%, and MLP / GCN / APPNP 4.4% each.

| Representation | Dim. | Regret (pp) ↓ |
|---|---|---|
| Constant selector | – | 7.18 |
| Node Homophily | 1 | 6.89 |
| Homophily-4 | 4 | 1.62 |
| NSA | 3 | 2.63 |
| **NSA + Node Homophily** | 4 | **1.18** |
| NSA + Homophily-4 | 7 | 1.31 |

Adding NSA to Homophily-4 reduces regret from 1.62 to 1.31 pp
(*p* = 0.0234, paired one-sided Wilcoxon), and the combination also beats NSA
alone (*p* = 0.0106). The standalone comparisons were pre-registered; the two
concatenated representations were introduced after inspecting the standalone
results, with the generators, architecture bank, and performances already
frozen.
