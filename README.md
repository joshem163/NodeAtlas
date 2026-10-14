# Node Signal Atlas (NSA)

**Node Signal Atlas: Mapping Predictive Signal Beyond Homophily**

*Submitted to ICLR 2027*

NSA is a pre-training diagnostic that embeds a node-classification dataset into a
three-dimensional signal space:

- **FSS** (Feature Signal Score) — label signal accessible from node features and
  their local organization
- **TSS** (Topology Signal Score) — label signal accessible from *unlabeled* graph
  structure
- **LSS** (Label Smoothness Score) — label signal accessible from the smoothness of
  observed training labels along edges

All three come from the same fixed logistic-regression probe applied to annulus
descriptors at hops `k = 0,1,2,3`. Each score is a normalized macro-accuracy,

```
s̃ = (balanced_accuracy − 1/C) / (1 − 1/C),
```

so 0 is chance and 1 is perfect, independently of the class count. NSA trains no
GNN and touches no GPU.

---

## Repository layout

```
NSA/
├── descriptors.py               # FSS/TSS/LSS descriptors + probes  (the core)
├── benchmarks.py                # 20 real benchmarks: metadata + loaders
├── models.py                    # 9 baseline models + split/eval helpers
├── synthetics.py                # 24 synthetic generators (8 corners + 16 sweep)
│
├── run_nsa.py                   # NSA for one dataset (+ official split, probe subsampling)
├── run_all_benchmarks.py        # NSA for every dataset in one process (Table 8 timings)
├── run_baselines.py             # NSA + 9 models, stratified 60/20/20  (Tables 3, 10)
├── run_baselines_ogbsplit.py    # ogbn-arxiv under OGB's official time-based split
├── run_baselines_macro.py       # as run_baselines.py, but models scored in normalized macro-acc
├── run_synthetics.py            # synthetic suite runner
│
├── slurm/                       # job scripts used for the reported runs
├── benchmark_npz_all/           # cached .npz for the small/medium benchmarks
└── arch_selection/              # the cross-generator architecture-triage study
    ├── nsa_v5.py                #   frozen benchmark: generators, architectures, protocol
    ├── complementarity.py       #   refits the Ridge selectors -> Tables 4, 13
    └── artifacts/               #   the exact outputs behind the reported numbers
```

The architecture-triage study is a separate experiment on synthetic
cross-generator problems, with its own architecture bank and protocol. See
[`arch_selection/README.md`](arch_selection/README.md).

---

## Install

```bash
pip install numpy scipy scikit-learn networkx
# PyTorch and PyG depend on your CUDA version:
#   https://pytorch.org/get-started/locally/
#   https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install ogb          # only for ogbn-arxiv
```

**PyG version.** `models.py` imports `SGFormer` from `torch_geometric.nn.models`,
which is only present in PyG ≥ 2.7 (master at time of writing). PyG 2.6.x will
fail on import. Everything else works on 2.6.

Dataset caches for the two large benchmarks are downloaded on first use. Override
their location with `NSA_OGB_ROOT` (ogbn-arxiv) and `NSA_LINKX_ROOT` (genius).

---

## Quick start

### NSA scores for one dataset

```bash
python run_nsa.py cora
# FSS=0.6463  TSS=0.2870  LSS=0.7660
```

```python
from benchmarks import load_dataset
from descriptors import NodeDiagnostic
import numpy as np

edge_index, X, y, C = load_dataset("cora")
rng = np.random.RandomState(42)
train_mask = np.zeros(len(y), dtype=bool)
for c in np.unique(y[y >= 0]):                 # stratified 60% train mask
    idx = rng.permutation(np.where(y == c)[0])
    train_mask[idx[: int(0.6 * len(idx))]] = True

res = NodeDiagnostic(r=32, T=4, K=3, n_seeds=3, metric="macro").fit(
    edge_index, X, y, train_mask)
print(res["FSS"], res["TSS"], res["LSS"])
```

### All benchmarks, with runtimes

```bash
python run_all_benchmarks.py                   # normalized macro-accuracy (the NSA scores)
python run_all_benchmarks.py --metric acc      # raw micro-accuracy of the same probes
```

Running them in one process is deliberate: the wall-clock times reported in the
appendix must come from an identical machine state to be comparable. The two
metrics are the two halves of the scoring-protocol comparison table.

### Synthetic suite

```bash
python run_synthetics.py          # 24 datasets x 9 models x 10 seeds, plus NSA scores
```

### Baseline models

```bash
python run_baselines.py --datasets cora --runs 10 --device cuda \
       --models-only --models MLP GCN SAGE APPNP SGC H2GCN LP GPS SGFormer \
       --out results/cora_10.csv
```

### ogbn-arxiv on the OGB official split

```bash
python run_baselines_ogbsplit.py --datasets ogbn-arxiv --runs 10 --device cuda \
       --models-only --models GCN --out results/arxiv_GCN_ogb10.csv
```

### Architecture triage

```bash
cd arch_selection
python complementarity.py        # reruns Tables 4 and 13 from the cached features
```

### Probe-node subsampling study

```bash
python run_nsa.py ogbn-arxiv --split official --probe 10000
```

`--probe N` reduces the number of *diagnostic target nodes* to `N`, drawn from the
training set with the training set's class proportions. **The graph is never
modified** — every edge is still used for neighborhood construction. This is a
cost/accuracy trade-off for the probe, not graph sparsification.

---

## Protocol

| | |
|---|---|
| NSA hyperparameters | `r=32`, `T=4`, `K=3`, `m_k=64`, fixed across all datasets |
| Probe | multinomial logistic regression, ℓ2, `C=1.0`, L-BFGS, `max_iter=1000`, uniform class weights — **not** tuned per dataset |
| Cross-validation | stratified 5-fold over the training nodes only |
| Sampling seeds | 3 (annulus sampling); scores are the mean |
| Node split | stratified 60/20/20; ogbn-arxiv additionally supports OGB's official time-based split |
| Model metric | micro-accuracy, mean ± std over 10 runs |
| Model grid | lr ∈ {0.01, 0.005}, wd ∈ {0, 5e-4, 5e-3}, dropout ∈ {0.3, 0.5}; LP α over 7 values |
| Model architecture | 2 layers, width 64, ≤500 epochs, early stopping patience 50 |

### LSS is rebuilt inside every CV fold

The LSS descriptor is a function of *observed labels*, so it cannot be built once
and then cross-validated: the validation fold's labels would already sit inside
the descriptors of their neighbours. `run_probe_lss` therefore rebuilds the LSS
design matrix for each fold using only that fold's inner-training labels. FSS and
TSS are label-independent and are built once per sampling seed.

This makes the LSS probe roughly 5× more expensive than FSS or TSS, and it is the
reason LSS degrades faster than FSS/TSS under probe subsampling — a smaller probe
set also means fewer labelled neighbours are visible to the descriptor.

### NSA uses no GPU

`descriptors.py` is pure numpy/scikit-learn. All reported NSA runtimes were
measured on CPU (16 cores). Only the baseline models need a GPU.

---

## Datasets

20 benchmarks, ordered by node homophily. The first 18 load from
`benchmark_npz_all/`; **genius** (LINKX) and **ogbn-arxiv** (OGB) download on first
use.

Heterophilic: Roman-Empire, Texas, Wisconsin, Squirrel-filtered, Cornell, Actor,
Squirrel, Flickr, Chameleon-filtered, Chameleon, Amazon-Ratings, BlogCatalog,
Genius.
Homophilic: Deezer-Europe, Ogbn-arxiv, CiteSeer, PubMed, Amazon-Computers, Cora,
Amazon-Photo.

`deezer_europe.npz` exceeds GitHub's file limit and is gitignored; it is fetched
through the PyG fallback.

---

## Notes on the baselines

- **H2GCN** materializes 2-hop neighbourhoods and OOMs on a 24 GB GPU for the
  larger benchmarks. Use an 80 GB+ device.
- **GPS** materializes a dense `[heads, N, N]` attention tensor, which is
  independent of hidden width. It OOMs on genius and ogbn-arxiv even at 80 GB.
- On CUDA 12.4 builds of PyTorch there are no Blackwell (sm_120) kernels; jobs
  landing on such a GPU fail with *"no kernel image is available for execution on
  the device"*. Pin the partition accordingly.
