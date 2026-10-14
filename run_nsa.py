"""NSA scores for a single benchmark.

Usage:
  run_nsa.py <dataset> [--split official|strat60] [--probe N] [--tag TAG]

  --split official : OGB's own time-based split (ogbn-arxiv only).
  --split strat60  : stratified 60% train mask, RandomState(42)  [default]
  --probe N        : reduce the number of DIAGNOSTIC TARGET nodes to N,
                     drawn from the training set with the training set's
                     class proportions. The graph is NOT modified -- every
                     edge is still used for neighborhood construction.
                     Omit for the full run.

LSS uses the leakage-free probe (descriptors.run_probe_lss): the LSS design
matrix is rebuilt inside each CV fold from that fold's inner-training labels.
"""
import argparse, json, os, socket, time
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from benchmarks import load_dataset
from descriptors import NodeDiagnostic

ap = argparse.ArgumentParser()
ap.add_argument('dataset')
ap.add_argument('--split', default='strat60', choices=['official', 'strat60'])
ap.add_argument('--probe', type=int, default=0)
ap.add_argument('--metric', default='macro', choices=['macro', 'acc'],
                help="macro: normalized macro-accuracy (the NSA score); "
                     "acc: raw micro-accuracy of the same probe")
ap.add_argument('--tag', default='')
a = ap.parse_args()

print(f'host={socket.gethostname()}  dataset={a.dataset}  split={a.split}'
      + (f'  probe={a.probe:,}' if a.probe else '  probe=all'), flush=True)

t = time.time()
edge_index, X, y, C = load_dataset(a.dataset)
N = X.shape[0]
print(f'  load: {time.time()-t:.1f}s   N={N:,} E={edge_index.shape[1]:,} '
      f'd={X.shape[1]} C={C}', flush=True)

# ---- training mask ----
if a.split == 'official':
    from ogb.nodeproppred import PygNodePropPredDataset
    split_idx = PygNodePropPredDataset(
        name=a.dataset, root=os.environ.get('NSA_OGB_ROOT', 'ogb_data')
    ).get_idx_split()
    train_mask = np.zeros(N, dtype=bool)
    train_mask[split_idx['train'].numpy()] = True
    print(f'  official split: train={train_mask.sum():,} '
          f'valid={len(split_idx["valid"]):,} test={len(split_idx["test"]):,}', flush=True)
else:
    rng = np.random.RandomState(42)
    train_mask = np.zeros(N, dtype=bool)
    for c in np.unique(y[y >= 0]):
        idx = rng.permutation(np.where(y == c)[0])
        train_mask[idx[:int(len(idx) * 0.6)]] = True
    print(f'  stratified 60% split: train={train_mask.sum():,}', flush=True)

n_train_full = int(train_mask.sum())

# ---- probe-node subsample (graph untouched) ----
if a.probe and a.probe < n_train_full:
    tr = np.flatnonzero(train_mask)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=a.probe, random_state=0)
    sel, _ = next(sss.split(np.zeros(len(tr)), y[tr]))
    train_mask = np.zeros(N, dtype=bool)
    train_mask[np.sort(tr[sel])] = True
    print(f'  probe nodes: {train_mask.sum():,} of {n_train_full:,} '
          f'(stratified on training-set class ratio; graph unchanged, '
          f'E={edge_index.shape[1]:,})', flush=True)

t0 = time.time()
res = NodeDiagnostic(r=32, T=4, K=3, n_seeds=3, metric=a.metric).fit(
    edge_index, X, y, train_mask, verbose=True)
el = time.time() - t0

row = dict(dataset=a.dataset, split=a.split, metric=a.metric,
           probe_nodes=int(train_mask.sum()), train_nodes_full=n_train_full,
           N=N, E=int(edge_index.shape[1]), d=int(X.shape[1]), C=C,
           FSS=round(res['FSS'], 4), TSS=round(res['TSS'], 4),
           LSS=round(res['LSS'], 4), FSS_std=round(res['FSS_std'], 4),
           TSS_std=round(res['TSS_std'], 4), LSS_std=round(res['LSS_std'], 4),
           time_s=round(el, 1), host=socket.gethostname())
print(f"\n  FSS={row['FSS']:.4f}+-{row['FSS_std']:.4f}  "
      f"TSS={row['TSS']:.4f}+-{row['TSS_std']:.4f}  "
      f"LSS={row['LSS']:.4f}+-{row['LSS_std']:.4f}")
print(f'  wall-clock: {el:.1f}s ({el/60:.1f} min)')

os.makedirs('results', exist_ok=True)
tag = a.tag or f"{a.dataset}_{a.split}_{a.metric}" + (f"_probe{a.probe}" if a.probe else "_full")
with open(f'results/nsa_{tag}.json', 'w') as f:
    json.dump(row, f, indent=2)
print(f'  wrote results/nsa_{tag}.json')
