"""Compute NSA for every registered benchmark, in one process.

Protocol matches the paper: stratified 60% train mask (RandomState(42)),
NodeDiagnostic(r=32, T=4, K=3, n_seeds=3).

  --metric macro   normalized macro-accuracy -- the NSA scores used throughout
                   the paper (Tables 3, 6, and the left half of Table 7)
  --metric acc     raw micro-accuracy of the same probes, for the right half
                   of Table 7 (the scoring-protocol comparison)
They run sequentially inside one process so the reported runtimes come from
an identical machine state (Appendix Table 8). Pass dataset names to restrict
the set; with no arguments it runs all of ALL_DATASETS.
"""
import argparse, csv, os, socket, time
import numpy as np
from benchmarks import load_dataset, DATASET_INFO, dataset_category, ALL_DATASETS
from descriptors import NodeDiagnostic

ap = argparse.ArgumentParser()
ap.add_argument("datasets", nargs="*", help="defaults to every registered dataset")
ap.add_argument("--metric", default="macro", choices=["macro", "acc"])
ap.add_argument("--out", default=None)
args = ap.parse_args()

OUT = args.out or f"results/nsa_benchmarks_{args.metric}.csv"
FIELDS = ["dataset", "category", "N", "E", "d", "C", "h_edge", "h_node",
          "FSS", "TSS", "LSS", "FSS_std", "TSS_std", "LSS_std",
          "time_s", "host"]

os.makedirs("results", exist_ok=True)
done = set()
if os.path.exists(OUT):
    done = {r["dataset"] for r in csv.DictReader(open(OUT))}
else:
    with open(OUT, "w", newline="") as f:
        csv.DictWriter(f, FIELDS).writeheader()

names = args.datasets or list(ALL_DATASETS)
print("host=%s  datasets=%d  metric=%s"
      % (socket.gethostname(), len(names), args.metric), flush=True)

for name in names:
    if name in done:
        print("skip %s (already done)" % name, flush=True)
        continue
    edge_index, X, y, C = load_dataset(name)
    N = X.shape[0]
    rng = np.random.RandomState(42)
    tm = np.zeros(N, dtype=bool)
    for c in np.unique(y[y >= 0]):
        idx = rng.permutation(np.where(y == c)[0])
        tm[idx[:int(len(idx) * 0.6)]] = True

    t0 = time.time()
    res = NodeDiagnostic(r=32, T=4, K=3, n_seeds=3, metric=args.metric).fit(
        edge_index, X, y, tm, verbose=False)
    el = time.time() - t0

    info = DATASET_INFO.get(name, (0, 0, 0, 0, 0, 0, "?"))
    row = dict(dataset=name, category=dataset_category(name), N=N,
               E=int(edge_index.shape[1]), d=int(X.shape[1]), C=C,
               h_edge=info[4], h_node=info[5],
               FSS=round(res["FSS"], 4), TSS=round(res["TSS"], 4),
               LSS=round(res["LSS"], 4), FSS_std=round(res["FSS_std"], 4),
               TSS_std=round(res["TSS_std"], 4), LSS_std=round(res["LSS_std"], 4),
               time_s=round(el, 1), host=socket.gethostname())
    with open(OUT, "a", newline="") as f:
        csv.DictWriter(f, FIELDS).writerow(row)
    print("%-20s FSS=%.4f TSS=%.4f LSS=%.4f  %8.1fs"
          % (name, row["FSS"], row["TSS"], row["LSS"], el), flush=True)

print("done", flush=True)
