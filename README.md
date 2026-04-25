# Node Signal Atlas (NSA)

**Node Signal Atlas: Characterizing Node Classification Beyond Homophily**

*Submitted to NeurIPS 2026*

NSA is a pre-training diagnostic framework that embeds a node classification dataset into a three-dimensional signal space:

- **FSS** (Feature Signal Score) -- how much label signal is accessible from node features and their local organization
- **TSS** (Topology Signal Score) -- how much label signal is accessible from unlabeled graph structure
- **LSS** (Label Smoothness Score) -- how much label signal is accessible from the smoothness of observed training labels along edges

All three scores are computed from the same fixed logistic-regression probe applied to cheap annulus-based descriptors at hops k=0,1,2,3. Each score is a normalized macro-accuracy: (balanced_acc - 1/C) / (1 - 1/C), mapping chance to 0 and perfect classification to 1. NSA does not train any GNN and requires only lightweight preprocessing and probe evaluation.

## Project Structure

```
NSA/
├── descriptors.py          # Core: FSS/TSS/LSS descriptors + probe
├── synthetics.py           # 8 corner + 16 sweep synthetic dataset generators
├── models.py               # 9 baseline models (MLP, LP, GCN, H2GCN, GPS, SGFormer, APPNP, SAGE, SGC)
├── benchmarks.py           # 18 real benchmark dataset loader + metadata
├── run_synthetics.py       # Synthetic experiment runner
├── run_baselines.py        # Benchmark runner (NSA scores + 9 models x 10 seeds)
├── benchmark_npz_all/      # Pre-cached .npz dataset files (18 datasets)
└── README.md
```

## Installation

PyTorch and PyG installation depends on your CUDA version. See:
- https://pytorch.org/get-started/locally/
- https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Quick Start

### Compute NSA scores (FSS, TSS, LSS)

```python
from descriptors import NodeDiagnostic
import numpy as np

# Your data: edge_index (2, E), X (N, d), y (N,), train_mask (N,) bool
diag = NodeDiagnostic(r=32, T=4, K=3, m=(64, 64, 64), n_seeds=3)
results = diag.fit(edge_index, X, y, train_mask)

print(f"FSS={results['FSS']:.3f}  TSS={results['TSS']:.3f}  LSS={results['LSS']:.3f}")
```

### Run baseline models on benchmarks

```bash
python run_baselines.py
python run_baselines.py --datasets cora cornell --models MLP GCN H2GCN
python run_baselines.py --device cuda:0 --runs 10
```

### Run synthetic experiments

```bash
python run_synthetics.py
```

## Benchmark Suite (18 datasets)

| Category | Datasets |
|---|---|
| Heterophilic | Roman-Empire, Texas, Wisconsin, Squirrel-filtered, Cornell, Actor, Squirrel, Flickr, Chameleon-filtered, Chameleon, Amazon-Ratings, BlogCatalog |
| Homophilic | Deezer-Europe, CiteSeer, PubMed, Amazon-Computers, Cora, Amazon-Photo |

## Baseline Models

| Family | Models | Signal Channel |
|---|---|---|
| Feature-only | MLP | FSS |
| Label diffusion | LP, APPNP | LSS |
| Topology-aware GNNs | GCN, H2GCN, SAGE, SGC | TSS |
| Graph transformers | GPS, SGFormer | TSS |

## Evaluation Protocol

- **NSA scores**: Normalized macro-accuracy (balanced accuracy), 0 = chance, 1 = perfect
- **Model performance**: Micro-accuracy on stratified 60/20/20 split, averaged over 10 seeds
- **Hyperparameter search**: lr in {0.01, 0.005}, weight_decay in {0, 5e-4, 5e-3}, dropout in {0.3, 0.5}
- **Training**: 2 hidden layers, width 64, Adam, up to 500 epochs with early stopping (patience 50)

## Citation

```bibtex
@inproceedings{nsa2026,
  title={Node Signal Atlas: Characterizing Node Classification Beyond Homophily},
  author={Anonymous},
  booktitle={Submitted to NeurIPS 2026},
  year={2026}
}
```

## License

MIT
