"""
Benchmark Metadata & Data Loader for Node Signal Atlas (NSA)
=============================================================
Single entry point for all 23 benchmark datasets.

Metadata:
  DATASET_INFO  -- (N, E, d, C, h_edge, h_node, category) per dataset
  ALL_DATASETS  -- ordered list of all dataset names

Loading priority:
  1. Local npz files in benchmark_npz_all/  (fast, offline, reproducible)
  2. PyG / OGB / LINKX download fallback    (for datasets without a local npz)

All loaders return the same format:
  edge_index:  (2, E) numpy int64
  X:           (N, d) numpy float32
  y:           (N,)   numpy int
  num_classes: int

Splits are NOT included -- each script generates its own splits
(stratified 60/20/20) to match the paper's multi-seed protocol.
"""

import os
import warnings
import numpy as np


# =============================================================
# DATASET METADATA
# =============================================================

# Canonical dataset metadata: (N, E, d, C, h_edge, h_node, category)
DATASET_INFO = {
    # Heterophilic -- Malignant
    'cornell':            (183,    295,       1703,   5,  0.298, 0.200, 'het-malignant'),
    'wisconsin':          (251,    499,       1703,   5,  0.170, 0.099, 'het-malignant'),
    'texas':              (183,    309,       1703,   5,  0.062, 0.056, 'het-malignant'),
    'actor':              (7600,   33544,     931,    5,  0.216, 0.202, 'het-malignant'),
    'deezer-europe':      (28281,  92752,     31241,  2,  0.525, 0.530, 'het-malignant'),
    'roman-empire':       (22662,  32927,     300,    18, 0.047, 0.046, 'het-malignant'),
    'blogcatalog':        (5196,   171743,    8189,   6,  0.401, 0.391, 'het-malignant'),
    'flickr':             (7575,   239738,    12047,  9,  0.239, 0.243, 'het-malignant'),
    # Heterophilic -- Benign
    'chameleon':          (2277,   36101,     2325,   5,  0.234, 0.247, 'het-benign'),
    'squirrel':           (5201,   217073,    2089,   5,  0.223, 0.215, 'het-benign'),
    'chameleon-filtered': (890,    8854,      2325,   5,  0.236, 0.244, 'het-benign'),
    'amazon-ratings':     (24492,  93050,     300,    5,  0.380, 0.376, 'het-benign'),
    # Heterophilic -- Ambiguous
    'squirrel-filtered':  (2223,   46998,     2089,   5,  0.207, 0.191, 'het-ambiguous'),
    # Homophilic
    'cora':               (2708,   5429,      1433,   7,  0.810, 0.825, 'homophilic'),
    'citeseer':           (3327,   4732,      3703,   6,  0.736, 0.706, 'homophilic'),
    'pubmed':             (19717,  44338,     500,    3,  0.802, 0.792, 'homophilic'),
    'amazon-photo':       (7650,   119081,    745,    8,  0.827, 0.849, 'homophilic'),
    'amazon-computers':   (13752,  245861,    767,    10, 0.777, 0.802, 'homophilic'),
    # Added 2026-09: measured on the loaded graph after ToUndirected().
    'ogbn-arxiv':         (169343, 2315598,   128,    40, 0.654, 0.635, 'homophilic'),
    # genius (LINKX, Lim et al. NeurIPS 2021). Measured after symmetrisation.
    'genius':             (421961, 1845736,   12,     2,  0.593, 0.509, 'het-malignant'),
}

DATASET_GROUPS = {
    'het-malignant': ['cornell', 'wisconsin', 'texas', 'actor',
                      'deezer-europe', 'roman-empire', 'blogcatalog', 'flickr',
                      'genius'],
    'het-benign':    ['chameleon', 'squirrel', 'chameleon-filtered',
                      'amazon-ratings'],
    'het-ambiguous': ['squirrel-filtered'],
    'homophilic':    ['cora', 'citeseer', 'pubmed',
                      'amazon-photo', 'amazon-computers',
                      'ogbn-arxiv'],
}

ALL_DATASETS = (
    DATASET_GROUPS['het-malignant'] +
    DATASET_GROUPS['het-benign'] +
    DATASET_GROUPS['het-ambiguous'] +
    DATASET_GROUPS['homophilic']
)



def dataset_category(name: str) -> str:
    """Return the signal category string for a dataset name."""
    name = name.lower().replace('_', '-')
    info = DATASET_INFO.get(name, None)
    return info[6] if info is not None else 'unknown'


# =============================================================
# DATA LOADING  (npz first, PyG fallback)
# =============================================================

NPZ_DIR = "benchmark_npz_all"


def _symmetrize_and_clean(edge_index):
    """Ensure undirected graph: add reverse edges, remove self-loops, deduplicate."""
    src, dst = edge_index[0], edge_index[1]
    full_src = np.concatenate([src, dst])
    full_dst = np.concatenate([dst, src])
    # Remove self-loops
    mask = full_src != full_dst
    full_src, full_dst = full_src[mask], full_dst[mask]
    # Deduplicate
    pairs = np.stack([full_src, full_dst], axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs.T.astype(np.int64)


def _count_classes(y):
    """Count classes from labels, ignoring negative values (unlabeled markers)."""
    valid = y[y >= 0]
    return int(len(np.unique(valid)))


# PyG loader registry (lazy — torch_geometric imported only on fallback)
_PYG_LOADERS = {
    'cora':               lambda D, r: D.Planetoid(root=r, name='Cora'),
    'citeseer':           lambda D, r: D.Planetoid(root=r, name='CiteSeer'),
    'pubmed':             lambda D, r: D.Planetoid(root=r, name='PubMed'),
    'cornell':            lambda D, r: D.WebKB(root=r, name='Cornell'),
    'wisconsin':          lambda D, r: D.WebKB(root=r, name='Wisconsin'),
    'texas':              lambda D, r: D.WebKB(root=r, name='Texas'),
    'actor':              lambda D, r: D.Actor(root=r),
    'chameleon':          lambda D, r: D.WikipediaNetwork(root=r, name='chameleon'),
    'squirrel':           lambda D, r: D.WikipediaNetwork(root=r, name='squirrel'),
    'chameleon-filtered': lambda D, r: D.HeterophilousGraphDataset(root=r, name='Chameleon-filtered'),
    'squirrel-filtered':  lambda D, r: D.HeterophilousGraphDataset(root=r, name='Squirrel-filtered'),
    'roman-empire':       lambda D, r: D.HeterophilousGraphDataset(root=r, name='Roman-Empire'),
    'amazon-ratings':     lambda D, r: D.HeterophilousGraphDataset(root=r, name='Amazon-Ratings'),
    'amazon-photo':       lambda D, r: D.Amazon(root=r, name='Photo'),
    'amazon-computers':   lambda D, r: D.Amazon(root=r, name='Computers'),
    'deezer-europe':      lambda D, r: D.DeezerEurope(root=r),
    'blogcatalog':        lambda D, r: D.AttributedGraphDataset(root=r, name='BlogCatalog'),
    'flickr':             lambda D, r: D.AttributedGraphDataset(root=r, name='Flickr'),
}


def load_dataset(name, npz_dir=NPZ_DIR):
    """
    Load a node classification dataset by name.

    Loading priority: local npz → PyG download.

    Args:
        name:    dataset name (case-insensitive, hyphens/underscores ok)
        npz_dir: path to directory with .npz files

    Returns:
        edge_index:  (2, E) int64 numpy array
        X:           (N, d) float32 numpy array
        y:           (N,)   int numpy array
        num_classes: int
    """
    name = name.lower().replace('_', '-')

    # OGB datasets have their own loader
    if name.startswith('ogbn-'):
        return _load_ogb(name)

    if name == 'genius':
        return _load_linkx('genius')


    # Try local npz first
    npz_file = os.path.join(npz_dir, f"{name.replace('-', '_')}.npz")
    if os.path.exists(npz_file):
        return _load_npz(npz_file)

    # Fallback to PyG
    return _load_pyg(name)


def _load_npz(path):
    """Load dataset from a local .npz file."""
    raw = np.load(path)
    X = raw['node_features'].astype(np.float32)
    y = raw['node_labels'].astype(int).flatten()
    edges = raw['edges'].astype(np.int64)

    # Normalize to (2, E)
    if edges.ndim == 2 and edges.shape[1] == 2 and edges.shape[0] != 2:
        edge_index = edges.T  # (E, 2) -> (2, E)
    elif edges.ndim == 2 and edges.shape[0] == 2:
        edge_index = edges    # already (2, E)
    else:
        raise ValueError(f"Unexpected edges shape {edges.shape} in {path}")

    edge_index = _symmetrize_and_clean(edge_index)
    num_classes = _count_classes(y)
    return edge_index, X, y, num_classes



def _load_linkx(name, root=None):
    """Load a LINKX dataset (Lim et al., NeurIPS 2021) via PyG.

    Downloads on first use. Override the cache location with
    NSA_LINKX_ROOT (default: ./linkx_data).
    """
    root = root or os.environ.get('NSA_LINKX_ROOT', 'linkx_data')
    import torch
    from torch_geometric.datasets import LINKXDataset
    from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
    torch.serialization.add_safe_globals(
        [Data, DataEdgeAttr, DataTensorAttr, GlobalStorage, NodeStorage, EdgeStorage])
    d = LINKXDataset(root=root, name=name)[0]
    X = d.x.numpy().astype(np.float32)
    y = d.y.numpy().reshape(-1).astype(int)
    edge_index = _symmetrize_and_clean(d.edge_index.numpy().astype(np.int64))
    return edge_index, X, y, _count_classes(y)


def _load_ogb(name, root=None):
    """Load an OGB node-property-prediction dataset.

    ToUndirected() is applied because published ogbn-arxiv numbers assume it.
    Labels arrive as (N, 1) and are flattened to (N,).

    Downloads on first use. Override the cache location with
    NSA_OGB_ROOT (default: ./ogb_data).
    """
    root = root or os.environ.get('NSA_OGB_ROOT', 'ogb_data')
    import torch
    from ogb.nodeproppred import PygNodePropPredDataset
    import torch_geometric.transforms as T

    # torch>=2.6 defaults torch.load(weights_only=True); OGB reloads its cached
    # PyG objects with a plain torch.load, so allowlist the PyG containers.
    from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
    torch.serialization.add_safe_globals(
        [Data, DataEdgeAttr, DataTensorAttr, GlobalStorage, NodeStorage, EdgeStorage])

    ds = PygNodePropPredDataset(name=name, root=root, transform=T.ToUndirected())
    data = ds[0]
    X = data.x.numpy().astype(np.float32)
    y = data.y.numpy().reshape(-1).astype(int)
    edge_index = _symmetrize_and_clean(data.edge_index.numpy().astype(np.int64))
    return edge_index, X, y, _count_classes(y)


def _load_pyg(name):
    """Load dataset via PyG (downloads on first use)."""
    try:
        import torch
        from torch_geometric import datasets as D
    except ImportError:
        raise ImportError(
            f"Dataset '{name}' not found in local npz files and "
            f"torch_geometric is not installed. Either place the npz file "
            f"in {NPZ_DIR}/ or install PyG: pip install torch_geometric")

    if name not in _PYG_LOADERS:
        raise ValueError(
            f"Unknown dataset: '{name}'. "
            f"Available: {sorted(_PYG_LOADERS.keys())}")

    ds = _PYG_LOADERS[name](D, '/tmp/datasets')
    data = ds[0]

    x = data.x
    if x.is_sparse or (hasattr(x, 'is_sparse_csr') and x.is_sparse_csr):
        x = x.to_dense()

    edge_index = data.edge_index.numpy().astype(np.int64)
    X = x.numpy().astype(np.float32)
    y = data.y.numpy().astype(int).flatten()

    edge_index = _symmetrize_and_clean(edge_index)
    num_classes = _count_classes(y)
    return edge_index, X, y, num_classes
