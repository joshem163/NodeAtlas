"""
Benchmark Metadata & Data Loader for Node Signal Atlas (NSA)
=============================================================
Single entry point for all 18 benchmark datasets (Table 6 in paper).

Dataset sources:
  WebKB [PWC+19], multi-scale attributed node embeddings [RAS21],
  heterophilous benchmarks [PKD+23], citation networks [MNRS00],
  co-purchase graphs [SMBG18].

Metadata:
  DATASET_INFO  -- (N, E, d, C, h_edge, h_node, category) per dataset
  ALL_DATASETS  -- ordered list by node homophily

Loading priority:
  1. Local npz files in benchmark_npz_all/  (fast, offline, reproducible)
  2. PyG download fallback                  (for missing npz)

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
    # Heterophilic
    'roman-empire':       (22662,  32927,     300,    18, 0.047, 0.046, 'heterophilic'),
    'texas':              (183,    309,       1703,   5,  0.062, 0.056, 'heterophilic'),
    'wisconsin':          (251,    499,       1703,   5,  0.170, 0.099, 'heterophilic'),
    'squirrel-filtered':  (2223,   46998,     2089,   5,  0.207, 0.191, 'heterophilic'),
    'cornell':            (183,    295,       1703,   5,  0.298, 0.200, 'heterophilic'),
    'actor':              (7600,   33544,     931,    5,  0.216, 0.202, 'heterophilic'),
    'squirrel':           (5201,   217073,    2089,   5,  0.223, 0.215, 'heterophilic'),
    'flickr':             (7575,   239738,    12047,  9,  0.239, 0.243, 'heterophilic'),
    'chameleon-filtered': (890,    8854,      2325,   5,  0.236, 0.244, 'heterophilic'),
    'chameleon':          (2277,   36101,     2325,   5,  0.234, 0.247, 'heterophilic'),
    'amazon-ratings':     (24492,  93050,     300,    5,  0.380, 0.376, 'heterophilic'),
    'blogcatalog':        (5196,   171743,    8189,   6,  0.401, 0.391, 'heterophilic'),
    # Homophilic
    'deezer-europe':      (28281,  92752,     31241,  2,  0.525, 0.530, 'homophilic'),
    'citeseer':           (3327,   4732,      3703,   6,  0.736, 0.706, 'homophilic'),
    'pubmed':             (19717,  44338,     500,    3,  0.802, 0.792, 'homophilic'),
    'amazon-computers':   (13752,  245861,    767,    10, 0.780, 0.800, 'homophilic'),
    'cora':               (2708,   5429,      1433,   7,  0.810, 0.825, 'homophilic'),
    'amazon-photo':       (7650,   119081,    745,    8,  0.830, 0.850, 'homophilic'),
}

DATASET_GROUPS = {
    'heterophilic': ['roman-empire', 'texas', 'wisconsin', 'squirrel-filtered',
                     'cornell', 'actor', 'squirrel', 'flickr',
                     'chameleon-filtered', 'chameleon', 'amazon-ratings',
                     'blogcatalog'],
    'homophilic':   ['deezer-europe', 'citeseer', 'pubmed',
                     'amazon-computers', 'cora', 'amazon-photo'],
}

ALL_DATASETS = (
    DATASET_GROUPS['heterophilic'] +
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
