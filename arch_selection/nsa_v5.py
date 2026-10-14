"""
NSA synthetic interface-selection benchmark (v5)
================================================

Vendored verbatim from the authors' frozen benchmark specification; the
file is not modified by the analysis scripts in this directory. Renamed
from NSA_v5.py for import consistency.

This is the proposed primary synthetic architecture-selection experiment for
the ICLR revision of Node Signal Atlas (NSA).

Scientific question
-------------------
Given a previously unseen node-classification problem, can a compact
pre-training dataset diagnostic help prioritize the appropriate SIGNAL
INTERFACE?  The primary experiment intentionally uses specialist models rather
than flexible generalists such as GPS or GPR-GNN.

Primary architecture bank
-------------------------
  * MLP      : feature-only interface
  * GCN      : local positive message passing
  * GraphSAGE: local ego + neighborhood aggregation
  * APPNP    : long-range positive diffusion
  * H2GCN    : separated ego / 1-hop / exact-2-hop heterophily interface
  * GIN      : sum-aggregation / degree-sensitive structural interface

GPS and GPR-GNN are not part of this primary synthetic benchmark. They belong
to the separate generalist/hybrid stress-test story (including the real
benchmark experiment).

Primary signal regimes
----------------------
  * feature_signal
  * smooth_diffusion
  * structured_heterophily
  * structural_role
  * neighbor_feature_context
  * mixed_signal

These regimes are NOT assigned target architectures. The winner is always
determined empirically.

Pilot -> freeze -> final discipline
-----------------------------------
The code separates three stages:

1. --audit-only
   Cheap generator + NSA implementation sanity check. This is allowed to catch
   NaNs, saturation, descriptor leakage, and failures of already-documented NSA
   behavior. It must NOT be used to choose parameters because NSA beats or
   loses to homophily.

2. --pilot
   Architecture-only benchmark calibration. NSA and the representation
   predictors are NOT computed. The pilot inspects only architecture
   performance, winner diversity, selection headroom, and nuisance graph
   statistics.

3. default final run
   Uses fresh corpus seeds. Only after the benchmark specification is frozen
   are NSA, homophily, and meta-feature predictors compared.

Pre-specified representation comparison
---------------------------------------
All learned representations use the EXACT SAME
StandardScaler + multi-output Ridge predictor:

  * NodeHomophily-1
  * Homophily-4
  * NSA-3
  * Homophily-4 + Meta-5
  * NSA-3 + Meta-5

Meta-5 is fixed a priori:
  * mean log-degree
  * std. log-degree
  * average clustering coefficient
  * normalized spectral gap
  * mean edge-feature cosine similarity

Important
---------
The NSA scoring code is unchanged. In particular, LSS remains the paper's
current learned label-context probe and may respond to structured heterophily.
The benchmark must not be altered merely to make NSA outperform homophily.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import APPNP as APPNPProp
    from torch_geometric.nn import GCNConv, SAGEConv, GINConv
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import to_undirected
    PYG_AVAILABLE = True
except Exception:  # pragma: no cover
    Data = Any
    APPNPProp = GCNConv = SAGEConv = GINConv = MessagePassing = gcn_norm = None
    to_undirected = None
    PYG_AVAILABLE = False


# =============================================================================
# Global conventions
# =============================================================================

GENERATOR_VERSION = "v5_specialist_interfaces_2026_08_31"
MODEL_BANK_VERSION = "v5_mlp_gcn_sage_appnp_h2gcn_gin_2026_08_31"
EXPERIMENT_VERSION = "v5_pilot_freeze_final_2026_08_31"

N_CLASSES = 3
SYNTHETIC_N = 512
FEATURE_DIM = 8

NATIVE_MECHANISMS: Tuple[str, ...] = (
    "feature_signal",
    "smooth_diffusion",
    "structured_heterophily",
    "structural_role",
    "neighbor_feature_context",
    "mixed_signal",
)

ARCHITECTURE_NAMES: Tuple[str, ...] = (
    "mlp",
    "gcn",
    "sage",
    "appnp",
    "h2gcn",
    "gin",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NSA protocol constants -- intentionally unchanged.
NSA_R = 32
NSA_WL_ROUNDS = 4
NSA_WL_BUCKETS = 1024
NSA_MK = 64
NSA_SEEDS: Tuple[int, ...] = (0, 1, 2)


def active_architectures(cfg: "ExperimentConfig") -> Tuple[str, ...]:
    excluded = set(cfg.exclude_architectures)
    return tuple(a for a in ARCHITECTURE_NAMES if a not in excluded)


def _require_pyg() -> None:
    if not PYG_AVAILABLE:
        raise ImportError(
            "torch_geometric is required for architecture training. Install a "
            "PyTorch-Geometric build compatible with your PyTorch version."
        )


def set_all_seeds(seed: int) -> None:
    # sklearn RandomState requires a 32-bit seed.
    seed32 = int(seed) % (2**32 - 1)
    random.seed(seed32)
    np.random.seed(seed32)
    torch.manual_seed(seed32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed32)


@dataclass
class ArchConfig:
    hidden: int = 64
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 300
    patience: int = 50
    dropout: float = 0.5
    hops: int = 10
    alpha: float = 0.1
    h2_layers: int = 2


DEFAULT_ARCH_CONFIG: Dict[str, ArchConfig] = {
    "mlp": ArchConfig(hidden=64, lr=0.01, weight_decay=5e-4),
    "gcn": ArchConfig(hidden=64, lr=0.01, weight_decay=5e-4),
    "sage": ArchConfig(hidden=64, lr=0.01, weight_decay=5e-4),
    "appnp": ArchConfig(hidden=64, lr=0.01, weight_decay=5e-4, hops=10, alpha=0.1),
    "h2gcn": ArchConfig(hidden=64, lr=0.01, weight_decay=5e-4, h2_layers=2),
    "gin": ArchConfig(hidden=64, lr=0.01, weight_decay=5e-4),
}


@dataclass
class ExperimentConfig:
    synthetic_n: int = SYNTHETIC_N
    n_classes: int = N_CLASSES

    # Final corpus.
    train_problems_per_mechanism: int = 30
    test_problems_per_mechanism: int = 15
    model_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)

    # Architecture-only pilot.
    pilot_train_per_mechanism: int = 6
    pilot_test_per_mechanism: int = 6
    pilot_model_seeds: Tuple[int, ...] = (0, 1)

    meta_ridge_alpha: float = 1.0
    random_baseline_draws: int = 500
    cache_dir: str = "./cache_v5"
    exclude_architectures: Tuple[str, ...] = ()

    # PRE-SPECIFIED benchmark-headroom criteria.
    max_winner_share: float = 0.45
    min_distinct_winners: int = 4
    min_constant_regret: float = 0.05
    min_median_spread: float = 0.08
    min_fraction_spread_gt_005: float = 0.70
    max_fixed_top3_hit: float = 0.80
    min_median_best_second_gap: float = 0.01

    arch: Dict[str, ArchConfig] = field(default_factory=lambda: copy.deepcopy(DEFAULT_ARCH_CONFIG))

    @classmethod
    def quick(cls) -> "ExperimentConfig":
        c = cls()
        c.synthetic_n = 150
        c.train_problems_per_mechanism = 2
        c.test_problems_per_mechanism = 1
        c.model_seeds = (0,)
        c.pilot_train_per_mechanism = 1
        c.pilot_test_per_mechanism = 1
        c.pilot_model_seeds = (0,)
        c.random_baseline_draws = 50
        for a in c.arch.values():
            a.epochs = 25
            a.patience = 6
            a.hidden = 32
        return c


# =============================================================================
# Problem container
# =============================================================================

@dataclass
class Problem:
    name: str
    G: nx.Graph
    X: np.ndarray
    y: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    metric: str = "accuracy"

    def y_dict(self, mask: np.ndarray | None = None) -> Dict[int, int]:
        idx = np.arange(len(self.y)) if mask is None else np.flatnonzero(mask)
        return {int(i): int(self.y[i]) for i in idx}

    @property
    def n_classes(self) -> int:
        return int(np.max(self.y)) + 1


def stratified_masks(y: np.ndarray, seed: int, train_frac: float = 0.6,
                     val_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        idx = rng.permutation(idx)
        nt = max(1, int(round(train_frac * len(idx))))
        nv = max(1, int(round(val_frac * len(idx)))) if len(idx) >= 5 else 0
        if nt + nv >= len(idx):
            nt = max(1, len(idx) - 2)
            nv = 1 if len(idx) - nt >= 2 else 0
        train[idx[:nt]] = True
        val[idx[nt:nt + nv]] = True
        test[idx[nt + nv:]] = True
    return train, val, test


_TWO_HOP_EDGE_CACHE: Dict[str, torch.Tensor] = {}


def _two_hop_edge_index(problem: Problem) -> torch.Tensor:
    """Directed exact-distance-2 edge index, excluding self and 1-hop edges.

    Cached per problem name because H2GCN is trained for multiple seeds.
    """
    key = problem.name
    if key in _TWO_HOP_EDGE_CACHE:
        return _TWO_HOP_EDGE_CACHE[key]

    G = problem.G
    src: List[int] = []
    dst: List[int] = []
    for v in G.nodes():
        n1 = set(G.neighbors(v))
        n2: set[int] = set()
        for u in n1:
            n2.update(G.neighbors(u))
        n2.discard(v)
        n2.difference_update(n1)
        for u in n2:
            src.append(int(u))
            dst.append(int(v))

    if src:
        out = torch.tensor([src, dst], dtype=torch.long)
    else:
        out = torch.empty((2, 0), dtype=torch.long)
    _TWO_HOP_EDGE_CACHE[key] = out
    return out


def problem_to_pyg(problem: Problem) -> "Data":
    _require_pyg()
    edges = np.asarray(list(problem.G.edges()), dtype=np.int64)
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        edge_index = to_undirected(edge_index, num_nodes=len(problem.y))

    return Data(
        x=torch.tensor(problem.X, dtype=torch.float32),
        edge_index=edge_index,
        edge_index_2=_two_hop_edge_index(problem).clone(),
        y=torch.tensor(problem.y, dtype=torch.long),
        train_mask=torch.tensor(problem.train_mask, dtype=torch.bool),
        val_mask=torch.tensor(problem.val_mask, dtype=torch.bool),
        test_mask=torch.tensor(problem.test_mask, dtype=torch.bool),
    )


# =============================================================================
# Shared generator helpers
# =============================================================================

def _stitch(G: nx.Graph) -> None:
    comps = list(nx.connected_components(G))
    for a, b in zip(comps[:-1], comps[1:]):
        G.add_edge(next(iter(a)), next(iter(b)))


def _sbm_from_assignments(assign: np.ndarray, P: np.ndarray, rng: np.random.Generator) -> nx.Graph:
    n = len(assign)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    n_blocks = P.shape[0]
    blocks = [np.flatnonzero(assign == b) for b in range(n_blocks)]
    for a in range(n_blocks):
        aa = blocks[a]
        if len(aa) > 1 and P[a, a] > 0:
            for ii in range(len(aa)):
                js = aa[ii + 1:]
                if len(js):
                    keep = rng.random(len(js)) < P[a, a]
                    G.add_edges_from((int(aa[ii]), int(v)) for v in js[keep])
        for b in range(a + 1, n_blocks):
            bb = blocks[b]
            if not len(aa) or not len(bb) or P[a, b] <= 0:
                continue
            for u in aa:
                keep = rng.random(len(bb)) < P[a, b]
                G.add_edges_from((int(u), int(v)) for v in bb[keep])
    _stitch(G)
    return G


def _balanced_labels(n: int, n_classes: int, rng: np.random.Generator) -> np.ndarray:
    base, rem = divmod(n, n_classes)
    counts = [base + (1 if i < rem else 0) for i in range(n_classes)]
    labels = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    rng.shuffle(labels)
    return labels


def _knn_edges(pos: np.ndarray, k: int, allowed: np.ndarray | None = None) -> List[Tuple[int, int]]:
    n = len(pos)
    edges: set[Tuple[int, int]] = set()
    for i in range(n):
        dist = np.linalg.norm(pos - pos[i], axis=1)
        dist[i] = np.inf
        order = np.argsort(dist)
        count = 0
        for j in order:
            if allowed is not None and not allowed[i, j]:
                continue
            a, b = sorted((i, int(j)))
            edges.add((a, b))
            count += 1
            if count >= k:
                break
    return list(edges)



# =============================================================================
# Synthetic signal regimes (v5)
# =============================================================================

def _unit_directions(n_classes: int, d: int, rng: np.random.Generator) -> np.ndarray:
    D = rng.normal(size=(n_classes, d))
    return D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)


def _class_features(y: np.ndarray, d: int, strength: float,
                    rng: np.random.Generator) -> np.ndarray:
    D = _unit_directions(int(np.max(y)) + 1, d, rng)
    return rng.normal(size=(len(y), d)) + strength * D[y]


def _weighted_pair_sample(n: int, m: int, pair_weight: np.ndarray,
                          rng: np.random.Generator) -> nx.Graph:
    """Sample an undirected simple graph with exactly m weighted edges."""
    ii, jj = np.triu_indices(n, 1)
    m = int(min(max(m, n - 1), len(ii)))
    w = np.asarray(pair_weight, dtype=float)
    if len(w) != len(ii):
        raise ValueError("pair_weight has wrong size")
    w = np.maximum(w, 1e-12)
    p = w / w.sum()
    take = rng.choice(len(ii), size=m, replace=False, p=p)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((int(ii[t]), int(jj[t])) for t in take)
    _stitch(G)
    return G


def _weighted_random_graph(n: int, avg_degree: float,
                           rng: np.random.Generator) -> nx.Graph:
    """Low-clustering test-family analogue of ER with mild degree heterogeneity."""
    ii, jj = np.triu_indices(n, 1)
    node_w = rng.lognormal(mean=0.0, sigma=0.15, size=n)
    pair_w = node_w[ii] * node_w[jj]
    m = int(round(n * avg_degree / 2))
    return _weighted_pair_sample(n, m, pair_w, rng)


def _sbm_mixing_graph(y: np.ndarray, avg_degree: float, same_fraction: float,
                      rng: np.random.Generator) -> nx.Graph:
    """Train-family SBM approximately calibrated by degree and homophily."""
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    mean_same_candidates = float(np.mean(counts - 1))
    mean_diff_candidates = float(n - np.mean(counts))
    p_in = min(0.95, avg_degree * same_fraction / max(mean_same_candidates, 1.0))
    p_out = min(0.95, avg_degree * (1.0 - same_fraction) / max(mean_diff_candidates, 1.0))
    P = np.full((len(classes), len(classes)), p_out, dtype=float)
    np.fill_diagonal(P, p_in)
    return _sbm_from_assignments(y, P, rng)


def _weighted_mixing_graph(y: np.ndarray, avg_degree: float, same_fraction: float,
                           rng: np.random.Generator) -> nx.Graph:
    """Test-family fixed-edge degree-corrected mixing graph.

    It has low clustering like the SBM family but is generated by weighted
    sampling rather than independent Bernoulli edges.
    """
    n = len(y)
    ii, jj = np.triu_indices(n, 1)
    same = (y[ii] == y[jj])
    node_w = rng.lognormal(mean=0.0, sigma=0.15, size=n)
    base = node_w[ii] * node_w[jj]

    s_same = float(base[same].sum())
    s_diff = float(base[~same].sum())
    h = float(np.clip(same_fraction, 1e-3, 1 - 1e-3))
    ratio = (h / (1.0 - h)) * (s_diff / max(s_same, 1e-12))
    pair_w = base * np.where(same, ratio, 1.0)

    m = int(round(n * avg_degree / 2))
    return _weighted_pair_sample(n, m, pair_w, rng)


def _role_expected_degree_graph(role: np.ndarray, avg_degree: float,
                                multipliers: np.ndarray,
                                rng: np.random.Generator) -> nx.Graph:
    w = multipliers[role].astype(float)
    w *= avg_degree / max(float(np.mean(w)), 1e-12)
    G = nx.expected_degree_graph(
        w.tolist(), selfloops=False, seed=int(rng.integers(1, 2**31 - 1))
    )
    G = nx.Graph(G)
    G.add_nodes_from(range(len(role)))
    _stitch(G)
    return G


def _role_fixed_edge_graph(role: np.ndarray, avg_degree: float,
                           multipliers: np.ndarray,
                           rng: np.random.Generator) -> nx.Graph:
    n = len(role)
    ii, jj = np.triu_indices(n, 1)
    node_w = multipliers[role].astype(float)
    node_w *= rng.lognormal(mean=0.0, sigma=0.10, size=n)
    pair_w = node_w[ii] * node_w[jj]
    m = int(round(n * avg_degree / 2))
    return _weighted_pair_sample(n, m, pair_w, rng)


def _heterophily_blocks(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Split each class into two latent blocks.

    Preferred block pairs are:
        0A <-> 1A, 1B <-> 2A, 2B <-> 0B.
    Therefore 1-hop neighbors are overwhelmingly different-label while
    exact 2-hop walks return to the same latent block/class.
    """
    block = np.empty(len(y), dtype=int)
    for c in range(3):
        idx = rng.permutation(np.flatnonzero(y == c))
        cut = len(idx) // 2
        block[idx[:cut]] = 2 * c
        block[idx[cut:]] = 2 * c + 1
    return block


_HET_PAIR = {0: 2, 2: 0, 3: 4, 4: 3, 5: 1, 1: 5}


def _heterophily_train_graph(block: np.ndarray, avg_degree: float,
                             noise_fraction: float,
                             rng: np.random.Generator) -> nx.Graph:
    n_blocks = 6
    n = len(block)
    counts = np.bincount(block, minlength=n_blocks).astype(float)
    pair_candidates = float(np.mean([counts[_HET_PAIR[b]] for b in range(n_blocks)]))
    other_candidates = max(float(n - 1 - pair_candidates), 1.0)

    p_pair = min(0.95, avg_degree * (1.0 - noise_fraction) / max(pair_candidates, 1.0))
    p_noise = min(0.95, avg_degree * noise_fraction / other_candidates)

    P = np.full((n_blocks, n_blocks), p_noise, dtype=float)
    np.fill_diagonal(P, p_noise)
    for b, q in _HET_PAIR.items():
        P[b, q] = p_pair
    P = np.maximum(P, P.T)
    return _sbm_from_assignments(block, P, rng)


def _heterophily_test_graph(block: np.ndarray, avg_degree: float,
                            noise_fraction: float,
                            rng: np.random.Generator) -> nx.Graph:
    n = len(block)
    ii, jj = np.triu_indices(n, 1)
    preferred = np.array([_HET_PAIR[int(block[a])] == int(block[b])
                          for a, b in zip(ii, jj)], dtype=bool)
    node_w = rng.lognormal(mean=0.0, sigma=0.12, size=n)
    base = node_w[ii] * node_w[jj]

    pref_target = float(np.clip(1.0 - noise_fraction, 0.01, 0.99))
    s_pref = float(base[preferred].sum())
    s_other = float(base[~preferred].sum())
    ratio = (pref_target / (1.0 - pref_target)) * (s_other / max(s_pref, 1e-12))
    pair_w = base * np.where(preferred, ratio, 1.0)

    m = int(round(n * avg_degree / 2))
    return _weighted_pair_sample(n, m, pair_w, rng)


def _neighbor_context_train_graph(y: np.ndarray, token: np.ndarray,
                                  avg_degree: float, preferred_fraction: float,
                                  rng: np.random.Generator) -> nx.Graph:
    """Per-node random-choice construction.

    A node of label c preferentially connects to nodes whose FEATURE TOKEN is c.
    The node's own token is independent of its label, so raw X alone is weak,
    while neighborhood feature composition is predictive.
    """
    n = len(y)
    k = max(1, int(round(avg_degree / 2.0)))
    edges: set[Tuple[int, int]] = set()
    all_idx = np.arange(n)
    for i in range(n):
        preferred = np.flatnonzero((token == y[i]) & (all_idx != i))
        other = np.flatnonzero((token != y[i]) & (all_idx != i))
        kp = min(len(preferred), int(round(k * preferred_fraction)))
        ko = min(len(other), k - kp)
        if kp:
            js = rng.choice(preferred, size=kp, replace=False)
            edges.update(tuple(sorted((i, int(j)))) for j in js)
        if ko:
            js = rng.choice(other, size=ko, replace=False)
            edges.update(tuple(sorted((i, int(j)))) for j in js)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    _stitch(G)
    return G


def _neighbor_context_test_graph(y: np.ndarray, token: np.ndarray,
                                 avg_degree: float, preferred_fraction: float,
                                 rng: np.random.Generator) -> nx.Graph:
    """Different fixed-edge pair-sampling implementation of the same mechanism."""
    n = len(y)
    ii, jj = np.triu_indices(n, 1)
    # Symmetric compatibility: either endpoint's token carries information
    # requested by the other endpoint's label.
    compatible = ((token[jj] == y[ii]) | (token[ii] == y[jj]))
    node_w = rng.lognormal(mean=0.0, sigma=0.12, size=n)
    base = node_w[ii] * node_w[jj]

    q = float(np.clip(preferred_fraction, 0.51, 0.98))
    s_good = float(base[compatible].sum())
    s_bad = float(base[~compatible].sum())
    ratio = (q / (1.0 - q)) * (s_bad / max(s_good, 1e-12))
    pair_w = base * np.where(compatible, ratio, 1.0)

    m = int(round(n * avg_degree / 2))
    return _weighted_pair_sample(n, m, pair_w, rng)


def _sample_regime_params(mechanism: str, rng: np.random.Generator) -> Dict[str, Any]:
    """Parameter ranges are shared by train/test generator families.

    These are PILOT parameters. They may be revised using ONLY architecture
    headroom + nuisance-statistic criteria, then must be frozen before the
    final descriptor comparison.
    """
    if mechanism == "feature_signal":
        return {
            "avg_degree": rng.uniform(10.0, 16.0),
            "feature_strength": rng.uniform(1.25, 1.80),
        }

    if mechanism == "smooth_diffusion":
        # Deliberately sparse: prevents neighborhood feature averaging from
        # turning weak class features into an artificial high-FSS signal.
        return {
            "avg_degree": rng.uniform(2.5, 3.5),
            "homophily": rng.uniform(0.80, 0.92),
            "feature_strength": rng.uniform(0.05, 0.18),
        }

    if mechanism == "structured_heterophily":
        return {
            "avg_degree": rng.uniform(3.5, 5.0),
            "noise_fraction": rng.uniform(0.08, 0.18),
            "feature_strength": rng.uniform(0.03, 0.12),
        }

    if mechanism == "structural_role":
        base = np.array([0.40, 1.00, 2.00], dtype=float)
        rng.shuffle(base)
        return {
            "avg_degree": rng.uniform(6.0, 10.0),
            "multipliers": base,
            "feature_strength": rng.uniform(0.00, 0.08),
        }

    if mechanism == "neighbor_feature_context":
        return {
            "avg_degree": rng.uniform(6.0, 10.0),
            "preferred_fraction": rng.uniform(0.72, 0.88),
            "token_strength": rng.uniform(1.30, 1.90),
        }

    if mechanism == "mixed_signal":
        return {
            "avg_degree": rng.uniform(4.0, 8.0),
            "homophily": rng.uniform(0.50, 0.70),
            "feature_strength": rng.uniform(0.65, 1.00),
        }

    raise ValueError(mechanism)


def _generate_problem(family: str, mechanism: str, n: int, seed: int,
                      n_classes: int = N_CLASSES) -> Tuple[nx.Graph, np.ndarray, Dict[int, int]]:
    if n_classes != 3:
        raise ValueError("v5 generators currently assume exactly 3 classes.")

    rng = np.random.default_rng(seed)
    d = FEATURE_DIM
    p = _sample_regime_params(mechanism, rng)

    if mechanism == "feature_signal":
        y = _balanced_labels(n, n_classes, rng)
        X = _class_features(y, d, p["feature_strength"], rng)
        if family == "train":
            G = nx.gnp_random_graph(
                n,
                p=p["avg_degree"] / max(n - 1, 1),
                seed=int(rng.integers(1, 2**31 - 1)),
            )
            G.add_nodes_from(range(n))
            _stitch(G)
        elif family == "test":
            G = _weighted_random_graph(n, p["avg_degree"], rng)
        else:
            raise ValueError(family)

    elif mechanism == "smooth_diffusion":
        y = _balanced_labels(n, n_classes, rng)
        X = _class_features(y, d, p["feature_strength"], rng)
        if family == "train":
            G = _sbm_mixing_graph(y, p["avg_degree"], p["homophily"], rng)
        elif family == "test":
            G = _weighted_mixing_graph(y, p["avg_degree"], p["homophily"], rng)
        else:
            raise ValueError(family)

    elif mechanism == "structured_heterophily":
        y = _balanced_labels(n, n_classes, rng)
        block = _heterophily_blocks(y, rng)
        X = _class_features(y, d, p["feature_strength"], rng)
        if family == "train":
            G = _heterophily_train_graph(
                block, p["avg_degree"], p["noise_fraction"], rng
            )
        elif family == "test":
            G = _heterophily_test_graph(
                block, p["avg_degree"], p["noise_fraction"], rng
            )
        else:
            raise ValueError(family)

    elif mechanism == "structural_role":
        y = _balanced_labels(n, n_classes, rng)
        X = _class_features(y, d, p["feature_strength"], rng)
        if family == "train":
            G = _role_expected_degree_graph(
                y, p["avg_degree"], p["multipliers"], rng
            )
        elif family == "test":
            G = _role_fixed_edge_graph(
                y, p["avg_degree"], p["multipliers"], rng
            )
        else:
            raise ValueError(family)

    elif mechanism == "neighbor_feature_context":
        y = _balanced_labels(n, n_classes, rng)
        token = _balanced_labels(n, n_classes, rng)  # independent of y
        D = _unit_directions(n_classes, d, rng)
        X = rng.normal(size=(n, d)) + p["token_strength"] * D[token]
        if family == "train":
            G = _neighbor_context_train_graph(
                y, token, p["avg_degree"], p["preferred_fraction"], rng
            )
        elif family == "test":
            G = _neighbor_context_test_graph(
                y, token, p["avg_degree"], p["preferred_fraction"], rng
            )
        else:
            raise ValueError(family)

    elif mechanism == "mixed_signal":
        y = _balanced_labels(n, n_classes, rng)
        X = _class_features(y, d, p["feature_strength"], rng)
        if family == "train":
            G = _sbm_mixing_graph(y, p["avg_degree"], p["homophily"], rng)
        elif family == "test":
            G = _weighted_mixing_graph(y, p["avg_degree"], p["homophily"], rng)
        else:
            raise ValueError(family)

    else:
        raise ValueError(mechanism)

    return G, np.asarray(X, dtype=np.float32), {
        int(i): int(y[i]) for i in range(n)
    }


def generate_train_problem(mechanism: str, n: int, seed: int,
                           n_classes: int = N_CLASSES) -> Tuple[nx.Graph, np.ndarray, Dict[int, int]]:
    return _generate_problem("train", mechanism, n, seed, n_classes)


def generate_test_problem(mechanism: str, n: int, seed: int,
                          n_classes: int = N_CLASSES) -> Tuple[nx.Graph, np.ndarray, Dict[int, int]]:
    return _generate_problem("test", mechanism, n, seed, n_classes)


# =============================================================================
# Conventional/meta-feature competitors
# =============================================================================

def adjusted_homophily(G: nx.Graph, labels: Mapping[int, int]) -> float:
    nodes = set(labels)
    edges = [(u, v) for u, v in G.edges() if u in nodes and v in nodes]
    if not edges:
        return 0.0
    h = np.mean([labels[u] == labels[v] for u, v in edges])
    endpoint = []
    for u, v in edges:
        endpoint.extend([labels[u], labels[v]])
    _, counts = np.unique(endpoint, return_counts=True)
    p = counts / counts.sum()
    expected = float(np.sum(p ** 2))
    return float((h - expected) / max(1.0 - expected, 1e-12))


def edge_homophily(G: nx.Graph, labels: Mapping[int, int]) -> float:
    nodes = set(labels)
    edges = [(u, v) for u, v in G.edges() if u in nodes and v in nodes]
    if not edges:
        return 0.0
    return float(np.mean([labels[u] == labels[v] for u, v in edges]))


def node_homophily(G: nx.Graph, labels: Mapping[int, int]) -> float:
    vals = []
    for v in labels:
        nb = [u for u in G.neighbors(v) if u in labels]
        if not nb:
            continue
        vals.append(np.mean([labels[u] == labels[v] for u in nb]))
    return float(np.mean(vals)) if vals else 0.0


def label_informativeness(G: nx.Graph, labels: Mapping[int, int]) -> float:
    nodes = set(labels)
    pairs = []
    for u, v in G.edges():
        if u in nodes and v in nodes:
            pairs.extend([(labels[u], labels[v]), (labels[v], labels[u])])
    if not pairs:
        return 0.0
    a = np.array([p[0] for p in pairs], dtype=int)
    b = np.array([p[1] for p in pairs], dtype=int)
    classes = np.unique(np.r_[a, b])
    idx = {c: i for i, c in enumerate(classes)}
    joint = np.zeros((len(classes), len(classes)), dtype=float)
    for x, yy in zip(a, b):
        joint[idx[x], idx[yy]] += 1
    joint /= joint.sum()
    pa, pb = joint.sum(1), joint.sum(0)
    nz = joint > 0
    denom = pa[:, None] * pb[None, :]
    mi = float(np.sum(joint[nz] * np.log((joint[nz] + 1e-15) / (denom[nz] + 1e-15))))
    H = float(-np.sum(pa[pa > 0] * np.log(pa[pa > 0])))
    return mi / max(H, 1e-12)


def _spectral_gap(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    if n < 4 or G.number_of_edges() == 0:
        return 0.0
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    deg = np.asarray(A.sum(1)).ravel()
    dinv = np.zeros_like(deg)
    nz = deg > 0
    dinv[nz] = deg[nz] ** -0.5
    S = sparse.diags(dinv) @ A @ sparse.diags(dinv)
    try:
        top2 = eigsh(S, k=2, which="LA", return_eigenvectors=False)
        top2 = np.sort(top2)[::-1]
        return float(top2[0] - top2[1])
    except Exception:
        return 0.0


def _feature_probe(problem: Problem) -> float:
    idx = np.flatnonzero(problem.train_mask)
    y = problem.y[idx]
    if len(idx) < 10:
        return 0.0
    return _probe_score(problem.X[idx], y, problem.n_classes, seed=173)


def _centroid_fisher(problem: Problem) -> float:
    idx = np.flatnonzero(problem.train_mask)
    X, y = problem.X[idx].astype(float), problem.y[idx]
    mu = X.mean(0)
    between = 0.0
    within = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        if not len(Xc):
            continue
        muc = Xc.mean(0)
        between += len(Xc) * float(np.sum((muc - mu) ** 2))
        within += float(np.sum((Xc - muc) ** 2))
    return float(between / max(within, 1e-12))


def _edge_feature_stats(problem: Problem) -> Tuple[float, float, float]:
    X = problem.X.astype(float)
    edges = np.asarray(list(problem.G.edges()), dtype=int)
    if len(edges) == 0:
        return 0.0, 0.0, 0.0
    a, b = X[edges[:, 0]], X[edges[:, 1]]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    cos = np.sum(a * b, axis=1) / np.maximum(na * nb, 1e-12)
    sqdist = np.sum((a - b) ** 2, axis=1)
    return float(np.mean(cos)), float(np.std(cos)), float(np.mean(sqdist))


def _feature_effective_rank(X: np.ndarray) -> float:
    Xc = X.astype(float) - np.mean(X, axis=0, keepdims=True)
    if min(Xc.shape) < 2:
        return 0.0
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    e = s ** 2
    if e.sum() <= 1e-12:
        return 0.0
    p = e / e.sum()
    H = -np.sum(p[p > 0] * np.log(p[p > 0]))
    return float(np.exp(H) / max(len(p), 1))


def homophily_statistics(problem: Problem, y_observed: Dict[int, int] | None) -> np.ndarray:
    if not y_observed:
        return np.zeros(4, dtype=float)
    return np.array([
        edge_homophily(problem.G, y_observed),
        node_homophily(problem.G, y_observed),
        adjusted_homophily(problem.G, y_observed),
        label_informativeness(problem.G, y_observed),
    ], dtype=float)


HOMOPHILY_STAT_NAMES: Tuple[str, ...] = (
    "edge_homophily", "node_homophily", "adjusted_homophily", "label_informativeness",
)


def conventional_statistics(problem: Problem, y_observed: Dict[int, int] | None) -> np.ndarray:
    G, X = problem.G, problem.X
    h = homophily_statistics(problem, y_observed)
    deg = np.array([dg for _, dg in G.degree()], dtype=float)
    if len(deg) == 0:
        deg = np.zeros(1)
    log_deg = np.log1p(deg)
    try:
        clustering = float(nx.average_clustering(G))
    except Exception:
        clustering = 0.0
    try:
        transitivity = float(nx.transitivity(G))
    except Exception:
        transitivity = 0.0
    try:
        assort = nx.degree_assortativity_coefficient(G)
        assort = float(assort) if np.isfinite(assort) else 0.0
    except Exception:
        assort = 0.0
    try:
        core = np.array(list(nx.core_number(G).values()), dtype=float)
    except Exception:
        core = np.zeros(len(deg), dtype=float)
    edge_cos_mean, edge_cos_std, edge_sqdist_mean = _edge_feature_stats(problem)
    fnorm = np.linalg.norm(X.astype(float), axis=1)

    return np.concatenate([h, np.array([
        float(log_deg.mean()), float(log_deg.std()),
        clustering, transitivity, _spectral_gap(G), assort,
        float(core.mean()) if len(core) else 0.0,
        float(core.std()) if len(core) else 0.0,
        _feature_probe(problem), _centroid_fisher(problem),
        edge_cos_mean, edge_cos_std, edge_sqdist_mean,
        _feature_effective_rank(X),
        float(fnorm.mean()), float(fnorm.std()),
    ], dtype=float)])


CONVENTIONAL_STAT_NAMES: Tuple[str, ...] = HOMOPHILY_STAT_NAMES + (
    "mean_log_degree", "std_log_degree", "avg_clustering", "transitivity",
    "spectral_gap", "degree_assortativity", "mean_core", "std_core",
    "raw_feature_probe", "centroid_fisher", "edge_feature_cos_mean",
    "edge_feature_cos_std", "edge_feature_sqdist_mean", "feature_effective_rank",
    "feature_norm_mean", "feature_norm_std",
)


# Fixed a-priori Meta-5. These are deliberately simple, interpretable,
# mostly label-free dataset statistics. Do not tune this subset after seeing
# benchmark outcomes.
META5_STAT_NAMES: Tuple[str, ...] = (
    "mean_log_degree",
    "std_log_degree",
    "avg_clustering",
    "spectral_gap",
    "edge_feature_cos_mean",
)

# Indices into conventional_statistics():
# 0:3 are Homophily-4, then the generic meta-features follow.
_META5_INDICES: Tuple[int, ...] = tuple(
    CONVENTIONAL_STAT_NAMES.index(name) for name in META5_STAT_NAMES
)

# =============================================================================
# NSA scoring: FSS / TSS / LSS -- UNCHANGED from the validated version
# =============================================================================

def _bfs_annuli(G: nx.Graph, max_hop: int = 3) -> Dict[int, Dict[int, np.ndarray]]:
    annuli: Dict[int, Dict[int, np.ndarray]] = {}
    for v in G.nodes():
        dist = nx.single_source_shortest_path_length(G, v, cutoff=max_hop)
        by_hop: Dict[int, List[int]] = {k: [] for k in range(max_hop + 1)}
        for u, dd in dist.items():
            if u != v:
                by_hop[dd].append(u)
        annuli[v] = {k: np.asarray(vs, dtype=np.int64) for k, vs in by_hop.items()}
    return annuli


def _sample_annulus(nodes: np.ndarray, m_k: int, rng: np.random.Generator) -> np.ndarray:
    return nodes if len(nodes) <= m_k else rng.choice(nodes, size=m_k, replace=False)


def _fss_descriptor(X: np.ndarray, annuli: Dict[int, Dict[int, np.ndarray]], seed: int) -> np.ndarray:
    n, d = X.shape
    rng = np.random.default_rng(seed)
    Xr = X @ (rng.normal(size=(d, NSA_R)) / np.sqrt(NSA_R))
    feats = np.zeros((n, 4 * (2 * NSA_R + 1)))
    for v in range(n):
        blocks = [np.concatenate([Xr[v], np.zeros(NSA_R), [0.0]])]
        for k in (1, 2, 3):
            nodes_k = annuli[v][k]
            if len(nodes_k) == 0:
                blocks.append(np.zeros(2 * NSA_R + 1))
                continue
            block = Xr[_sample_annulus(nodes_k, NSA_MK, rng)]
            mu, sigma = block.mean(0), block.std(0)
            blocks.append(np.concatenate([mu, sigma, [np.linalg.norm(mu - Xr[v])]]))
        feats[v] = np.concatenate(blocks)
    return feats


def _wl_role_entropy(G: nx.Graph, annuli: Dict[int, Dict[int, np.ndarray]]) -> np.ndarray:
    n = G.number_of_nodes()
    colors = np.floor(np.log2(np.array([dg for _, dg in G.degree()]) + 1)).astype(int)
    lam = np.zeros((n, 3))
    for _ in range(NSA_WL_ROUNDS):
        new_colors = np.zeros(n, dtype=np.int64)
        for v in G.nodes():
            nb = tuple(sorted(colors[u] for u in G.neighbors(v)))
            new_colors[v] = hash((int(colors[v]), nb)) % NSA_WL_BUCKETS
        colors = new_colors
        for v in G.nodes():
            for idx, k in enumerate((1, 2, 3)):
                nodes_k = annuli[v][k]
                if len(nodes_k) == 0:
                    continue
                _, counts = np.unique(colors[nodes_k], return_counts=True)
                p = counts / len(nodes_k)
                H = -np.sum(p * np.log(p + 1e-15))
                lam[v, idx] += H / max(np.log(len(nodes_k) + 1), 1e-12)
    return lam / NSA_WL_ROUNDS


def _tss_descriptor(G: nx.Graph, annuli: Dict[int, Dict[int, np.ndarray]]) -> np.ndarray:
    n = G.number_of_nodes()
    deg = dict(G.degree())
    core = nx.core_number(G)

    def ego_dens(v: int) -> float:
        nb = list(G.neighbors(v))
        if len(nb) < 2:
            return 0.0
        sub = G.subgraph(nb)
        return 2.0 * sub.number_of_edges() / (len(nb) * (len(nb) - 1) + 1e-8)

    ego = {v: ego_dens(v) for v in G.nodes()}
    logd = {v: math.log1p(deg[v]) for v in G.nodes()}
    lam = _wl_role_entropy(G, annuli)

    feats = np.zeros((n, 3 + 9 * 3))
    for v in G.nodes():
        row = [logd[v], float(core[v]), ego[v]]
        b_prev, n_prev = 1, 1
        for idx, k in enumerate((1, 2, 3)):
            nodes_k = annuli[v][k]
            nk = len(nodes_k)
            b_k = b_prev + nk
            if nk == 0:
                row.extend([0.0] * 9)
            else:
                dks = np.array([logd[u] for u in nodes_k])
                row.extend([
                    math.log1p(nk), float(dks.mean()), float(dks.std()),
                    nk / max(n_prev, 1), nk / (b_k + 1),
                    float(np.mean([core[u] for u in nodes_k])),
                    float(np.mean([ego[u] for u in nodes_k])),
                    float(lam[v, idx]), math.log1p(b_k),
                ])
            b_prev, n_prev = b_k, max(nk, 1)
        feats[v] = np.asarray(row)
    return feats


def _lss_descriptor_for_fold(n: int, annuli: Dict[int, Dict[int, np.ndarray]],
                             y_fold: Dict[int, int], n_classes: int) -> np.ndarray:
    feats = np.zeros((n, 3 * (n_classes + 2)))
    for v in range(n):
        row: List[float] = []
        for k in (1, 2, 3):
            nodes_k = annuli[v][k]
            lab_k = [u for u in nodes_k if u in y_fold]
            m_k, n_k = len(lab_k), len(nodes_k)
            if m_k == 0:
                p = np.full(n_classes, 1.0 / n_classes)
            else:
                counts = np.zeros(n_classes)
                for u in lab_k:
                    counts[y_fold[u]] += 1
                p = counts / m_k
            row.extend(p.tolist() + [m_k / (n_k + 1), math.log1p(m_k)])
        feats[v] = np.asarray(row)
    return feats


def _normalized_macro_acc(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    chance = 1.0 / n_classes
    return (balanced_accuracy_score(y_true, y_pred) - chance) / max(1.0 - chance, 1e-12)


def _probe_score(X_desc: np.ndarray, y: np.ndarray, n_classes: int, seed: int) -> float:
    n_splits = min(5, int(np.min(np.bincount(y))))
    if n_splits < 2:
        return 0.0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, va in skf.split(X_desc, y):
        scaler = StandardScaler().fit(X_desc[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(X_desc[tr]), y[tr])
        accs.append(_normalized_macro_acc(y[va], clf.predict(scaler.transform(X_desc[va])), n_classes))
    return float(np.mean(accs))


@dataclass
class NSAScore:
    fss: float
    tss: float
    lss: float


def compute_nsa_scores(problem: Problem, seed: int = 0) -> NSAScore:
    G, X = problem.G, problem.X
    n_classes = problem.n_classes
    y_train = problem.y_dict(problem.train_mask)
    train_nodes = np.array(sorted(y_train.keys()))
    y_arr = np.array([y_train[v] for v in train_nodes])
    annuli = _bfs_annuli(G)

    fss_vals, tss_vals, lss_vals = [], [], []
    for s in NSA_SEEDS:
        seed_s = int((seed * 1000 + s) % (2**32 - 1))
        fss_vals.append(_probe_score(_fss_descriptor(X, annuli, seed_s)[train_nodes], y_arr, n_classes, seed_s))
        tss_vals.append(_probe_score(_tss_descriptor(G, annuli)[train_nodes], y_arr, n_classes, seed_s))

        n_splits = min(5, int(np.min(np.bincount(y_arr))))
        if n_splits < 2:
            lss_vals.append(0.0)
            continue
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_s)
        accs = []
        for inner_tr, inner_va in skf.split(train_nodes, y_arr):
            fold_labels = {int(train_nodes[i]): int(y_arr[i]) for i in inner_tr}
            lss_feats = _lss_descriptor_for_fold(G.number_of_nodes(), annuli, fold_labels, n_classes)
            scaler = StandardScaler().fit(lss_feats[train_nodes[inner_tr]])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(
                scaler.transform(lss_feats[train_nodes[inner_tr]]), y_arr[inner_tr]
            )
            accs.append(_normalized_macro_acc(
                y_arr[inner_va], clf.predict(scaler.transform(lss_feats[train_nodes[inner_va]])), n_classes
            ))
        lss_vals.append(float(np.mean(accs)))

    return NSAScore(fss=float(np.mean(fss_vals)), tss=float(np.mean(tss_vals)), lss=float(np.mean(lss_vals)))



# =============================================================================
# Architecture bank: specialist interfaces
# =============================================================================

class MLPModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index=None, edge_index_2=None):
        h = F.dropout(F.relu(self.l1(x)), p=self.dropout, training=self.training)
        return self.l2(h)


class GCNModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        _require_pyg()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_index_2=None):
        h = F.dropout(F.relu(self.c1(x, edge_index)), p=self.dropout, training=self.training)
        return self.c2(h, edge_index)


class SAGEModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        _require_pyg()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_index_2=None):
        h = F.dropout(F.relu(self.c1(x, edge_index)), p=self.dropout, training=self.training)
        return self.c2(h, edge_index)


class APPNPModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 dropout: float, hops: int, alpha: float):
        super().__init__()
        _require_pyg()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, out_dim)
        self.prop = APPNPProp(K=hops, alpha=alpha)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_index_2=None):
        h = F.dropout(F.relu(self.l1(x)), p=self.dropout, training=self.training)
        h = self.l2(h)
        return self.prop(h, edge_index)


class _MeanProp(MessagePassing if PYG_AVAILABLE else object):
    def __init__(self):
        super().__init__(aggr="mean")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros_like(x)
        return self.propagate(edge_index, x=x, size=(x.size(0), x.size(0)))

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


class H2GCNModel(nn.Module):
    """Compact faithful H2GCN-style interface.

    Core H2GCN ingredients used here:
      * ego information kept separate from neighborhood information;
      * 1-hop and exact-2-hop neighborhoods are aggregated separately;
      * representations from all depths are concatenated.

    We use mean aggregation for stable comparison at n=512.
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 dropout: float, layers: int = 2):
        super().__init__()
        _require_pyg()
        if layers not in (1, 2):
            raise ValueError("v5 H2GCN supports 1 or 2 aggregation layers.")
        self.input_proj = nn.Linear(in_dim, hidden)
        self.mean1 = _MeanProp()
        self.mean2 = _MeanProp()
        self.layers = layers
        self.dropout = dropout

        dims = [hidden]
        current = hidden
        for _ in range(layers):
            current *= 2
            dims.append(current)
        self.output = nn.Linear(sum(dims), out_dim)

    def forward(self, x, edge_index, edge_index_2):
        h = F.relu(self.input_proj(x))
        reps = [h]
        cur = h
        for _ in range(self.layers):
            h1 = self.mean1(cur, edge_index)
            h2 = self.mean2(cur, edge_index_2)
            cur = torch.cat([h1, h2], dim=-1)
            cur = F.relu(cur)
            cur = F.dropout(cur, p=self.dropout, training=self.training)
            reps.append(cur)
        z = torch.cat(reps, dim=-1)
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.output(z)


class GINModel(nn.Module):
    """Two-layer GIN with trainable epsilon; sum aggregation is degree-sensitive."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        _require_pyg()
        mlp1 = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        mlp2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.c1 = GINConv(mlp1, train_eps=True)
        self.c2 = GINConv(mlp2, train_eps=True)
        self.output = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_index_2=None):
        h = self.c1(x, edge_index)
        h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        h = self.c2(h, edge_index)
        h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        return self.output(h)


def make_model(name: str, in_dim: int, n_classes: int,
               cfg: ArchConfig) -> nn.Module:
    if name == "mlp":
        return MLPModel(in_dim, cfg.hidden, n_classes, cfg.dropout)
    if name == "gcn":
        return GCNModel(in_dim, cfg.hidden, n_classes, cfg.dropout)
    if name == "sage":
        return SAGEModel(in_dim, cfg.hidden, n_classes, cfg.dropout)
    if name == "appnp":
        return APPNPModel(
            in_dim, cfg.hidden, n_classes, cfg.dropout, cfg.hops, cfg.alpha
        )
    if name == "h2gcn":
        return H2GCNModel(
            in_dim, cfg.hidden, n_classes, cfg.dropout, layers=cfg.h2_layers
        )
    if name == "gin":
        return GINModel(in_dim, cfg.hidden, n_classes, cfg.dropout)
    raise ValueError(name)


def _metric_from_logits(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    yy = y[mask].detach().cpu().numpy()
    if len(yy) == 0:
        return float("nan")
    pred = logits[mask].detach().cpu().argmax(dim=1).numpy()
    return float(np.mean(pred == yy))


def _forward_arch(model: nn.Module, data: "Data", arch_name: str) -> torch.Tensor:
    if arch_name == "h2gcn":
        return model(data.x, data.edge_index, data.edge_index_2)
    return model(data.x, data.edge_index)


def train_one_seed(problem: Problem, arch_name: str, arch_cfg: ArchConfig, seed: int) -> float:
    _require_pyg()
    set_all_seeds(seed)
    data = problem_to_pyg(problem).to(DEVICE)
    model = make_model(arch_name, data.x.shape[1], problem.n_classes, arch_cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=arch_cfg.lr, weight_decay=arch_cfg.weight_decay)

    best_val = -np.inf
    best_state = None
    bad = 0
    for _ in range(arch_cfg.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = _forward_arch(model, data, arch_name)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val = _metric_from_logits(_forward_arch(model, data, arch_name), data.y, data.val_mask)
        if np.isfinite(val) and val > best_val + 1e-8:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= arch_cfg.patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test = _metric_from_logits(_forward_arch(model, data, arch_name), data.y, data.test_mask)
    return float(test)



# =============================================================================
# Symmetric meta-prediction and evaluation
# =============================================================================

class PerformancePredictor:
    def __init__(self, alpha: float = 1.0):
        self.pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "PerformancePredictor":
        self.pipe.fit(X, Y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.pipe.predict(np.asarray(x).reshape(1, -1))[0])


@dataclass
class SelectionMetrics:
    spearman: float
    top1: float
    top3: float
    regret: float


def selection_metrics(true: np.ndarray, pred: np.ndarray) -> SelectionMetrics:
    rho = spearmanr(true, pred).statistic
    if not np.isfinite(rho):
        rho = 0.0
    best = int(np.argmax(true))
    chosen = int(np.argmax(pred))
    top3 = set(np.argsort(pred)[-min(3, len(pred)):].tolist())
    return SelectionMetrics(
        spearman=float(rho),
        top1=float(best == chosen),
        top3=float(best in top3),
        regret=float(np.max(true) - true[chosen]),
    )


def metrics_mean_std(ms: Sequence[SelectionMetrics]) -> Dict[str, Dict[str, float]]:
    return {
        k: {
            "mean": float(np.nanmean([getattr(m, k) for m in ms])),
            "std": float(np.nanstd([getattr(m, k) for m in ms])),
        }
        for k in SelectionMetrics.__annotations__
    }


def paired_regret_tests(results: Mapping[str, Sequence[SelectionMetrics]], ours: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    a = np.array([x.regret for x in results[ours]])
    for b, vals in results.items():
        if b == ours or b in {"random", "oracle"}:
            continue
        bb = np.array([x.regret for x in vals])
        if len(a) != len(bb) or len(a) < 5:
            out[b] = float("nan")
            continue
        try:
            out[b] = float(wilcoxon(a, bb, alternative="less").pvalue)
        except ValueError:
            out[b] = 1.0
    return out


# =============================================================================
# Cache with problem-content fingerprinting
# =============================================================================

class PerformanceCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db: Dict[str, float] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, key: str) -> float | None:
        return self.db.get(key)

    def put(self, key: str, value: float) -> None:
        self.db[key] = float(value)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.db, indent=2, sort_keys=True))
        tmp.replace(self.path)


def _problem_fingerprint(problem: Problem) -> str:
    h = hashlib.sha1()
    h.update(GENERATOR_VERSION.encode())
    edges = np.asarray(sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in problem.G.edges()),
                       dtype=np.int64)
    h.update(edges.tobytes())
    h.update(np.ascontiguousarray(problem.X, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(problem.y, dtype=np.int64).tobytes())
    h.update(np.packbits(problem.train_mask).tobytes())
    h.update(np.packbits(problem.val_mask).tobytes())
    h.update(np.packbits(problem.test_mask).tobytes())
    return h.hexdigest()


def _perf_cache_key(problem: Problem, arch: str, seed: int, cfg: ArchConfig) -> str:
    payload = json.dumps({
        "fingerprint": _problem_fingerprint(problem),
        "model_bank_version": MODEL_BANK_VERSION,
        "arch": arch,
        "seed": seed,
        "cfg": asdict(cfg),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def performance_vector(problem: Problem, cfg: ExperimentConfig,
                       cache: PerformanceCache) -> Tuple[np.ndarray, np.ndarray]:
    means, stds = [], []
    for arch in active_architectures(cfg):
        vals = []
        ac = cfg.arch[arch]
        for seed in cfg.model_seeds:
            key = _perf_cache_key(problem, arch, seed, ac)
            val = cache.get(key)
            if val is None:
                val = train_one_seed(problem, arch, ac, seed)
                cache.put(key, val)
            vals.append(val)
        means.append(np.nanmean(vals))
        stds.append(np.nanstd(vals))
    return np.asarray(means), np.asarray(stds)


# =============================================================================
# Corpus construction
# =============================================================================

@dataclass
class CorpusItem:
    problem: Problem
    nsa: NSAScore
    homophily: np.ndarray
    conventional: np.ndarray
    perf: np.ndarray
    perf_std: np.ndarray
    mechanism: str


def _problem_from_generated(name: str, G: nx.Graph, X: np.ndarray,
                            y_dict: Dict[int, int], seed: int) -> Problem:
    y = np.array([y_dict[i] for i in range(len(y_dict))], dtype=int)
    tr, va, te = stratified_masks(y, seed)
    return Problem(
        name=name,
        G=G,
        X=np.asarray(X, dtype=np.float32),
        y=y,
        train_mask=tr,
        val_mask=va,
        test_mask=te,
        metric="accuracy",
    )


def build_synthetic_corpus(family: str, per_mechanism: int,
                           cfg: ExperimentConfig, cache: PerformanceCache,
                           seed: int, train_models: bool = True,
                           compute_descriptors: bool = True,
                           model_seeds_override: Tuple[int, ...] | None = None
                           ) -> List[CorpusItem]:
    rng = np.random.default_rng(seed)
    items: List[CorpusItem] = []

    if model_seeds_override is None:
        model_cfg = cfg
    else:
        model_cfg = copy.deepcopy(cfg)
        model_cfg.model_seeds = tuple(model_seeds_override)

    for mech in NATIVE_MECHANISMS:
        for j in range(per_mechanism):
            s = int(rng.integers(1, 2**31 - 1))
            if family == "train":
                G, X, y = generate_train_problem(
                    mech, cfg.synthetic_n, s, cfg.n_classes
                )
            elif family == "test":
                G, X, y = generate_test_problem(
                    mech, cfg.synthetic_n, s, cfg.n_classes
                )
            else:
                raise ValueError(family)

            p = _problem_from_generated(
                f"{GENERATOR_VERSION}_{family}_{mech}_{j}_{s}",
                G, X, y, seed=s
            )

            if compute_descriptors:
                y_train = p.y_dict(p.train_mask)
                nsa = compute_nsa_scores(p, seed=s)
                hom = homophily_statistics(p, y_train)
                conv = conventional_statistics(p, y_train)
            else:
                # Architecture-only pilot: do not compute NSA or any predictor.
                nsa = NSAScore(float("nan"), float("nan"), float("nan"))
                hom = np.full(len(HOMOPHILY_STAT_NAMES), np.nan, dtype=float)
                conv = np.full(len(CONVENTIONAL_STAT_NAMES), np.nan, dtype=float)

            if train_models:
                perf, perf_std = performance_vector(p, model_cfg, cache)
            else:
                k = len(active_architectures(cfg))
                perf = np.full(k, np.nan)
                perf_std = np.full(k, np.nan)

            items.append(CorpusItem(
                problem=p,
                nsa=nsa,
                homophily=hom,
                conventional=conv,
                perf=perf,
                perf_std=perf_std,
                mechanism=mech,
            ))
    return items


def nsa_matrix(items: Sequence[CorpusItem]) -> np.ndarray:
    return np.asarray([[x.nsa.fss, x.nsa.tss, x.nsa.lss] for x in items], dtype=float)


def homophily_matrix(items: Sequence[CorpusItem]) -> np.ndarray:
    """Homophily-4: edge, node, adjusted homophily, label informativeness."""
    return np.vstack([x.homophily for x in items])


def node_homophily_matrix(items: Sequence[CorpusItem]) -> np.ndarray:
    """NodeHomophily-1 as a 2D design matrix."""
    # HOMOPHILY_STAT_NAMES = (edge, node, adjusted, LI)
    return np.asarray([[float(x.homophily[1])] for x in items], dtype=float)


def conventional_matrix(items: Sequence[CorpusItem]) -> np.ndarray:
    """Full 20-D diagnostic vector retained only for auditing/debugging."""
    return np.vstack([x.conventional for x in items])


def meta5_matrix(items: Sequence[CorpusItem]) -> np.ndarray:
    """Fixed Meta-5 subset used in the reported v5 experiment."""
    return np.vstack([x.conventional[list(_META5_INDICES)] for x in items])


def performance_matrix(items: Sequence[CorpusItem]) -> np.ndarray:
    return np.vstack([x.perf for x in items])


# =============================================================================
# Dataset-level diagnostics
# =============================================================================

def _nuisance_summary_problem(problem: Problem) -> Dict[str, float]:
    G = problem.G
    y_all = problem.y_dict()
    deg = np.asarray([d for _, d in G.degree()], dtype=float)
    edge_cos, _, _ = _edge_feature_stats(problem)
    return {
        "avg_degree": float(deg.mean()) if len(deg) else 0.0,
        "std_degree": float(deg.std()) if len(deg) else 0.0,
        "edge_homophily_full": edge_homophily(G, y_all),
        "avg_clustering": float(nx.average_clustering(G)) if G.number_of_nodes() else 0.0,
        "spectral_gap": _spectral_gap(G),
        "edge_feature_cos_mean": edge_cos,
    }


def mechanism_shift_summary(train_items: Sequence[CorpusItem],
                            test_items: Sequence[CorpusItem]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for mech in NATIVE_MECHANISMS:
        tr = [_nuisance_summary_problem(x.problem)
              for x in train_items if x.mechanism == mech]
        te = [_nuisance_summary_problem(x.problem)
              for x in test_items if x.mechanism == mech]
        keys = list(tr[0]) if tr else []
        out[mech] = {
            "train_mean": {k: float(np.mean([z[k] for z in tr])) for k in keys},
            "train_std": {k: float(np.std([z[k] for z in tr])) for k in keys},
            "test_mean": {k: float(np.mean([z[k] for z in te])) for k in keys},
            "test_std": {k: float(np.std([z[k] for z in te])) for k in keys},
        }
    return out


def mechanism_nsa_table(items: Sequence[CorpusItem]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for mech in NATIVE_MECHANISMS:
        ss = [x.nsa for x in items if x.mechanism == mech]
        out[mech] = {
            "FSS": float(np.nanmean([s.fss for s in ss])),
            "TSS": float(np.nanmean([s.tss for s in ss])),
            "LSS": float(np.nanmean([s.lss for s in ss])),
        }
    return out


def meta5_shift_diagnostic(train_items: Sequence[CorpusItem],
                           test_items: Sequence[CorpusItem]) -> Dict[str, Any]:
    Xtr = meta5_matrix(train_items)
    Xte = meta5_matrix(test_items)
    scaler = StandardScaler().fit(Xtr)
    z = scaler.transform(Xte)
    return {
        name: {
            "train_mean": float(np.mean(Xtr[:, j])),
            "train_std": float(np.std(Xtr[:, j])),
            "test_mean": float(np.mean(Xte[:, j])),
            "test_std": float(np.std(Xte[:, j])),
            "test_mean_abs_z": float(np.mean(np.abs(z[:, j]))),
            "test_max_abs_z": float(np.max(np.abs(z[:, j]))),
        }
        for j, name in enumerate(META5_STAT_NAMES)
    }


def architecture_headroom_report(train_items: Sequence[CorpusItem],
                                 test_items: Sequence[CorpusItem],
                                 arch_names: Sequence[str],
                                 cfg: ExperimentConfig) -> Dict[str, Any]:
    """Architecture-only pilot criteria.

    IMPORTANT: this function does not use NSA or homophily predictors.
    The fixed baseline ranking is learned from pilot-train architecture
    performance and evaluated on pilot-test problems.
    """
    Ytr = performance_matrix(train_items)
    Yte = performance_matrix(test_items)

    mean_train = np.nanmean(Ytr, axis=0)
    fixed_order = np.argsort(mean_train)[::-1]
    fixed_best = int(fixed_order[0])
    fixed_top3 = set(fixed_order[:min(3, len(fixed_order))].tolist())

    winners = np.nanargmax(Yte, axis=1)
    counts = {
        a: int(np.sum(winners == j))
        for j, a in enumerate(arch_names)
    }
    top_share = max(counts.values()) / max(len(Yte), 1)
    distinct = sum(v > 0 for v in counts.values())

    oracle = np.nanmax(Yte, axis=1)
    fixed_acc = Yte[:, fixed_best]
    constant_regret = oracle - fixed_acc

    spread = np.nanmax(Yte, axis=1) - np.nanmin(Yte, axis=1)
    sorted_perf = np.sort(Yte, axis=1)
    best_second_gap = sorted_perf[:, -1] - sorted_perf[:, -2]

    top3_hit = float(np.mean([int(w) in fixed_top3 for w in winners]))

    checks = {
        "winner_share": bool(top_share <= cfg.max_winner_share),
        "distinct_winners": bool(distinct >= cfg.min_distinct_winners),
        "constant_regret": bool(float(np.mean(constant_regret)) >= cfg.min_constant_regret),
        "median_spread": bool(float(np.median(spread)) >= cfg.min_median_spread),
        "fraction_spread_gt_005": bool(float(np.mean(spread > 0.05)) >= cfg.min_fraction_spread_gt_005),
        "fixed_top3_hit": bool(top3_hit <= cfg.max_fixed_top3_hit),
        "median_best_second_gap": bool(float(np.median(best_second_gap)) >= cfg.min_median_best_second_gap),
    }

    return {
        "pilot_only": True,
        "uses_nsa_for_selection": False,
        "train_mean_architecture_performance": {
            a: float(mean_train[j]) for j, a in enumerate(arch_names)
        },
        "fixed_best_architecture_from_train": arch_names[fixed_best],
        "fixed_top3_from_train": [arch_names[j] for j in fixed_order[:3]],
        "test_winner_counts": counts,
        "top_architecture_share": float(top_share),
        "distinct_winners": int(distinct),
        "oracle_accuracy": float(np.mean(oracle)),
        "fixed_best_accuracy": float(np.mean(fixed_acc)),
        "mean_constant_regret": float(np.mean(constant_regret)),
        "median_within_graph_spread": float(np.median(spread)),
        "mean_within_graph_spread": float(np.mean(spread)),
        "fraction_spread_gt_005": float(np.mean(spread > 0.05)),
        "fixed_top3_hit": top3_hit,
        "median_best_second_gap": float(np.median(best_second_gap)),
        "criteria": {
            "max_winner_share": cfg.max_winner_share,
            "min_distinct_winners": cfg.min_distinct_winners,
            "min_constant_regret": cfg.min_constant_regret,
            "min_median_spread": cfg.min_median_spread,
            "min_fraction_spread_gt_005": cfg.min_fraction_spread_gt_005,
            "max_fixed_top3_hit": cfg.max_fixed_top3_hit,
            "min_median_best_second_gap": cfg.min_median_best_second_gap,
        },
        "checks": checks,
        "passes_all": bool(all(checks.values())),
    }


# =============================================================================
# Prediction and budgeted-search evaluation
# =============================================================================

def _order_to_curve(true_perf: np.ndarray, order_idx: Sequence[int]) -> np.ndarray:
    curve = np.zeros(len(order_idx))
    best = -np.inf
    for k, idx in enumerate(order_idx):
        best = max(best, true_perf[idx])
        curve[k] = best
    return curve



def evaluate_predictors(train_items: Sequence[CorpusItem], test_items: Sequence[CorpusItem],
                        arch_names: Sequence[str], cfg: ExperimentConfig,
                        seed: int = 0) -> Dict[str, Any]:
    """Fit identical Ridge meta-predictors for every representation.

    The target is the full architecture-performance vector. Prediction methods
    differ ONLY in their input representation.
    """
    Y_train = performance_matrix(train_items)
    global_mean = np.nanmean(Y_train, axis=0)

    nsa_tr = nsa_matrix(train_items)
    hom4_tr = homophily_matrix(train_items)
    meta5_tr = meta5_matrix(train_items)

    Xtr = {
        "node_homophily_ridge": node_homophily_matrix(train_items),
        "homophily4_ridge": hom4_tr,
        "nsa_ridge": nsa_tr,
        "homophily4_meta5_ridge": np.hstack([hom4_tr, meta5_tr]),
        "nsa_meta5_ridge": np.hstack([nsa_tr, meta5_tr]),
    }

    predictors = {
        name: PerformancePredictor(cfg.meta_ridge_alpha).fit(X, Y_train)
        for name, X in Xtr.items()
    }

    rng = np.random.default_rng(seed)
    metric_rows: Dict[str, List[SelectionMetrics]] = {name: [] for name in predictors}
    metric_rows.update({"global_mean": [], "random": []})

    curves: Dict[str, List[np.ndarray]] = {name: [] for name in metric_rows}
    curves["oracle"] = []
    per_problem: List[Dict[str, Any]] = []

    for item in test_items:
        true = item.perf
        nsa = np.array([item.nsa.fss, item.nsa.tss, item.nsa.lss], dtype=float)
        hom4 = np.asarray(item.homophily, dtype=float)
        meta5 = np.asarray(item.conventional[list(_META5_INDICES)], dtype=float)

        xmap = {
            "node_homophily_ridge": np.array([hom4[1]], dtype=float),
            "homophily4_ridge": hom4,
            "nsa_ridge": nsa,
            "homophily4_meta5_ridge": np.concatenate([hom4, meta5]),
            "nsa_meta5_ridge": np.concatenate([nsa, meta5]),
        }

        predictions = {
            name: predictors[name].predict(xmap[name])
            for name in predictors
        }
        predictions["global_mean"] = global_mean.copy()

        row: Dict[str, Any] = {
            "problem": item.problem.name,
            "mechanism": item.mechanism,
            "true_performance": true.tolist(),
            "true_best": arch_names[int(np.nanargmax(true))],
            "performance_spread": float(np.nanmax(true) - np.nanmin(true)),
            "nsa": asdict(item.nsa),
            "node_homophily": float(hom4[1]),
            "homophily4": hom4.tolist(),
            "meta5": meta5.tolist(),
        }

        for name, pred in predictions.items():
            metric_rows[name].append(selection_metrics(true, pred))
            order = np.argsort(pred)[::-1]
            curves[name].append(_order_to_curve(true, order))
            row[name] = {
                "predicted_performance": pred.tolist(),
                "chosen": arch_names[int(np.argmax(pred))],
            }

        # Random ranking baseline, averaged over many rankings for this graph.
        rms: List[SelectionMetrics] = []
        rcs: List[np.ndarray] = []
        for _ in range(cfg.random_baseline_draws):
            order = rng.permutation(len(arch_names))
            pred = np.empty(len(arch_names), dtype=float)
            pred[order] = np.arange(len(arch_names), 0, -1)
            rms.append(selection_metrics(true, pred))
            rcs.append(_order_to_curve(true, order))

        metric_rows["random"].append(SelectionMetrics(
            spearman=float(np.mean([m.spearman for m in rms])),
            top1=float(np.mean([m.top1 for m in rms])),
            top3=float(np.mean([m.top3 for m in rms])),
            regret=float(np.mean([m.regret for m in rms])),
        ))
        curves["random"].append(np.mean(rcs, axis=0))
        curves["oracle"].append(np.full(len(arch_names), np.max(true)))
        per_problem.append(row)

    aggregate = {
        name: metrics_mean_std(vals)
        for name, vals in metric_rows.items()
    }

    # Three pre-specified comparisons of primary interest.
    aggregate["regret_tests"] = {
        "nsa_vs_node_homophily":
            paired_regret_tests(
                {
                    "nsa_ridge": metric_rows["nsa_ridge"],
                    "node_homophily_ridge": metric_rows["node_homophily_ridge"],
                },
                "nsa_ridge",
            ).get("node_homophily_ridge", float("nan")),

        "nsa_vs_homophily4":
            paired_regret_tests(
                {
                    "nsa_ridge": metric_rows["nsa_ridge"],
                    "homophily4_ridge": metric_rows["homophily4_ridge"],
                },
                "nsa_ridge",
            ).get("homophily4_ridge", float("nan")),

        "nsa_meta5_vs_homophily4_meta5":
            paired_regret_tests(
                {
                    "nsa_meta5_ridge": metric_rows["nsa_meta5_ridge"],
                    "homophily4_meta5_ridge": metric_rows["homophily4_meta5_ridge"],
                },
                "nsa_meta5_ridge",
            ).get("homophily4_meta5_ridge", float("nan")),
    }

    budget: Dict[str, Any] = {}
    for name, cs in curves.items():
        arr = np.vstack(cs)
        budget[name] = {
            "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(),
        }

    oracle_mean = np.asarray(budget["oracle"]["mean"])
    for name in metric_rows:
        m = np.asarray(budget[name]["mean"])
        budget[name]["pct_of_oracle"] = (
            m / np.where(oracle_mean == 0, 1.0, oracle_mean)
        ).tolist()

    # Per-regime reporting.
    per_mechanism: Dict[str, Any] = {}
    for mech in NATIVE_MECHANISMS:
        ids = [i for i, x in enumerate(test_items) if x.mechanism == mech]
        per_mechanism[mech] = {
            name: metrics_mean_std([metric_rows[name][i] for i in ids])
            for name in metric_rows
        }

    return {
        "aggregate": aggregate,
        "budgeted_search": budget,
        "per_mechanism": per_mechanism,
        "per_problem": per_problem,
    }


def _architecture_diagnostics(items: Sequence[CorpusItem],
                              arch_names: Sequence[str]) -> Dict[str, Any]:
    Y = performance_matrix(items)
    winners = np.nanargmax(Y, axis=1)
    counts = {a: int(np.sum(winners == j)) for j, a in enumerate(arch_names)}
    per_mechanism_winner: Dict[str, Dict[str, int]] = {
        m: {a: 0 for a in arch_names} for m in NATIVE_MECHANISMS
    }
    per_mechanism_perf: Dict[str, Any] = {}

    for item, winner_idx in zip(items, winners):
        per_mechanism_winner[item.mechanism][arch_names[int(winner_idx)]] += 1

    for mech in NATIVE_MECHANISMS:
        arr = np.vstack([x.perf for x in items if x.mechanism == mech])
        per_mechanism_perf[mech] = {
            "mean_accuracy": {
                a: float(np.mean(arr[:, j])) for j, a in enumerate(arch_names)
            },
            "mean_spread": float(np.mean(np.max(arr, axis=1) - np.min(arr, axis=1))),
            "median_spread": float(np.median(np.max(arr, axis=1) - np.min(arr, axis=1))),
            "mean_best_second_gap": float(np.mean(
                np.sort(arr, axis=1)[:, -1] - np.sort(arr, axis=1)[:, -2]
            )),
        }

    return {
        "win_counts": counts,
        "per_mechanism_winner": per_mechanism_winner,
        "per_mechanism_architecture_performance": per_mechanism_perf,
    }


def run_nsa_experiment(cfg: ExperimentConfig, out_dir: Path) -> Dict[str, Any]:
    """FINAL descriptor comparison. Use only after generator/pilot freeze."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = PerformanceCache(
        Path(cfg.cache_dir) / f"architecture_perf_{GENERATOR_VERSION}_{MODEL_BANK_VERSION}.json"
    )
    arch_names = active_architectures(cfg)

    # Fresh final seeds, intentionally distinct from pilot/audit seeds.
    print("Building FINAL G_train corpus ...")
    train_items = build_synthetic_corpus(
        "train", cfg.train_problems_per_mechanism, cfg, cache,
        seed=1010, train_models=True, compute_descriptors=True
    )
    print("Building FINAL G_test corpus ...")
    test_items = build_synthetic_corpus(
        "test", cfg.test_problems_per_mechanism, cfg, cache,
        seed=2020, train_models=True, compute_descriptors=True
    )

    results = evaluate_predictors(train_items, test_items, arch_names, cfg, seed=917)
    arch_diag = _architecture_diagnostics(test_items, arch_names)

    payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "model_bank_version": MODEL_BANK_VERSION,
        "stage": "final",
        "arch_names": list(arch_names),
        "mechanisms": list(NATIVE_MECHANISMS),
        "n_train": len(train_items),
        "n_test": len(test_items),
        "representations": {
            "node_homophily_names": ["node_homophily"],
            "homophily4_names": list(HOMOPHILY_STAT_NAMES),
            "meta5_names": list(META5_STAT_NAMES),
            "nsa_names": ["FSS", "TSS", "LSS"],
            "reported_rows": [
                "node_homophily_ridge",
                "homophily4_ridge",
                "nsa_ridge",
                "homophily4_meta5_ridge",
                "nsa_meta5_ridge",
            ],
        },
        "generator_shift": mechanism_shift_summary(train_items, test_items),
        "meta5_shift": meta5_shift_diagnostic(train_items, test_items),
        "nsa_train_by_mechanism": mechanism_nsa_table(train_items),
        "nsa_test_by_mechanism": mechanism_nsa_table(test_items),
        "win_counts_on_test": arch_diag["win_counts"],
        "per_mechanism_winner_on_test": arch_diag["per_mechanism_winner"],
        "per_mechanism_architecture_performance":
            arch_diag["per_mechanism_architecture_performance"],
        **results,
    }

    out_file = out_dir / "nsa_interface_benchmark_v5_final.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_file}")
    return payload


def run_pilot(cfg: ExperimentConfig, out_dir: Path) -> Dict[str, Any]:
    """Architecture-only pilot. NSA/descriptors are deliberately not computed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = PerformanceCache(
        Path(cfg.cache_dir) / f"pilot_architecture_perf_{GENERATOR_VERSION}_{MODEL_BANK_VERSION}.json"
    )
    arch_names = active_architectures(cfg)

    print("Building PILOT train corpus (architecture-only) ...")
    train_items = build_synthetic_corpus(
        "train", cfg.pilot_train_per_mechanism, cfg, cache,
        seed=310, train_models=True, compute_descriptors=False,
        model_seeds_override=cfg.pilot_model_seeds,
    )
    print("Building PILOT test corpus (architecture-only) ...")
    test_items = build_synthetic_corpus(
        "test", cfg.pilot_test_per_mechanism, cfg, cache,
        seed=320, train_models=True, compute_descriptors=False,
        model_seeds_override=cfg.pilot_model_seeds,
    )

    report = architecture_headroom_report(train_items, test_items, arch_names, cfg)
    arch_diag = _architecture_diagnostics(test_items, arch_names)

    payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "model_bank_version": MODEL_BANK_VERSION,
        "stage": "pilot_architecture_only",
        "explicit_rule": (
            "Do not inspect NSA-vs-homophily prediction results during pilot. "
            "Parameter revisions may use only architecture headroom and nuisance "
            "statistics. After acceptance, freeze generators and use fresh final seeds."
        ),
        "arch_names": list(arch_names),
        "mechanisms": list(NATIVE_MECHANISMS),
        "n_train": len(train_items),
        "n_test": len(test_items),
        "headroom": report,
        "generator_shift": mechanism_shift_summary(train_items, test_items),
        "win_counts_on_test": arch_diag["win_counts"],
        "per_mechanism_winner_on_test": arch_diag["per_mechanism_winner"],
        "per_mechanism_architecture_performance":
            arch_diag["per_mechanism_architecture_performance"],
    }

    out_file = out_dir / "nsa_v5_pilot_headroom.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_file}")
    return payload


def run_generator_audit(cfg: ExperimentConfig, out_dir: Path,
                        per_mechanism: int = 3) -> Dict[str, Any]:
    """Cheap generator + NSA implementation sanity audit; no GNN training."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = PerformanceCache(Path(cfg.cache_dir) / "unused_audit_cache.json")

    train_items = build_synthetic_corpus(
        "train", per_mechanism, cfg, cache, seed=110,
        train_models=False, compute_descriptors=True
    )
    test_items = build_synthetic_corpus(
        "test", per_mechanism, cfg, cache, seed=120,
        train_models=False, compute_descriptors=True
    )

    payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "stage": "audit_sanity_only",
        "explicit_rule": (
            "NSA values here are for implementation/sanity checking only. "
            "Do not choose generator parameters because NSA compares favorably "
            "or unfavorably with homophily."
        ),
        "generator_shift": mechanism_shift_summary(train_items, test_items),
        "meta5_shift": meta5_shift_diagnostic(train_items, test_items),
        "nsa_train_by_mechanism": mechanism_nsa_table(train_items),
        "nsa_test_by_mechanism": mechanism_nsa_table(test_items),
    }

    out_file = out_dir / "generator_audit_v5.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_file}")
    return payload


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./results/nsa_v5")
    ap.add_argument("--cache-dir", type=str, default="./cache_v5")
    ap.add_argument(
        "--exclude", type=str, default="",
        help="debug only: comma-separated primary architectures to drop"
    )
    ap.add_argument(
        "--quick", action="store_true",
        help="pipeline smoke test only; never report these numbers"
    )
    ap.add_argument(
        "--audit-only", action="store_true",
        help="generator + NSA sanity audit; no architecture training"
    )
    ap.add_argument(
        "--pilot", action="store_true",
        help="architecture-only pilot; DOES NOT compute NSA predictors"
    )
    ap.add_argument("--audit-per-mechanism", type=int, default=3)
    ap.add_argument("--pilot-train-per-mechanism", type=int, default=None)
    ap.add_argument("--pilot-test-per-mechanism", type=int, default=None)
    args = ap.parse_args()

    if args.audit_only and args.pilot:
        raise ValueError("Choose at most one of --audit-only and --pilot.")

    cfg = ExperimentConfig.quick() if args.quick else ExperimentConfig()
    cfg.cache_dir = args.cache_dir
    cfg.exclude_architectures = tuple(
        a.strip() for a in args.exclude.split(",") if a.strip()
    )
    if args.pilot_train_per_mechanism is not None:
        cfg.pilot_train_per_mechanism = args.pilot_train_per_mechanism
    if args.pilot_test_per_mechanism is not None:
        cfg.pilot_test_per_mechanism = args.pilot_test_per_mechanism

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print(
        f"device={DEVICE}; architectures={','.join(active_architectures(cfg))}; "
        f"quick={args.quick}; pilot={args.pilot}; audit={args.audit_only}"
    )
    if args.quick:
        warnings.warn("QUICK MODE is only a pipeline smoke test. Do not report these numbers.")

    if args.audit_only:
        payload = run_generator_audit(
            cfg, out, per_mechanism=args.audit_per_mechanism
        )
    elif args.pilot:
        payload = run_pilot(cfg, out)
    else:
        payload = run_nsa_experiment(cfg, out)

    # Compact terminal summary.
    skip = {"per_problem", "budgeted_search", "per_mechanism"}
    print(json.dumps({k: v for k, v in payload.items() if k not in skip}, indent=2))
    print("\nDone.")




if __name__ == "__main__":
    main()
