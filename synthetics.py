"""
Synthetic Dataset Generators
==============================
Eight controlled regime generators covering all 2x2x2 combinations of
F-signal (high/low), T-signal (high/low), and L-signal (high/low).

Design matrix:
  Name   FSS  TSS  LSS  Construction
  ----   ---  ---  ---  ------------
  F       +    -    -   ER graph + class-conditional features + shuffled labels
  T       -    +    -   Disassortative DC-SBM (distinct degrees) + noise features
  L       -    -    +   Homophilic SBM (matched degrees) + noise features
  FT      +    +    -   Disassortative DC-SBM (distinct degrees) + class-conditional features
  FL      +    -    +   Homophilic SBM (matched degrees) + class-conditional features
  TL      -    +    +   Assortative DC-SBM (distinct degrees) + noise features
  FTL     +    +    +   Assortative DC-SBM (distinct degrees) + class-conditional features
  C       -    -    -   Node Twins: coupling-only signal

Key design principles:
  - T-signal ON  requires distinct degree profiles across blocks (DC-SBM with
    different expected degrees per block). WL color entropy differentiates blocks.
  - T-signal OFF requires matched degree profiles across blocks (uniform theta).
  - L-signal ON  requires assortative edges (p_intra >> p_inter).
  - L-signal OFF requires disassortative or near-random edges (p_inter >= p_intra).
  - F-signal ON  requires class-conditional Gaussian features (orthogonalized means).
  - F-signal OFF requires pure noise features.
  - The two mechanisms (feature and structural) are made independent by construction:
    feature assignment is post-hoc and does not affect graph generation.

All generators return (edge_index, X, y) as numpy arrays:
  edge_index: (2, E) int64
  X:          (N, d) float32
  y:          (N,)   int
"""

import numpy as np
from typing import Tuple, List


# =============================================================
# SHARED HELPERS
# =============================================================

def _balanced_labels(num_nodes: int,
                     num_classes: int,
                     rng: np.random.RandomState) -> np.ndarray:
    """Balanced label assignment, randomly permuted."""
    y = np.repeat(np.arange(num_classes),
                  num_nodes // num_classes + 1)[:num_nodes]
    rng.shuffle(y)
    return y.astype(int)


def _contiguous_labels(num_nodes: int, num_classes: int) -> np.ndarray:
    """Contiguous block label assignment (community-aligned)."""
    block_size = num_nodes // num_classes
    return np.minimum(np.arange(num_nodes) // block_size,
                      num_classes - 1).astype(int)


def _noise_features(num_nodes: int, d: int,
                    rng: np.random.RandomState) -> np.ndarray:
    """Pure isotropic Gaussian noise features."""
    return rng.randn(num_nodes, d).astype(np.float32)


def _class_conditional_features(num_nodes: int, d: int,
                                  y: np.ndarray,
                                  num_classes: int,
                                  feature_sep: float,
                                  rng: np.random.RandomState) -> np.ndarray:
    """
    Class-conditional Gaussian features with orthogonalized means.
    Mean separation is feature_sep * sqrt(d) in each direction.
    """
    class_means = rng.randn(num_classes, d)
    if num_classes <= d:
        Q, _ = np.linalg.qr(class_means.T)
        class_means = Q.T * feature_sep * np.sqrt(d)
    else:
        norms = np.linalg.norm(class_means, axis=1, keepdims=True) + 1e-8
        class_means = class_means / norms * feature_sep
    # Vectorized: index means by label, add noise in one shot
    X = class_means[y] + rng.randn(num_nodes, d)
    return X.astype(np.float32)


def _dc_sbm(num_nodes: int,
            num_classes: int,
            y: np.ndarray,
            p_intra: float,
            p_inter: float,
            degree_weights: np.ndarray,
            rng: np.random.RandomState) -> np.ndarray:
    """
    Degree-corrected SBM -- vectorized, no Python loops over node pairs.

    degree_weights: per-class expected-degree multipliers, shape (num_classes,).
      - Uniform weights (all equal): matched degrees across classes -> T-signal OFF.
      - Distinct weights: different expected degrees per class -> T-signal ON.

    p_intra > p_inter: assortative    -> L-signal ON.
    p_intra < p_inter: disassortative -> L-signal OFF (h ~= 1/C).
    p_intra ~= p_inter: near-random   -> L-signal OFF.

    At N=1500 the upper triangle has ~1.1M pairs; numpy handles this in
    milliseconds, replacing the previous pure-Python O(N^2) loop.
    """
    theta = degree_weights[y]      # (N,) per-node weight
    mean_w = degree_weights.mean()

    # All upper-triangle pairs
    u_idx, v_idx = np.triu_indices(num_nodes, k=1)

    # Base probability: p_intra for same-class pairs, p_inter otherwise
    same = y[u_idx] == y[v_idx]
    p_base = np.where(same, p_intra, p_inter)

    # Degree-corrected probability, clipped to [0, 1]
    p_uv = np.clip(
        p_base * theta[u_idx] * theta[v_idx] / (mean_w ** 2),
        0.0, 1.0
    )

    # Sample edges
    keep = rng.random_sample(len(u_idx)) < p_uv
    u_keep, v_keep = u_idx[keep], v_idx[keep]

    if len(u_keep) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    # Undirected: both directions
    return np.stack([
        np.concatenate([u_keep, v_keep]),
        np.concatenate([v_keep, u_keep])
    ]).astype(np.int64)


def _er_graph(num_nodes: int,
              avg_degree: float,
              rng: np.random.RandomState) -> np.ndarray:
    """Erdos-Renyi random graph with given average degree."""
    n_edges = int(num_nodes * avg_degree / 2)
    src = rng.randint(0, num_nodes, n_edges)
    dst = rng.randint(0, num_nodes, n_edges)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    return np.stack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src])
    ]).astype(np.int64)


# Degree weight presets
_MATCHED_WEIGHTS    = np.array([1.0, 1.0, 1.0])   # T-signal OFF
_DISTINCT_WEIGHTS   = np.array([1.0, 2.0, 3.5])   # T-signal ON


# =============================================================
# 1. F-DOMINANT  (F+, T-, L-)
# =============================================================

def generate_f_dominant(num_nodes: int = 1500,
                        num_classes: int = 3,
                        feature_sep: float = 0.5,
                        d: int = 16,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    F+, T-, L-: labels predictable from node features alone.

    Graph: Erdos-Renyi (no structural label signal).
    Features: class-conditional Gaussians (orthogonalized means).
    Labels: from feature clusters, randomly shuffled so graph position != label.

    ER graph ensures h ≈ 1/C (L-) and uniform degree distribution (T-).
    Shuffling labels breaks any residual spatial correlation.

    Expected: high FSS, low TSS, low LSS. Best model: MLP.
    """
    rng = np.random.RandomState(seed)
    y = _balanced_labels(num_nodes, num_classes, rng)
    edge_index = _er_graph(num_nodes, avg_degree=4.0, rng=rng)
    X = _class_conditional_features(num_nodes, d, y, num_classes, feature_sep, rng)
    return edge_index, X, y


# =============================================================
# 2. T-DOMINANT  (F-, T+, L-)
# =============================================================

def generate_t_dominant(num_nodes: int = 1500,
                        num_classes: int = 3,
                        p_intra: float = 0.002,
                        p_inter: float = 0.010,
                        d: int = 16,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    F-, T+, L-: labels predictable from unlabeled topology alone.

    Graph: disassortative DC-SBM with DISTINCT expected degrees per block.
      - p_inter >> p_intra -> h ≈ 1/C -> L-signal OFF.
      - Distinct degree weights [1, 2, 3.5] -> different WL colors, degree
        profiles, expansion ratios per block -> T-signal ON.
    Features: pure noise -> F-signal OFF.
    Labels: aligned with block membership.

    Expected: low FSS, high TSS, low LSS. Best model: GPS > H2GCN >> LP.
    """
    rng = np.random.RandomState(seed)
    y = _contiguous_labels(num_nodes, num_classes)
    edge_index = _dc_sbm(num_nodes, num_classes, y,
                         p_intra, p_inter, _DISTINCT_WEIGHTS, rng)
    X = _noise_features(num_nodes, d, rng)
    return edge_index, X, y



# =============================================================
# 3. L-DOMINANT  (F-, T-, L+)
# =============================================================

def generate_l_dominant(num_nodes: int = 1500,
                        num_classes: int = 3,
                        p_intra: float = 0.012,
                        p_inter: float = 0.003,
                        d: int = 16,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    F-, T-, L+: labels predictable from label smoothness alone.

    Graph: assortative SBM with MATCHED expected degrees across blocks.
      - p_intra >> p_inter -> high homophily -> L-signal ON.
      - Uniform degree weights [1, 1, 1] -> same degree distribution across
        blocks -> T-signal OFF (no structural fingerprint).
    Features: pure noise -> F-signal OFF.
    Labels: aligned with community membership.

    Expected: low FSS, low TSS, high LSS. Best model: LP ≈ APPNP >> MLP.
    """
    rng = np.random.RandomState(seed)
    y = _contiguous_labels(num_nodes, num_classes)
    edge_index = _dc_sbm(num_nodes, num_classes, y,
                         p_intra, p_inter, _MATCHED_WEIGHTS, rng)
    X = _noise_features(num_nodes, d, rng)
    return edge_index, X, y



# =============================================================
# 4. FT-DOMINANT  (F+, T+, L-)
# =============================================================

def generate_ft_dominant(num_nodes: int = 1500,
                         num_classes: int = 3,
                         p_intra: float = 0.002,
                         p_inter: float = 0.010,
                         feature_sep: float = 0.5,
                         d: int = 16,
                         seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    F+, T+, L-: labels predictable from features OR topology, not smoothness.

    Graph: disassortative DC-SBM with DISTINCT expected degrees per block.
      - p_inter >> p_intra -> h ≈ 1/C -> L-signal OFF.
      - Distinct degree weights -> structurally identifiable blocks -> T-signal ON.
    Features: class-conditional Gaussians -> F-signal ON.
    Labels: aligned with block membership (consistent with both channels).

    Expected: high FSS, high TSS, low LSS. Best models: MLP and GPS both strong.
    """
    rng = np.random.RandomState(seed)
    y = _contiguous_labels(num_nodes, num_classes)
    edge_index = _dc_sbm(num_nodes, num_classes, y,
                         p_intra, p_inter, _DISTINCT_WEIGHTS, rng)
    X = _class_conditional_features(num_nodes, d, y, num_classes, feature_sep, rng)
    return edge_index, X, y


# =============================================================
# 5. FL-DOMINANT  (F+, T-, L+)
# =============================================================

def generate_fl_dominant(num_nodes: int = 1500,
                         num_classes: int = 3,
                         p_intra: float = 0.012,
                         p_inter: float = 0.003,
                         feature_sep: float = 0.5,
                         d: int = 16,
                         seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    F+, T-, L+: labels predictable from features OR smoothness, not topology.

    Graph: assortative SBM with MATCHED expected degrees across blocks.
      - p_intra >> p_inter -> high homophily -> L-signal ON.
      - Uniform degree weights -> no structural fingerprint -> T-signal OFF.
    Features: class-conditional Gaussians -> F-signal ON.
    Labels: aligned with communities (consistent across both channels).

    Expected: high FSS, low TSS, high LSS. Best models: MLP and LP both strong.
    """
    rng = np.random.RandomState(seed)
    y = _contiguous_labels(num_nodes, num_classes)
    edge_index = _dc_sbm(num_nodes, num_classes, y,
                         p_intra, p_inter, _MATCHED_WEIGHTS, rng)
    X = _class_conditional_features(num_nodes, d, y, num_classes, feature_sep, rng)
    return edge_index, X, y


# =============================================================
# 6. TL-DOMINANT  (F-, T+, L+)
# =============================================================

def generate_tl_dominant(num_nodes: int = 1500,
                         num_classes: int = 3,
                         p_intra: float = 0.012,
                         p_inter: float = 0.003,
                         d: int = 16,
                         seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    F-, T+, L+: labels predictable from topology OR smoothness, not features.

    Graph: assortative DC-SBM with DISTINCT expected degrees per block.
      - p_intra >> p_inter -> high homophily -> L-signal ON.
      - Distinct degree weights -> structurally identifiable blocks -> T-signal ON.
    Features: pure noise -> F-signal OFF.
    Labels: aligned with block membership.

    This is the most common real-world heterophily regime where nodes of
    distinct structural roles form communities with high intra-class edge density.

    Expected: low FSS, high TSS, high LSS. Best models: GPS and LP both strong.
    """
    rng = np.random.RandomState(seed)
    y = _contiguous_labels(num_nodes, num_classes)
    edge_index = _dc_sbm(num_nodes, num_classes, y,
                         p_intra, p_inter, _DISTINCT_WEIGHTS, rng)
    X = _noise_features(num_nodes, d, rng)
    return edge_index, X, y


# =============================================================
# 7. FTL-DOMINANT  (F+, T+, L+)
# =============================================================

def generate_ftl_dominant(num_nodes: int = 1500,
                          num_classes: int = 3,
                          p_intra: float = 0.012,
                          p_inter: float = 0.003,
                          feature_sep: float = 0.5,
                          d: int = 16,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    F+, T+, L+: all three signal channels active simultaneously.

    Graph: assortative DC-SBM with DISTINCT expected degrees per block.
      - p_intra >> p_inter -> high homophily -> L-signal ON.
      - Distinct degree weights -> structurally identifiable blocks -> T-signal ON.
    Features: class-conditional Gaussians -> F-signal ON.
    Labels: aligned with block membership (consistent across all channels).

    Note: the three channels are correlated by construction here -- the same
    label assignment (block membership) drives FSS, TSS, and LSS simultaneously.
    This dataset does not demonstrate channel independence; it shows that all
    three channels can be simultaneously active. See the pure-regime datasets
    (F, T, L) for evidence that each channel can be activated in isolation.

    Expected: high FSS, high TSS, high LSS. All models competitive.
    """
    rng = np.random.RandomState(seed)
    y = _contiguous_labels(num_nodes, num_classes)
    edge_index = _dc_sbm(num_nodes, num_classes, y,
                         p_intra, p_inter, _DISTINCT_WEIGHTS, rng)
    X = _class_conditional_features(num_nodes, d, y, num_classes, feature_sep, rng)
    return edge_index, X, y


# =============================================================
# 8. C-DOMINANT  (F-, T-, L-)
# =============================================================

def generate_c_dominant(num_nodes: int = 1500,
                        num_classes: int = 3,
                        avg_degree: float = 8.0,
                        d: int = 16,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    C-dominant (Node Twins): F-, T-, L-.

    Labels depend ONLY on the coupling between structural role and feature type.
    All three marginal channels are near chance by construction.

    Construction:
      - Erdos-Renyi graph (avg_degree ~8) -> h ~= 1/C -> L-signal OFF.
        ER gives a roughly Poisson degree distribution; roles are defined by
        degree tertiles which are naturally balanced over a Poisson distribution,
        so the three role groups are well-populated without relying on block
        structure. Using ER (rather than a near-random SBM) avoids any residual
        block signal leaking into TSS.
      - Three orthogonal feature prototypes assigned so that every class
        contains all three prototypes in equal proportion -> F-signal OFF.
      - Labels follow y = (role + prototype_index) mod 3, so joint
        role-feature reasoning recovers labels but no marginal channel can.

    Role assignment:
      - Degree tertiles define roles r in {0, 1, 2} after graph generation.
      - The 9 node types (r, f) are filled with num_nodes//9 nodes each.
        Remainder nodes are distributed evenly across the diagonal types.
      - Class c contains node type (r, f) iff (r + f) mod 3 == c, so
        every class has equal numbers of all roles and all prototypes.

    This is a genuine node classification problem (labels are recoverable
    by a model that jointly observes role and feature type) but all three
    NSA marginal channels are near 1/C = 0.33.

    Expected: low FSS, low TSS, low LSS. All standard models near chance.
    A model with explicit role-feature interaction could exceed chance.
    """
    rng = np.random.RandomState(seed)

    # Erdos-Renyi graph: Poisson degrees, no block structure -> T- and L-
    edge_index = _er_graph(num_nodes, avg_degree=avg_degree, rng=rng)

    # Degree values used for role assignment by rank (see below)
    deg = np.zeros(num_nodes, dtype=np.float32)
    if edge_index.shape[1] > 0:
        np.add.at(deg, edge_index[0], 1)

    # Assign roles by random balanced partition rather than degree tertiles.
    # This guarantees exactly num_nodes//9 * 9 nodes are cleanly assigned.
    # We still use degree tertiles to DEFINE which degree stratum each role
    # represents, but we assign nodes by shuffled rank rather than threshold
    # to ensure exact balance.
    deg_order = np.argsort(deg)            # nodes sorted by degree, low -> high
    n_usable  = (num_nodes // 9) * 9      # largest multiple of 9 <= num_nodes
    usable    = deg_order[:n_usable]       # drop at most 8 high-degree nodes

    # Divide sorted nodes into 3 equal role groups (by degree rank)
    n_per_role = n_usable // 3
    role_groups = [usable[r * n_per_role:(r + 1) * n_per_role]
                   for r in range(3)]
    roles_assigned = np.zeros(num_nodes, dtype=int)
    for r, grp in enumerate(role_groups):
        roles_assigned[grp] = r

    # Three orthogonal feature prototypes
    proto_raw = rng.randn(3, d)
    Q, _ = np.linalg.qr(proto_raw.T)
    prototypes = (Q.T * 2.0 * np.sqrt(d)).astype(np.float32)

    # Within each role group, shuffle and assign equal thirds to 3 prototypes
    n_per_type = n_per_role // 3          # exact: n_usable // 9
    node_proto = np.zeros(num_nodes, dtype=int)

    for r in range(3):
        grp = role_groups[r].copy()
        rng.shuffle(grp)
        for f in range(3):
            node_proto[grp[f * n_per_type:(f + 1) * n_per_type]] = f

    # Restrict to usable nodes only (drop at most 8 remainder nodes)
    keep = usable
    edge_keep = np.zeros(num_nodes, dtype=bool)
    edge_keep[keep] = True
    remap = -np.ones(num_nodes, dtype=int)
    remap[keep] = np.arange(len(keep))

    if edge_index.shape[1] > 0:
        u_ei, v_ei = edge_index
        valid = edge_keep[u_ei] & edge_keep[v_ei]
        edge_index = np.stack([
            remap[u_ei[valid]], remap[v_ei[valid]]
        ]).astype(np.int64)

    roles_f     = roles_assigned[keep]
    node_proto_f = node_proto[keep]

    # Labels and features (vectorized): y = (role + prototype) mod 3
    y = (roles_f + node_proto_f) % 3
    noise = rng.randn(len(keep), d).astype(np.float32) * 0.3
    X = prototypes[node_proto_f] + noise

    return edge_index, X, y



# =============================================================
# MEDIUM-STRENGTH PRESETS
# =============================================================
# These interpolate between OFF and ON for each channel:
#   F-medium: feature_sep=0.25  (half of 0.5)
#   T-medium: degree weights [1.0, 1.5, 2.0]  (half contrast)
#   L-medium: p_intra=0.006, p_inter=0.003  (halfway between L-off and L-on)

_MEDIUM_WEIGHTS = np.array([1.0, 1.5, 2.0])  # T-signal MEDIUM

_F_SEP_OFF = 0.0    # noise features
_F_SEP_MED = 0.25   # medium feature separation
_F_SEP_ON  = 0.5    # full feature separation


def _make_sweep_dataset(num_nodes: int,
                         num_classes: int,
                         d: int,
                         seed: int,
                         f_level: str,    # 'off', 'med', 'on'
                         t_level: str,    # 'off', 'med', 'on'
                         l_level: str,    # 'off', 'med', 'on'
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic generator for sweep datasets.

    Each channel is controlled independently by interpolating parameters:

    f_level: 'off' -> noise features (feature_sep=0)
             'med' -> feature_sep=0.25
             'on'  -> feature_sep=0.5

    t_level: controls degree weight contrast (structural fingerprint strength)
             'off' -> matched weights [1, 1, 1]        (no T-signal)
             'med' -> medium weights  [1, 1.5, 2.0]    (medium T-signal)
             'on'  -> distinct weights [1, 2.0, 3.5]   (full T-signal)

    l_level: controls edge assortativity (smoothness strength)
             'off' -> p_intra=0.003, p_inter=0.003   (near-random, h~1/C)
             'med' -> p_intra=0.006, p_inter=0.003   (moderate homophily)
             'on'  -> p_intra=0.012, p_inter=0.003   (high homophily)

    T and L are decoupled: degree weights control T-signal independently of
    edge assortativity. A dataset can have T-medium + L-medium simultaneously
    (assortative DC-SBM with moderate degree contrast).

    F is always post-hoc (feature assignment does not affect graph generation).

    Special case: F-only (t_level='off', l_level='off') uses an ER graph with
    shuffled labels to guarantee exactly zero graph-label correlation.
    """
    rng = np.random.RandomState(seed)

    # Degree weights (T-signal)
    dw = {'off': _MATCHED_WEIGHTS,
          'med': _MEDIUM_WEIGHTS,
          'on':  _DISTINCT_WEIGHTS}[t_level]

    # Edge assortativity (L-signal) -- independent of T
    p_intra = {'off': 0.003, 'med': 0.006, 'on': 0.012}[l_level]
    p_inter = 0.003

    # Graph and labels
    if t_level == 'off' and l_level == 'off':
        # No structural label signal: use ER + shuffled labels
        y = _balanced_labels(num_nodes, num_classes, rng)
        edge_index = _er_graph(num_nodes, avg_degree=4.0, rng=rng)
    else:
        y = _contiguous_labels(num_nodes, num_classes)
        edge_index = _dc_sbm(num_nodes, num_classes, y,
                             p_intra, p_inter, dw, rng)

    # Features (F-signal) -- post-hoc, independent of graph
    sep = {'off': 0.0, 'med': _F_SEP_MED, 'on': _F_SEP_ON}[f_level]
    if sep == 0.0:
        X = _noise_features(num_nodes, d, rng)
    else:
        X = _class_conditional_features(num_nodes, d, y,
                                         num_classes, sep, rng)
    return edge_index, X, y


# --- 16 sweep datasets ---

def generate_fm(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-medium, T-off, L-off"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'med', 'off', 'off')

def generate_tm(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-off, T-medium, L-off"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'off', 'med', 'off')

def generate_lm(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-off, T-off, L-medium"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'off', 'off', 'med')

def generate_fmt(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-medium, T-on, L-off"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'med', 'on', 'off')

def generate_fml(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-medium, T-off, L-on"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'med', 'off', 'on')

def generate_ftm(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-on, T-medium, L-off"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'on', 'med', 'off')

def generate_flm(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-on, T-off, L-medium"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'on', 'off', 'med')

def generate_tml(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-off, T-medium, L-on"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'off', 'med', 'on')

def generate_tlm(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-off, T-on, L-medium"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'off', 'on', 'med')

def generate_fmtm(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-medium, T-medium, L-off"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'med', 'med', 'off')

def generate_fmlm(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-medium, T-off, L-medium"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'med', 'off', 'med')

def generate_tmlm(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-off, T-medium, L-medium"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'off', 'med', 'med')

def generate_fmtl(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-medium, T-on, L-on"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'med', 'on', 'on')

def generate_ftml(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-on, T-medium, L-on"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'on', 'med', 'on')

def generate_ftlm(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-on, T-on, L-medium"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'on', 'on', 'med')

def generate_fmtmlm(num_nodes=1500, num_classes=3, d=16, seed=42):
    """F-medium, T-medium, L-medium (interior point)"""
    return _make_sweep_dataset(num_nodes, num_classes, d, seed, 'med', 'med', 'med')


SWEEP_DATASETS = {
    'Fm':     generate_fm,
    'Tm':     generate_tm,
    'Lm':     generate_lm,
    'FmT':    generate_fmt,
    'FmL':    generate_fml,
    'FTm':    generate_ftm,
    'FLm':    generate_flm,
    'TmL':    generate_tml,
    'TLm':    generate_tlm,
    'FmTm':   generate_fmtm,
    'FmLm':   generate_fmlm,
    'TmLm':   generate_tmlm,
    'FmTL':   generate_fmtl,
    'FTmL':   generate_ftml,
    'FTLm':   generate_ftlm,
    'FmTmLm': generate_fmtmlm,
}

# =============================================================
# REGISTRY
# =============================================================

ALL_SYNTHETIC_DATASETS = {
    # 8 corner datasets (pure/combined channels fully on/off)
    'F':   generate_f_dominant,
    'T':   generate_t_dominant,
    'L':   generate_l_dominant,
    'FT':  generate_ft_dominant,
    'FL':  generate_fl_dominant,
    'TL':  generate_tl_dominant,
    'FTL': generate_ftl_dominant,
    'C':   generate_c_dominant,
    # 16 sweep datasets (medium-strength variants)
    **SWEEP_DATASETS,
}

EXPECTED_DOMINANT = {
    # 8 corner datasets
    'F':   'F',
    'T':   'T',
    'L':   'L',
    'FT':  'FT',
    'FL':  'FL',
    'TL':  'TL',
    'FTL': 'FTL',
    'C':   'none',
    # 16 sweep datasets (medium-strength variants)
    # 'none' means no single dominant channel -- multiple are active at medium
    'Fm':     'F',      # only F active (at medium)
    'Tm':     'T',      # only T active (at medium)
    'Lm':     'L',      # only L active (at medium)
    'FmT':    'FT',     # F-medium + T-on
    'FmL':    'FL',     # F-medium + L-on
    'FTm':    'FT',     # F-on + T-medium
    'FLm':    'FL',     # F-on + L-medium
    'TmL':    'TL',     # T-medium + L-on
    'TLm':    'TL',     # T-on + L-medium
    'FmTm':   'FT',     # F-medium + T-medium
    'FmLm':   'FL',     # F-medium + L-medium
    'TmLm':   'TL',     # T-medium + L-medium
    'FmTL':   'FTL',    # F-medium + T-on + L-on
    'FTmL':   'FTL',    # F-on + T-medium + L-on
    'FTLm':   'FTL',    # F-on + T-on + L-medium
    'FmTmLm': 'FTL',    # all three at medium
}


def generate_all_synthetic(num_nodes: int = 1500,
                           num_classes: int = 3,
                           d: int = 16,
                           seed: int = 42) -> dict:
    """
    Generate all 24 synthetic datasets (8 corners + 16 sweep) with consistent parameters.
    Returns dict mapping name -> (edge_index, X, y).
    """
    # C-dominant uses avg_degree instead of p_intra/p_inter
    c_kwargs  = dict(num_nodes=num_nodes, num_classes=num_classes,
                     d=d, seed=seed)
    std_kwargs = dict(num_nodes=num_nodes, num_classes=num_classes,
                      d=d, seed=seed)

    datasets = {}
    for name, fn in ALL_SYNTHETIC_DATASETS.items():
        datasets[name] = fn(**std_kwargs)
    return datasets


# =============================================================
# ALIGNMENT PLANE SWEEP
# =============================================================

def generate_alignment_sweep(n_points: int = 5,
                              num_nodes: int = 500,
                              num_classes: int = 3,
                              d: int = 16,
                              seed: int = 42) -> List[Tuple]:
    """
    Grid sweep over (structural_sep, feature_sep).
    structural_sep controls SBM p_intra (L-signal strength).
    feature_sep controls Gaussian class-mean separation (F-signal strength).

    Returns list of (edge_index, X, y, s_param, f_param).
    Used for Experiment 2 (alignment plane validation).
    """
    datasets = []
    s_values = np.linspace(0.03, 0.20, n_points)   # structural separation
    f_values = np.linspace(0.0,  4.0,  n_points)   # feature separation

    for i, s in enumerate(s_values):
        for j, f in enumerate(f_values):
            rng = np.random.RandomState(seed + i * n_points + j)
            p_intra = float(s)
            p_inter = max(p_intra / 8.0, 0.005)

            y = _contiguous_labels(num_nodes, num_classes)
            edge_index = _dc_sbm(num_nodes, num_classes, y,
                                  p_intra, p_inter, _MATCHED_WEIGHTS, rng)

            if f > 0:
                X = _class_conditional_features(
                    num_nodes, d, y, num_classes, f, rng)
            else:
                X = _noise_features(num_nodes, d, rng)

            datasets.append((edge_index, X, y, s, f))

    return datasets
