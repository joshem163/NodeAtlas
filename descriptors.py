"""
Node Classification Diagnostic Descriptors
===========================================
FSS: Feature Signal Score   -- label information in node feature geometry
TSS: Topology Signal Score  -- label information in graph structure
LSS: Label Smoothness Score -- label information in neighborhood label distributions

Annulus-based descriptors at hops k=0,1,2,3.
Single BFS per node per seed; all three descriptors share frontiers.

Descriptor dimensions (r = projection dim, C = num classes):
  FSS: R^{8r+4}      (mean, std, shift per hop)
  TSS: R^{30}        (3 at k=0, 9 per hop k=1,2,3)
  LSS: R^{3C+6}      (class dist + cov_frac + log_mk per hop k=1,2,3)

Scores are normalized balanced accuracy: (macro_acc - 1/C) / (1 - 1/C).
Maps chance (1/C) to 0, perfect accuracy to 1.
Directly comparable to baseline model scores on the same dataset.
"""

import warnings
import numpy as np
from collections import Counter
from typing import Optional, Dict, List, Tuple

from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

warnings.filterwarnings('ignore')

EPS = 1e-8


def normalize_score(score, C):
    """Normalize: (score - 1/C) / (1 - 1/C). Maps chance→0, perfect→1."""
    chance = 1.0 / C
    denom = 1.0 - chance
    if denom < EPS:
        return 0.0
    return (score - chance) / denom


# =============================================================
# GRAPH CONSTRUCTION
# =============================================================

def build_adjacency_list(edge_index: np.ndarray,
                         num_nodes: int) -> List[List[int]]:
    """Undirected adjacency list. Self-loops removed."""
    adj = [set() for _ in range(num_nodes)]
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return [sorted(s) for s in adj]


def build_sparse_adj(edge_index: np.ndarray,
                     num_nodes: int) -> sparse.csr_matrix:
    """Symmetric binary sparse adjacency. Self-loops removed."""
    src = edge_index[0].astype(int)
    dst = edge_index[1].astype(int)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=np.float32)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    A.data[:] = 1.0   # binarize after deduplication
    return A


# =============================================================
# GLOBAL PRECOMPUTATION
# =============================================================

def compute_degrees(A: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(A.sum(axis=1)).flatten().astype(np.float32)


def compute_ego_density(A: sparse.csr_matrix,
                        degrees: np.ndarray) -> np.ndarray:
    """
    Egonet density: 2|E(N(v))| / (deg(v)*(deg(v)-1) + eps).

    |E(N(v))| = edges among neighbors of v = triangles through v.
    Computed via Hadamard product: tri(v) = (A ⊙ A²).sum(axis=1)[v] / 2.
    Each triangle v-u-w contributes A[v,u]*A²[v,u] once for path v->u->w->v,
    and A[v,w]*A²[v,w] once — so dividing row sum by 2 gives triangle count.

    O(|E| * d_avg). Result in [0, 1].
    """
    num_nodes = A.shape[0]
    A2 = A @ A
    # Element-wise product A ⊙ A², then row-sum / 2 = triangles per node
    edges_among_nbrs = np.asarray(A.multiply(A2).sum(axis=1)).flatten() / 2.0
    denom = degrees * (degrees - 1) + EPS
    ego = np.zeros(num_nodes, dtype=np.float32)
    ego = (2.0 * edges_among_nbrs / denom).astype(np.float32)
    return np.clip(ego, 0.0, 1.0)


def compute_coreness(adj: List[List[int]], num_nodes: int) -> np.ndarray:
    """K-core decomposition. Falls back to log(degree) if networkx absent."""
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        for v in range(num_nodes):
            for u in adj[v]:
                if u > v:
                    G.add_edge(v, u)
        cores = nx.core_number(G)
        return np.array([cores[v] for v in range(num_nodes)], dtype=np.float32)
    except ImportError:
        warnings.warn("networkx not found. Using log(degree+1) as coreness proxy.")
        return np.array([np.log(len(adj[v]) + 1) for v in range(num_nodes)],
                        dtype=np.float32)


def compute_wl_colors(adj: List[List[int]], num_nodes: int,
                      degrees: np.ndarray,
                      T: int = 4, B: int = 1024) -> np.ndarray:
    """
    1-WL color refinement. O(|E| * T).
    Initialized by binned log-degree.
    Colors hash-bucketed into B bins for entropy stability.
    Returns (num_nodes, T) int32 array of bucket indices.
    """
    # Initialize: bin log-degree into B buckets
    log_deg = np.floor(np.log2(degrees + 1)).astype(int)
    current = (log_deg % B).tolist()

    wl = np.zeros((num_nodes, T), dtype=np.int32)
    color_map: Dict[tuple, int] = {}

    for t in range(T):
        new_colors = []
        for v in range(num_nodes):
            nbr_sig = tuple(sorted(current[u] for u in adj[v]))
            sig = (current[v], nbr_sig)
            if sig not in color_map:
                color_map[sig] = len(color_map) % B
            new_colors.append(color_map[sig])
        current = new_colors
        wl[:, t] = np.array(current, dtype=np.int32)

    return wl


def random_projection(X: np.ndarray, r: int, seed: int = 42) -> np.ndarray:
    """(N, d) -> (N, r) scaled Gaussian projection. Identity if d <= r."""
    d = X.shape[1]
    if d <= r:
        return X.astype(np.float32)
    rng = np.random.RandomState(seed)
    R = rng.randn(d, r).astype(np.float32) / np.sqrt(r)
    return X.astype(np.float32) @ R


# =============================================================
# ANNULUS BFS  (single pass, exact distances, sampled frontiers)
# =============================================================

def annulus_frontiers(v: int,
                      adj: List[List[int]],
                      K: int,
                      m: Tuple[int, int, int],
                      seed: int) -> List[np.ndarray]:
    """
    BFS from v returning sampled exact-distance frontiers F[0..K].

    KEY INVARIANT: Seen and F_full are updated from the FULL candidate
    set so that hop distances are exact regardless of sampling.
    F[k] returned for descriptors may be sampled, but contains only
    true hop-k nodes.

    Args:
        v:    source node
        adj:  adjacency list
        K:    max hop
        m:    (m1, m2, m3) max sample sizes per hop
        seed: sampling seed

    Returns:
        frontiers: list of length K+1; frontiers[k] = np.ndarray of node ids
    """
    rng = np.random.RandomState(seed)
    F_full = [None] * (K + 1)
    F_samp = [None] * (K + 1)
    seen = {v}

    F_full[0] = np.array([v], dtype=np.int64)
    F_samp[0] = F_full[0]

    for k in range(1, K + 1):
        # Expand from FULL frontier to preserve exact distances
        cand_set = set()
        for u in F_full[k - 1]:
            for w in adj[u]:
                if w not in seen:
                    cand_set.add(w)
        seen.update(cand_set)
        F_full[k] = np.array(sorted(cand_set), dtype=np.int64)

        mk = m[k - 1]  # m is (m1,m2,m3) indexed from 0
        if len(F_full[k]) > mk:
            idx = rng.choice(len(F_full[k]), mk, replace=False)
            F_samp[k] = F_full[k][idx]
        else:
            F_samp[k] = F_full[k]

    return F_samp   # frontiers[k] = sampled hop-k nodes


# =============================================================
# DESCRIPTOR FUNCTIONS  (all take pre-computed frontiers)
# =============================================================

def fss_from_frontiers(v: int,
                       frontiers: List[np.ndarray],
                       Xr: np.ndarray) -> np.ndarray:
    """
    FSS descriptor for node v.

    phi_F(v, k) = [mean_k | std_k | shift_k] in R^{2r+1}
    shift_k = ||mean_k - Xr[v]||_2  (feature drift at hop k)
    k=0: mean=Xr[v], std=0, shift=0 (degenerate but consistent)

    Returns R^{4*(2r+1)} = R^{8r+4}
    """
    r = Xr.shape[1]
    block = 2 * r + 1
    phi = np.zeros(4 * block, dtype=np.float32)

    xv = Xr[v]

    for k, S in enumerate(frontiers):
        offset = k * block
        if len(S) == 0:
            pass  # zero-padded
        elif k == 0:
            phi[offset:offset + r] = xv   # mean = self
            # std = 0, shift = 0
        else:
            feats = Xr[S]
            mu = feats.mean(axis=0)
            sigma = feats.std(axis=0) if len(S) > 1 else np.zeros(r)
            shift = float(np.linalg.norm(mu - xv))
            phi[offset:offset + r]         = mu
            phi[offset + r:offset + 2 * r] = sigma
            phi[offset + 2 * r]            = shift

    return phi


def _local_wl_entropy(S: np.ndarray, wl: np.ndarray, T: int) -> float:
    """Mean WL color entropy over T iterations, normalized by log(|S|+1)."""
    nk = len(S)
    if nk == 0:
        return 0.0
    norm = np.log(nk + 1)
    if norm < EPS:
        return 0.0
    ent = 0.0
    for t in range(T):
        counts = np.bincount(wl[S, t])
        counts = counts[counts > 0]
        if len(counts) == 1:
            continue   # zero entropy
        p = counts / nk
        ent += -float(np.sum(p * np.log(p + EPS))) / norm
    return ent / T


def tss_from_frontiers(v: int,
                       frontiers: List[np.ndarray],
                       degrees: np.ndarray,
                       ego_dens: np.ndarray,
                       coreness: np.ndarray,
                       wl: np.ndarray,
                       T: int) -> np.ndarray:
    """
    TSS descriptor for node v.

    k=0 block (R^3): [log(deg+1), core, ego_dens]
    k=1,2,3 blocks (R^9 each):
        [log_nk, mean_deg, std_deg, exp_ratio, surf_ratio,
         mean_core, mean_ego, wl_ent, log_bk]

    NOTE: ego_dens in [0,1], deg unbounded. StandardScaler in the
    probe is load-bearing — without it degree features dominate.

    Returns R^{3 + 3*9} = R^{30}
    frontiers must have at least 4 entries (k=0,1,2,3).
    """
    phi = np.zeros(30, dtype=np.float32)
    K_actual = len(frontiers) - 1   # number of hops available

    # k=0 block
    phi[0] = float(np.log(degrees[v] + 1))
    phi[1] = float(coreness[v])
    phi[2] = float(ego_dens[v])

    # Ball sizes for surf_ratio (only for available hops)
    b = [1] + [0] * K_actual
    for k in range(1, K_actual + 1):
        b[k] = b[k - 1] + len(frontiers[k])

    offset = 3
    for k in range(1, 4):
        # If K < 3, hops beyond K_actual are zero-padded (empty annulus)
        if k > K_actual:
            offset += 9
            continue

        S = frontiers[k]
        nk = len(S)
        if nk == 0:
            offset += 9
            continue

        log_nk    = float(np.log(nk + 1))
        log_deg_S = np.log(degrees[S] + 1)
        mean_deg  = float(log_deg_S.mean())
        std_deg   = float(log_deg_S.std()) if nk > 1 else 0.0
        prev_size = max(len(frontiers[k - 1]), 1)
        exp_ratio = nk / prev_size
        surf_ratio = nk / (b[k] + 1)
        mean_core = float(coreness[S].mean())
        mean_ego  = float(ego_dens[S].mean())
        wl_ent    = _local_wl_entropy(S, wl, T)
        log_bk    = float(np.log(b[k] + 1))

        phi[offset:offset + 9] = [log_nk, mean_deg, std_deg,
                                   exp_ratio, surf_ratio,
                                   mean_core, mean_ego,
                                   wl_ent, log_bk]
        offset += 9

    return phi


def lss_from_frontiers(v: int,
                       frontiers: List[np.ndarray],
                       labels: np.ndarray,
                       train_mask: np.ndarray,
                       C: int) -> np.ndarray:
    """
    LSS descriptor for node v.

    k=0 omitted: Y[v] trivially predicts Y[v].
    Per hop k=1,2,3: [p_k | cov_frac | log_mk] in R^{C+2}

    cov_frac disambiguates "uniform due to mix" from "uniform due to
    no labeled neighbors" — critical for semi-supervised settings.

    Returns R^{3*(C+2)} = R^{3C+6}
    """
    block = C + 2
    phi = np.zeros(3 * block, dtype=np.float32)
    uniform = np.ones(C, dtype=np.float32) / C

    for i, k in enumerate([1, 2, 3]):
        offset = i * block
        # If K < k, this hop is empty — use uniform + zero coverage
        if k >= len(frontiers):
            phi[offset:offset + C] = uniform
            # cov_frac = 0, log_mk = 0 (already zeros)
            continue

        S = frontiers[k]
        nk = len(S)
        S_lab = S[train_mask[S]] if nk > 0 else np.array([], dtype=np.int64)
        mk = len(S_lab)

        if mk == 0:
            phi[offset:offset + C] = uniform
        else:
            p = np.zeros(C, dtype=np.float32)
            for u in S_lab:
                p[int(labels[u])] += 1
            p /= p.sum()
            phi[offset:offset + C] = p

        phi[offset + C]     = mk / (nk + 1)         # cov_frac
        phi[offset + C + 1] = float(np.log(mk + 1)) # log_mk

    return phi


def all_descriptors(v: int,
                    adj: List[List[int]],
                    Xr: np.ndarray,
                    degrees: np.ndarray,
                    ego_dens: np.ndarray,
                    coreness: np.ndarray,
                    wl: np.ndarray,
                    labels: np.ndarray,
                    train_mask: np.ndarray,
                    C: int,
                    K: int,
                    m: Tuple[int, int, int],
                    T: int,
                    seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single BFS, then compute all three descriptors.
    Call this in the main loop to avoid 3x redundant traversals.
    """
    frontiers = annulus_frontiers(v, adj, K, m, seed)
    phi_F = fss_from_frontiers(v, frontiers, Xr)
    phi_S = tss_from_frontiers(v, frontiers, degrees, ego_dens,
                                coreness, wl, T)
    phi_H = lss_from_frontiers(v, frontiers, labels, train_mask, C)
    return phi_F, phi_S, phi_H


# =============================================================
# DESIGN MATRIX BUILDER
# =============================================================

def build_design_matrices(train_idx: np.ndarray,
                           adj: List[List[int]],
                           Xr: np.ndarray,
                           degrees: np.ndarray,
                           ego_dens: np.ndarray,
                           coreness: np.ndarray,
                           wl: np.ndarray,
                           labels: np.ndarray,
                           train_mask: np.ndarray,
                           C: int,
                           K: int = 3,
                           m: Tuple[int, int, int] = (64, 64, 64),
                           T: int = 4,
                           seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build FSS, TSS, LSS design matrices for training nodes.
    One BFS per node (via all_descriptors).
    """
    r = Xr.shape[1]
    n_train = len(train_idx)

    X_F = np.zeros((n_train, 4 * (2 * r + 1)), dtype=np.float32)
    X_S = np.zeros((n_train, 30), dtype=np.float32)
    X_H = np.zeros((n_train, 3 * (C + 2)), dtype=np.float32)

    for i, v in enumerate(train_idx):
        phi_F, phi_S, phi_H = all_descriptors(
            int(v), adj, Xr, degrees, ego_dens, coreness, wl,
            labels, train_mask, C, K, m, T, seed)
        X_F[i] = phi_F
        X_S[i] = phi_S
        X_H[i] = phi_H

    return X_F, X_S, X_H


def build_frontiers_cache(train_idx: np.ndarray,
                          adj: List[List[int]],
                          K: int,
                          m: Tuple[int, int, int],
                          seed: int) -> List[List[np.ndarray]]:
    """BFS frontiers for every training node, computed once per sampling seed.

    Shared across FSS/TSS/LSS so the three channels see identical annuli, and
    so the LSS design matrix can be rebuilt per CV fold without redoing BFS.
    """
    return [annulus_frontiers(int(v), adj, K, m, seed) for v in train_idx]


def build_lss_matrix(train_idx: np.ndarray,
                     frontiers_cache: List[List[np.ndarray]],
                     labels: np.ndarray,
                     label_mask: np.ndarray,
                     C: int) -> np.ndarray:
    """LSS design matrix using only the labels visible in `label_mask`.

    label_mask is a boolean mask over ALL nodes. For leakage-free CV it must be
    the inner-training partition of the current fold, NOT the full train_mask:
    otherwise a held-out node's label reaches the descriptor of a neighbour
    that is being scored in the same fold.
    """
    X_H = np.zeros((len(train_idx), 3 * (C + 2)), dtype=np.float32)
    for i, v in enumerate(train_idx):
        X_H[i] = lss_from_frontiers(int(v), frontiers_cache[i],
                                    labels, label_mask, C)
    return X_H


# =============================================================
# PROBE  (stratified CV, balanced accuracy / macro, per-node scores)
# =============================================================

def run_probe(X: np.ndarray,
              y: np.ndarray,
              n_splits: int = 5,
              cv_seed: int = 0,
              metric: str = 'macro') -> Tuple[float, np.ndarray]:
    """
    L2 logistic regression probe with stratified K-fold CV.
    StandardScaler is fit within each fold to avoid data leakage.

    Args:
        metric: 'macro' for balanced accuracy (default), 'auc' for ROC-AUC, 'acc' for micro-accuracy.
                Use 'auc' for class-imbalanced binary datasets.

    Returns:
        score:        mean normalized score across folds
                      (macro: (balanced_acc - 1/C) / (1 - 1/C), auc: raw)
        node_scores:  per-node predicted probability of true class
                      using HELD-OUT fold predictions (no in-sample leakage)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    C = len(np.unique(y))

    scores = []
    node_scores = np.zeros(len(y), dtype=np.float32)

    for tr_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_val = scaler.transform(X[val_idx])

        clf = LogisticRegression(max_iter=1000, random_state=cv_seed,
                                 solver='lbfgs', C=1.0)
        clf.fit(X_tr, y[tr_idx])

        proba = clf.predict_proba(X_val)
        classes = list(clf.classes_)

        if metric == 'auc':
            if proba.shape[1] == 2:
                fold_score = roc_auc_score(y[val_idx], proba[:, 1])
            else:
                fold_score = roc_auc_score(y[val_idx], proba,
                                           multi_class='ovr')
        elif metric == 'macro':
            preds = clf.predict(X_val)
            fold_score = normalize_score(
                balanced_accuracy_score(y[val_idx], preds), C)
        else:
            preds = clf.predict(X_val)
            fold_score = accuracy_score(y[val_idx], preds)

        scores.append(fold_score)

        # Out-of-sample probabilities for per-node scores
        for i, (idx, lbl) in enumerate(zip(val_idx, y[val_idx])):
            if int(lbl) in classes:
                node_scores[idx] = proba[i, classes.index(int(lbl))]

    return float(np.mean(scores)), node_scores


def run_probe_lss(train_idx: np.ndarray,
                  frontiers_cache: List[List[np.ndarray]],
                  labels: np.ndarray,
                  C: int,
                  n_splits: int = 5,
                  cv_seed: int = 0,
                  metric: str = 'macro') -> Tuple[float, np.ndarray]:
    """LSS probe with per-fold descriptor reconstruction.

    Unlike FSS/TSS, the LSS descriptor is a function of observed labels, so it
    cannot be built once and then cross-validated: the validation fold's labels
    would already sit inside the descriptors of their neighbours. For every
    fold the design matrix is therefore rebuilt from that fold's inner-training
    labels alone. Everything else (scaler, probe, folds) matches `run_probe`.
    """
    y_train = labels[train_idx]
    num_nodes = len(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)

    scores = []
    node_scores = np.zeros(len(train_idx), dtype=np.float32)

    for inner_tr, inner_val in skf.split(np.zeros(len(train_idx)), y_train):
        label_mask = np.zeros(num_nodes, dtype=bool)
        label_mask[train_idx[inner_tr]] = True
        X_H = build_lss_matrix(train_idx, frontiers_cache, labels,
                               label_mask, C)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_H[inner_tr])
        X_val = scaler.transform(X_H[inner_val])

        clf = LogisticRegression(max_iter=1000, random_state=cv_seed,
                                 solver='lbfgs', C=1.0)
        clf.fit(X_tr, y_train[inner_tr])

        proba = clf.predict_proba(X_val)
        classes = list(clf.classes_)

        if metric == 'auc':
            if proba.shape[1] == 2:
                fold_score = roc_auc_score(y_train[inner_val], proba[:, 1])
            else:
                fold_score = roc_auc_score(y_train[inner_val], proba,
                                           multi_class='ovr')
        elif metric == 'macro':
            preds = clf.predict(X_val)
            fold_score = normalize_score(
                balanced_accuracy_score(y_train[inner_val], preds), C)
        else:
            preds = clf.predict(X_val)
            fold_score = accuracy_score(y_train[inner_val], preds)

        scores.append(fold_score)

        for i, idx in enumerate(inner_val):
            lbl = y_train[idx]
            if int(lbl) in classes:
                node_scores[idx] = proba[i, classes.index(int(lbl))]

    return float(np.mean(scores)), node_scores



# =============================================================
# INPUT VALIDATION AND SPARSE GRAPH CHECKS
# =============================================================

def validate_and_warn(edge_index: np.ndarray,
                      X: np.ndarray,
                      y: np.ndarray,
                      train_mask: np.ndarray,
                      adj: List[List[int]],
                      K: int = 3) -> None:
    """Validate inputs and emit warnings for common issues."""
    num_nodes = X.shape[0]

    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if len(y) != num_nodes or len(train_mask) != num_nodes:
        raise ValueError("y and train_mask must have length N")

    n_train = int(train_mask.sum())
    if n_train < 20:
        warnings.warn(f"Only {n_train} training nodes — results may be unreliable.")

    # Class imbalance check
    _, counts = np.unique(y[train_mask], return_counts=True)
    if counts.min() < 3:
        warnings.warn("Some classes have <3 training samples. CV may be unstable.")

    # Isolated nodes
    n_iso = sum(1 for v in range(num_nodes) if len(adj[v]) == 0)
    if n_iso > 0:
        warnings.warn(f"{n_iso} isolated nodes. FSS/TSS zero-padded for these.")

    # Sparse annulus warning (check k=2,3 on a sample of train nodes)
    train_idx = np.where(train_mask)[0]
    sample = train_idx[:min(200, len(train_idx))]
    for k in [2, 3]:
        n_empty = 0
        for v in sample:
            # Quick BFS to depth k
            seen = {int(v)}
            frontier = {int(v)}
            for _ in range(k):
                next_f = set()
                for u in frontier:
                    next_f.update(w for w in adj[u] if w not in seen)
                seen.update(next_f)
                frontier = next_f
            if len(frontier) == 0:
                n_empty += 1
        frac = n_empty / max(len(sample), 1)
        if frac > 0.20:
            warnings.warn(
                f"TSS/LSS at hop k={k} unreliable: {frac*100:.0f}% of sampled "
                f"training nodes have empty annulus. Consider reducing K or "
                f"interpreting scores with caution. (Common on Actor and sparse benchmarks.)"
            )


# =============================================================
# MAIN DIAGNOSTIC CLASS
# =============================================================

class NodeDiagnostic:
    """
    Computes FSS, TSS, LSS alignment scores for a node classification dataset.

    All scores are normalized balanced accuracy: (macro_acc - 1/C) / (1 - 1/C).
    Maps chance (1/C) to 0, perfect accuracy to 1.
    Directly comparable to baseline model scores on the same dataset.
    Higher = more label information accessible from that signal source.

    Naming:
      FSS: Feature Signal Score   (FLA in older versions)
      TSS: Topology Signal Score  (SLA in older versions)
      LSS: Label Smoothness Score (HLA in older versions)

    Usage:
        diag = NodeDiagnostic()
        results = diag.fit(edge_index, X, y, train_mask)
        print(results['FSS'], results['TSS'], results['LSS'])
    """

    def __init__(self,
                 r: int = 32,
                 T: int = 4,
                 K: int = 3,
                 m: Tuple[int, int, int] = (64, 64, 64),
                 n_splits: int = 5,
                 n_seeds: int = 3,
                 cv_seed: int = 0,
                 metric: str = 'macro'):
        self.r = r
        self.T = T
        self.K = K
        self.m = m
        self.n_splits = n_splits
        self.n_seeds = n_seeds
        self.cv_seed = cv_seed
        self.metric = metric

    def fit(self,
            edge_index: np.ndarray,
            X: np.ndarray,
            y: np.ndarray,
            train_mask: np.ndarray,
            verbose: bool = True) -> Dict:

        num_nodes = X.shape[0]
        C = int(y.max()) + 1

        def log(msg):
            if verbose:
                print(msg)

        # ---- Graph structures ----
        log("  [1/5] Building graph...")
        adj = build_adjacency_list(edge_index, num_nodes)
        A   = build_sparse_adj(edge_index, num_nodes)
        validate_and_warn(edge_index, X, y, train_mask, adj, self.K)

        # ---- Global precomputation ----
        log("  [2/5] Precomputing structural statistics...")
        degrees  = compute_degrees(A)
        ego_dens = compute_ego_density(A, degrees)
        coreness = compute_coreness(adj, num_nodes)
        wl       = compute_wl_colors(adj, num_nodes, degrees, self.T)
        r_eff    = min(self.r, X.shape[1])
        Xr       = random_projection(X, r_eff, seed=self.cv_seed)

        # ---- Descriptor matrices (averaged over sampling seeds) ----
        log("  [3/5] Computing descriptors...")
        train_idx = np.where(train_mask)[0]
        y_train   = y[train_idx]
        n_train   = len(train_idx)

        fss_dim = 4 * (2 * r_eff + 1)
        tss_dim = 30
        lss_dim = 3 * (C + 2)


        seed_accs_F, seed_accs_S, seed_accs_H = [], [], []
        seed_scores_F = np.zeros((self.n_seeds, n_train), dtype=np.float32)
        seed_scores_S = np.zeros((self.n_seeds, n_train), dtype=np.float32)
        seed_scores_H = np.zeros((self.n_seeds, n_train), dtype=np.float32)

        for s_idx in range(self.n_seeds):
            samp_seed = s_idx + 1   # sampling seeds: 1, 2, 3 (never 0 = cv_seed)

            # One BFS per training node, shared by all three channels.
            frontiers_cache = build_frontiers_cache(
                train_idx, adj, self.K, self.m, samp_seed)

            # FSS/TSS descriptors are label-independent: build once per seed.
            X_F_s = np.zeros((n_train, fss_dim), dtype=np.float32)
            X_S_s = np.zeros((n_train, tss_dim), dtype=np.float32)
            for i, v in enumerate(train_idx):
                fr = frontiers_cache[i]
                X_F_s[i] = fss_from_frontiers(int(v), fr, Xr)
                X_S_s[i] = tss_from_frontiers(int(v), fr, degrees, ego_dens,
                                              coreness, wl, self.T)

            log(f"  [3/5] Running probe (seed {s_idx+1}/{self.n_seeds})...")
            acc_F, sc_F = run_probe(X_F_s, y_train, self.n_splits, self.cv_seed, self.metric)
            acc_S, sc_S = run_probe(X_S_s, y_train, self.n_splits, self.cv_seed, self.metric)
            # LSS is label-dependent, so its design matrix is rebuilt inside
            # each CV fold from that fold's inner-training labels only.
            acc_H, sc_H = run_probe_lss(train_idx, frontiers_cache, y, C,
                                        self.n_splits, self.cv_seed, self.metric)

            seed_accs_F.append(acc_F)
            seed_accs_S.append(acc_S)
            seed_accs_H.append(acc_H)
            seed_scores_F[s_idx] = sc_F
            seed_scores_S[s_idx] = sc_S
            seed_scores_H[s_idx] = sc_H

        # Aggregate
        acc_F = float(np.mean(seed_accs_F))
        acc_S = float(np.mean(seed_accs_S))
        acc_H = float(np.mean(seed_accs_H))
        std_F = float(np.std(seed_accs_F))
        std_S = float(np.std(seed_accs_S))
        std_H = float(np.std(seed_accs_H))

        # Per-node scores: average over seeds (out-of-sample CV predictions)
        # Raw predicted probability of the true class -- no normalization applied.
        node_scores_F = seed_scores_F.mean(axis=0)
        node_scores_S = seed_scores_S.mean(axis=0)
        node_scores_H = seed_scores_H.mean(axis=0)

        # Dataset-level scores: normalized balanced accuracy
        # (normalization already applied inside run_probe for metric='macro')
        FSS = acc_F
        TSS = acc_S
        LSS = acc_H

        # Expand per-node scores to full node array (zeros for non-train)
        full_FSS = np.zeros(num_nodes, dtype=np.float32)
        full_TSS = np.zeros(num_nodes, dtype=np.float32)
        full_LSS = np.zeros(num_nodes, dtype=np.float32)
        full_FSS[train_idx] = node_scores_F
        full_TSS[train_idx] = node_scores_S
        full_LSS[train_idx] = node_scores_H

        log("  [4/5] Classifying node regimes...")
        regimes = self._classify_regimes(
            full_FSS, full_TSS, full_LSS, train_mask, C)

        return {
            # Dataset-level scores (normalized balanced accuracy)
            'FSS': FSS,
            'TSS': TSS,
            'LSS': LSS,
            # Normalized scores and standard deviations (over sampling seeds)
            'FSS_raw': acc_F, 'TSS_raw': acc_S, 'LSS_raw': acc_H,
            'FSS_std': std_F, 'TSS_std': std_S, 'LSS_std': std_H,
            # Per-node scores (raw predicted probability of true class, out-of-sample)
            'FSS_per_node': full_FSS,
            'TSS_per_node': full_TSS,
            'LSS_per_node': full_LSS,
            # Regime labels
            'regimes': regimes,
            'regime_counts': Counter(regimes[train_mask].tolist()),
            # Precomputed structures (for reuse)
            '_adj': adj,
            '_A': A,
            '_degrees': degrees,
            '_ego_dens': ego_dens,
            '_coreness': coreness,
            '_wl': wl,
            '_Xr': Xr,
        }

    def _classify_regimes(self,
                           fss: np.ndarray,
                           tss: np.ndarray,
                           lss: np.ndarray,
                           mask: np.ndarray,
                           C: int,
                           margin: float = 0.1) -> np.ndarray:
        # Per-node scores are raw predicted probabilities (chance = 1/C).
        # Threshold = 1/C + margin so that only nodes with signal above
        # chance are classified into a regime.
        threshold = 1.0 / C + margin
        regimes = np.full(len(fss), 'C', dtype=object)
        f = fss > threshold
        s = tss > threshold
        h = lss > threshold
        for v in np.where(mask)[0]:
            fv, sv, hv = f[v], s[v], h[v]
            if   fv and sv and hv: regimes[v] = 'FSH'
            elif fv and sv:        regimes[v] = 'FS'
            elif fv and hv:        regimes[v] = 'FH'
            elif sv and hv:        regimes[v] = 'SH'
            elif fv:               regimes[v] = 'F'
            elif sv:               regimes[v] = 'S'
            elif hv:               regimes[v] = 'H'
            else:                  regimes[v] = 'C'
        return regimes
