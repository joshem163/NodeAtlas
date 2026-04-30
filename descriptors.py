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

LSS label leakage prevention (Section 3.1, lines 161-164):
  LSS descriptors depend on observed labels and are recomputed inside
  each CV fold using only the inner training partition's labels.
  No held-out label ever contributes to the descriptor used to predict
  that same node.
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
    """Normalize: (score - 1/C) / (1 - 1/C). Maps chance->0, perfect->1."""
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
    Computed via Hadamard product: tri(v) = (A . A^2).sum(axis=1)[v] / 2.
    """
    A2 = A @ A
    edges_among_nbrs = np.asarray(A.multiply(A2).sum(axis=1)).flatten() / 2.0
    denom = degrees * (degrees - 1) + EPS
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
    """
    rng = np.random.RandomState(seed)
    F_full = [None] * (K + 1)
    F_samp = [None] * (K + 1)
    seen = {v}

    F_full[0] = np.array([v], dtype=np.int64)
    F_samp[0] = F_full[0]

    for k in range(1, K + 1):
        cand_set = set()
        for u in F_full[k - 1]:
            for w in adj[u]:
                if w not in seen:
                    cand_set.add(w)
        seen.update(cand_set)
        F_full[k] = np.array(sorted(cand_set), dtype=np.int64)

        mk = m[k - 1]
        if len(F_full[k]) > mk:
            idx = rng.choice(len(F_full[k]), mk, replace=False)
            F_samp[k] = F_full[k][idx]
        else:
            F_samp[k] = F_full[k]

    return F_samp


def build_frontiers_cache(train_idx: np.ndarray,
                          adj: List[List[int]],
                          K: int,
                          m: Tuple[int, int, int],
                          seed: int) -> List[List[np.ndarray]]:
    """Pre-compute BFS frontiers for all training nodes.

    Returns list of frontiers, one per training node (same order as train_idx).
    Frontiers are shared across FSS/TSS/LSS to avoid redundant BFS.
    """
    cache = []
    for v in train_idx:
        cache.append(annulus_frontiers(int(v), adj, K, m, seed))
    return cache


# =============================================================
# DESCRIPTOR FUNCTIONS  (all take pre-computed frontiers)
# =============================================================

def fss_from_frontiers(v: int,
                       frontiers: List[np.ndarray],
                       Xr: np.ndarray) -> np.ndarray:
    """
    FSS descriptor for node v.  R^{4*(2r+1)}
    phi_F(v, k) = [mean_k | std_k | shift_k]
    """
    r = Xr.shape[1]
    block = 2 * r + 1
    phi = np.zeros(4 * block, dtype=np.float32)

    xv = Xr[v]

    for k, S in enumerate(frontiers):
        offset = k * block
        if len(S) == 0:
            pass
        elif k == 0:
            phi[offset:offset + r] = xv
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
            continue
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
    """TSS descriptor for node v.  R^{30}"""
    phi = np.zeros(30, dtype=np.float32)
    K_actual = len(frontiers) - 1

    phi[0] = float(np.log(degrees[v] + 1))
    phi[1] = float(coreness[v])
    phi[2] = float(ego_dens[v])

    b = [1] + [0] * K_actual
    for k in range(1, K_actual + 1):
        b[k] = b[k - 1] + len(frontiers[k])

    offset = 3
    for k in range(1, 4):
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
                       label_mask: np.ndarray,
                       C: int) -> np.ndarray:
    """
    LSS descriptor for node v.  R^{3*(C+2)}

    label_mask: boolean mask indicating which nodes' labels are available.
    For proper CV: this should be the inner training partition only,
    NOT the full train_mask. This prevents label leakage (Section 3.1).
    """
    block = C + 2
    phi = np.zeros(3 * block, dtype=np.float32)
    uniform = np.ones(C, dtype=np.float32) / C

    for i, k in enumerate([1, 2, 3]):
        offset = i * block
        if k >= len(frontiers):
            phi[offset:offset + C] = uniform
            continue

        S = frontiers[k]
        nk = len(S)
        S_lab = S[label_mask[S]] if nk > 0 else np.array([], dtype=np.int64)
        mk = len(S_lab)

        if mk == 0:
            phi[offset:offset + C] = uniform
        else:
            p = np.zeros(C, dtype=np.float32)
            for u in S_lab:
                p[int(labels[u])] += 1
            p /= p.sum()
            phi[offset:offset + C] = p

        phi[offset + C]     = mk / (nk + 1)
        phi[offset + C + 1] = float(np.log(mk + 1))

    return phi


# =============================================================
# DESIGN MATRIX BUILDERS
# =============================================================

def build_fss_tss_matrices(train_idx: np.ndarray,
                           frontiers_cache: List[List[np.ndarray]],
                           Xr: np.ndarray,
                           degrees: np.ndarray,
                           ego_dens: np.ndarray,
                           coreness: np.ndarray,
                           wl: np.ndarray,
                           T: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build FSS and TSS design matrices (label-independent, built once)."""
    r = Xr.shape[1]
    n_train = len(train_idx)

    X_F = np.zeros((n_train, 4 * (2 * r + 1)), dtype=np.float32)
    X_S = np.zeros((n_train, 30), dtype=np.float32)

    for i, v in enumerate(train_idx):
        frontiers = frontiers_cache[i]
        X_F[i] = fss_from_frontiers(int(v), frontiers, Xr)
        X_S[i] = tss_from_frontiers(int(v), frontiers, degrees,
                                     ego_dens, coreness, wl, T)

    return X_F, X_S


def build_lss_matrix(train_idx: np.ndarray,
                     frontiers_cache: List[List[np.ndarray]],
                     labels: np.ndarray,
                     label_mask: np.ndarray,
                     C: int) -> np.ndarray:
    """Build LSS design matrix using only labels visible in label_mask.

    Args:
        label_mask: boolean mask over ALL nodes indicating which labels
                    are available for descriptor construction. For fold-level
                    leakage prevention, this should be the inner training
                    partition's mask (excluding the CV validation fold).
    """
    n_train = len(train_idx)
    X_H = np.zeros((n_train, 3 * (C + 2)), dtype=np.float32)

    for i, v in enumerate(train_idx):
        frontiers = frontiers_cache[i]
        X_H[i] = lss_from_frontiers(int(v), frontiers, labels,
                                     label_mask, C)

    return X_H


# =============================================================
# PROBES
# =============================================================

def run_probe(X: np.ndarray,
              y: np.ndarray,
              n_splits: int = 5,
              cv_seed: int = 0,
              metric: str = 'macro') -> Tuple[float, np.ndarray]:
    """
    L2 logistic regression probe with stratified K-fold CV.
    For FSS and TSS (label-independent descriptors).
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

        for i, (idx, lbl) in enumerate(zip(val_idx, y[val_idx])):
            if int(lbl) in classes:
                node_scores[idx] = proba[i, classes.index(int(lbl))]

    return float(np.mean(scores)), node_scores


def run_probe_lss(train_idx: np.ndarray,
                  frontiers_cache: List[List[np.ndarray]],
                  labels: np.ndarray,
                  train_mask: np.ndarray,
                  C: int,
                  n_splits: int = 5,
                  cv_seed: int = 0,
                  metric: str = 'macro') -> Tuple[float, np.ndarray]:
    """
    LSS probe with per-fold descriptor reconstruction (Section 3.1).

    For each CV fold, LSS descriptors are rebuilt using only the inner
    training partition's labels. The validation fold's labels are never
    used in descriptor construction, preventing label leakage.
    """
    y_train = labels[train_idx]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)

    scores = []
    node_scores = np.zeros(len(train_idx), dtype=np.float32)
    num_nodes = len(labels)

    for inner_tr_idx, inner_val_idx in skf.split(
            np.zeros(len(train_idx)), y_train):
        # Build label mask: only inner training nodes' labels are visible.
        # inner_tr_idx indexes into train_idx, not into the full node array.
        inner_label_mask = np.zeros(num_nodes, dtype=bool)
        inner_label_mask[train_idx[inner_tr_idx]] = True

        # Rebuild LSS descriptors for ALL train nodes using inner labels only
        X_H = build_lss_matrix(train_idx, frontiers_cache, labels,
                               inner_label_mask, C)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_H[inner_tr_idx])
        X_val = scaler.transform(X_H[inner_val_idx])

        clf = LogisticRegression(max_iter=1000, random_state=cv_seed,
                                 solver='lbfgs', C=1.0)
        clf.fit(X_tr, y_train[inner_tr_idx])

        proba = clf.predict_proba(X_val)
        classes = list(clf.classes_)

        if metric == 'auc':
            if proba.shape[1] == 2:
                fold_score = roc_auc_score(
                    y_train[inner_val_idx], proba[:, 1])
            else:
                fold_score = roc_auc_score(
                    y_train[inner_val_idx], proba, multi_class='ovr')
        elif metric == 'macro':
            preds = clf.predict(X_val)
            fold_score = normalize_score(
                balanced_accuracy_score(y_train[inner_val_idx], preds), C)
        else:
            preds = clf.predict(X_val)
            fold_score = accuracy_score(y_train[inner_val_idx], preds)

        scores.append(fold_score)

        for i, idx in enumerate(inner_val_idx):
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
        warnings.warn(f"Only {n_train} training nodes -- results may be unreliable.")

    _, counts = np.unique(y[train_mask], return_counts=True)
    if counts.min() < 3:
        warnings.warn("Some classes have <3 training samples. CV may be unstable.")

    n_iso = sum(1 for v in range(num_nodes) if len(adj[v]) == 0)
    if n_iso > 0:
        warnings.warn(f"{n_iso} isolated nodes. FSS/TSS zero-padded for these.")

    train_idx = np.where(train_mask)[0]
    sample = train_idx[:min(200, len(train_idx))]
    for k in [2, 3]:
        n_empty = 0
        for v in sample:
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
                f"training nodes have empty annulus.")


# =============================================================
# MAIN DIAGNOSTIC CLASS
# =============================================================

class NodeDiagnostic:
    """
    Computes FSS, TSS, LSS alignment scores for a node classification dataset.

    All scores are normalized balanced accuracy: (macro_acc - 1/C) / (1 - 1/C).

    LSS descriptors are rebuilt inside each CV fold to prevent label leakage
    (Section 3.1): validation fold labels never contribute to the LSS
    descriptor used to predict those same nodes.

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
        log("  [1/4] Building graph...")
        adj = build_adjacency_list(edge_index, num_nodes)
        A   = build_sparse_adj(edge_index, num_nodes)
        validate_and_warn(edge_index, X, y, train_mask, adj, self.K)

        # ---- Global precomputation ----
        log("  [2/4] Precomputing structural statistics...")
        degrees  = compute_degrees(A)
        ego_dens = compute_ego_density(A, degrees)
        coreness = compute_coreness(adj, num_nodes)
        wl       = compute_wl_colors(adj, num_nodes, degrees, self.T)
        r_eff    = min(self.r, X.shape[1])
        Xr       = random_projection(X, r_eff, seed=self.cv_seed)

        # ---- Descriptors and probes (averaged over sampling seeds) ----
        train_idx = np.where(train_mask)[0]
        y_train   = y[train_idx]
        n_train   = len(train_idx)

        seed_accs_F, seed_accs_S, seed_accs_H = [], [], []
        seed_scores_F = np.zeros((self.n_seeds, n_train), dtype=np.float32)
        seed_scores_S = np.zeros((self.n_seeds, n_train), dtype=np.float32)
        seed_scores_H = np.zeros((self.n_seeds, n_train), dtype=np.float32)

        for s_idx in range(self.n_seeds):
            samp_seed = s_idx + 1

            log(f"  [3/4] Seed {s_idx+1}/{self.n_seeds}: "
                f"building frontiers & descriptors...")

            # Cache BFS frontiers (shared across all three channels)
            frontiers_cache = build_frontiers_cache(
                train_idx, adj, self.K, self.m, samp_seed)

            # FSS/TSS: label-independent, built once per seed
            X_F, X_S = build_fss_tss_matrices(
                train_idx, frontiers_cache, Xr, degrees,
                ego_dens, coreness, wl, self.T)

            log(f"  [3/4] Seed {s_idx+1}/{self.n_seeds}: "
                f"running probes...")

            # FSS probe
            acc_F, sc_F = run_probe(
                X_F, y_train, self.n_splits, self.cv_seed, self.metric)
            # TSS probe
            acc_S, sc_S = run_probe(
                X_S, y_train, self.n_splits, self.cv_seed, self.metric)
            # LSS probe (per-fold descriptor reconstruction)
            acc_H, sc_H = run_probe_lss(
                train_idx, frontiers_cache, y, train_mask, C,
                self.n_splits, self.cv_seed, self.metric)

            seed_accs_F.append(acc_F)
            seed_accs_S.append(acc_S)
            seed_accs_H.append(acc_H)
            seed_scores_F[s_idx] = sc_F
            seed_scores_S[s_idx] = sc_S
            seed_scores_H[s_idx] = sc_H

        # Aggregate over seeds
        FSS = float(np.mean(seed_accs_F))
        TSS = float(np.mean(seed_accs_S))
        LSS = float(np.mean(seed_accs_H))
        std_F = float(np.std(seed_accs_F))
        std_S = float(np.std(seed_accs_S))
        std_H = float(np.std(seed_accs_H))

        node_scores_F = seed_scores_F.mean(axis=0)
        node_scores_S = seed_scores_S.mean(axis=0)
        node_scores_H = seed_scores_H.mean(axis=0)

        full_FSS = np.zeros(num_nodes, dtype=np.float32)
        full_TSS = np.zeros(num_nodes, dtype=np.float32)
        full_LSS = np.zeros(num_nodes, dtype=np.float32)
        full_FSS[train_idx] = node_scores_F
        full_TSS[train_idx] = node_scores_S
        full_LSS[train_idx] = node_scores_H

        log("  [4/4] Classifying node regimes...")
        regimes = self._classify_regimes(
            full_FSS, full_TSS, full_LSS, train_mask, C)

        return {
            'FSS': FSS, 'TSS': TSS, 'LSS': LSS,
            'FSS_std': std_F, 'TSS_std': std_S, 'LSS_std': std_H,
            'FSS_per_node': full_FSS,
            'TSS_per_node': full_TSS,
            'LSS_per_node': full_LSS,
            'regimes': regimes,
            'regime_counts': Counter(regimes[train_mask].tolist()),
            '_adj': adj, '_A': A,
            '_degrees': degrees, '_ego_dens': ego_dens,
            '_coreness': coreness, '_wl': wl, '_Xr': Xr,
        }

    def _classify_regimes(self,
                           fss: np.ndarray,
                           tss: np.ndarray,
                           lss: np.ndarray,
                           mask: np.ndarray,
                           C: int,
                           margin: float = 0.1) -> np.ndarray:
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
