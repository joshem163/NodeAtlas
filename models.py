"""
Baseline Models for Node Signal Atlas (NSA)
============================================
Nine baseline model definitions, training utilities, and high-level API.

Models (4 families):
  F:  MLP
  L:  LP, APPNP
  T:  GCN, SAGE, H2GCN, SGC
  GT: GPS, SGFormer

NSA paper settings (Section 4.1 / Appendix A.3):
    2-layer networks, hidden width 64
    Grid search: lr in {0.01, 0.005}, wd in {0, 5e-4, 5e-3}, dropout in {0.3, 0.5}
    Adam, 500 epochs, patience 50
    SGFormer: 3 layers (1 trans + 2 GNN), 4 attn heads, graph_weight=0.5
    GPS: 2 layers, RWSE dim 16, 4 attn heads
    SGC: dropout=0.0 (linear model, no dropout grid)
    Stratified 60/20/20 split, 10 seeds
    Model evaluation: micro-accuracy
"""

import copy
import itertools
import random
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GPSConv, SAGEConv, SGConv
from torch_geometric.nn import APPNP as APPNP_Prop
from torch_geometric.nn.models import LabelPropagation, SGFormer
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

EPS = 1e-8


def normalize_score(score, C):
    """Normalize: (score - 1/C) / (1 - 1/C). Maps chance→0, perfect→1."""
    chance = 1.0 / C
    denom = 1.0 - chance
    if denom < EPS:
        return 0.0
    return (score - chance) / denom


# ================================================================== #
#  Constants / Hyperparameter grids
# ================================================================== #
HIDDEN = 64
MAX_EPOCHS = 500
PATIENCE = 50
LP_ITERS = 50

GPS_LAYERS = 2
GPS_HEADS = 4
GPS_RWSE_DIM = 16

SGFORMER_TRANS_LAYERS = 1
SGFORMER_GNN_LAYERS = 2
SGFORMER_HEADS = 4
SGFORMER_GRAPH_WT = 0.5

H2GCN_K = 2

APPNP_K = 10
APPNP_ALPHA = 0.1

SGC_K = 2

LR_GRID = [0.01, 0.005]
WD_GRID = [0, 5e-4, 5e-3]
DROPOUT_GRID = [0.3, 0.5]
LP_ALPHA_GRID = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

MODEL_NAMES = ["MLP", "LP", "GCN", "H2GCN", "GPS", "SGFormer",
               "APPNP", "SAGE", "SGC"]

# ================================================================== #
#  Seed & Split
# ================================================================== #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_split(y, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Class-wise stratified train/val/test split. Returns index tensors."""
    set_seed(seed)
    tr, va, te = [], [], []
    for c in torch.unique(y):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))]
        n = idx.size(0)
        nt = int(n * train_ratio)
        nv = int(n * val_ratio)
        tr.append(idx[:nt])
        va.append(idx[nt:nt + nv])
        te.append(idx[nt + nv:])
    tr = torch.cat(tr)[torch.randperm(sum(t.size(0) for t in tr))]
    va = torch.cat(va)[torch.randperm(sum(v.size(0) for v in va))]
    te = torch.cat(te)[torch.randperm(sum(t.size(0) for t in te))]
    return tr, va, te


# ================================================================== #
#  Data helpers
# ================================================================== #
def numpy_to_pyg(edge_index: np.ndarray,
                 X: np.ndarray,
                 y: np.ndarray) -> Data:
    """Convert numpy arrays to a PyG Data object."""
    return Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
    )


def row_normalize_features(x: torch.Tensor) -> torch.Tensor:
    """D^{-1} X row normalization."""
    rowsum = x.sum(dim=1, keepdim=True)
    rowsum = torch.where(rowsum == 0, torch.ones_like(rowsum), rowsum)
    return x / rowsum


def build_adj(edge_index, N):
    r, c = edge_index[0].numpy(), edge_index[1].numpy()
    A = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)),
                      shape=(N, N))
    A.setdiag(0)
    A.eliminate_zeros()
    return A


# ================================================================== #
#  Positional Encoding (RWSE) for GPS
# ================================================================== #
def compute_rwse(adj_scipy, N, walk_length=16, max_nnz=30_000_000):
    """Diagonal of A_rw^k for k=1..walk_length (RWSE for GPS)."""
    d = np.array(adj_scipy.sum(1)).flatten()
    di = np.where(d > 0, 1.0 / d, 0.0)
    A = sp.diags(di) @ adj_scipy
    A = A.tocsc()
    rwse = np.zeros((N, walk_length), dtype=np.float32)
    cur = sp.eye(N, format="csc")
    for k in range(walk_length):
        cur = A @ cur
        rwse[:, k] = np.asarray(cur.diagonal()).flatten()
        if cur.nnz > max_nnz:
            cur = cur.tocsr()
            cur.data[np.abs(cur.data) < 1e-5] = 0
            cur.eliminate_zeros()
            cur = cur.tocsc()
            if cur.nnz > max_nnz:
                break
    return torch.tensor(rwse, dtype=torch.float32)


# ================================================================== #
#  H2GCN adjacency preprocessing
# ================================================================== #
def h2gcn_precompute_adj(edge_index, N, device="cpu"):
    """Precompute normalized 1-hop and 2-hop adjacency for H2GCN."""
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    vals = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((vals, (row, col)), shape=(N, N))
    A.setdiag(0)
    A.eliminate_zeros()

    AI = A + sp.eye(N)
    mt0 = sp.eye(N, format="csr")
    mt1 = (AI @ mt0)
    mt1 = (mt1 > 0).astype(np.float32)
    mt2 = (AI @ mt1)
    mt2 = (mt2 > 0).astype(np.float32)

    A1_sp = (mt1 - mt0).tocsr()
    A1_sp.eliminate_zeros()
    A2_sp = (mt2 - mt1).tocsr()
    A2_sp.eliminate_zeros()

    def sym_norm_sparse(M):
        d = np.array(M.sum(1)).flatten()
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        M_norm = D_inv_sqrt @ M @ D_inv_sqrt
        M_norm = M_norm.tocoo()
        indices = torch.tensor(
            np.vstack([M_norm.row, M_norm.col]), dtype=torch.long)
        values = torch.tensor(M_norm.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices, values, (N, N)).coalesce().to(device)

    return sym_norm_sparse(A1_sp), sym_norm_sparse(A2_sp)


# ================================================================== #
#  Model definitions
# ================================================================== #

class MLP2(nn.Module):
    """2-layer MLP baseline."""
    def __init__(self, in_ch, hid, out_ch, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_ch, hid)
        self.lin2 = nn.Linear(hid, out_ch)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, **kw):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x).log_softmax(dim=-1)


class GCN2(nn.Module):
    """2-layer GCN (Kipf & Welling 2017, via PyG GCNConv)."""
    def __init__(self, in_ch, hid, out_ch, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid, cached=False)
        self.conv2 = GCNConv(hid, out_ch, cached=False)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index=None, **kw):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index).log_softmax(dim=-1)


class H2GCN(nn.Module):
    """H2GCN-2 (Zhu et al., NeurIPS 2020). Faithful to official TF DSL."""
    def __init__(self, feat_dim, hidden_dim, class_dim, k=2, dropout=0.5):
        super().__init__()
        self.k = k
        self.dropout = dropout
        self.w_embed = nn.Parameter(torch.empty(feat_dim, hidden_dim))
        concat_dim = hidden_dim * (2 ** (k + 1) - 1)
        self.w_classify = nn.Parameter(torch.empty(concat_dim, class_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w_classify)

    def forward(self, x, a1=None, a2=None, **kw):
        r0 = F.relu(torch.mm(x, self.w_embed))
        r1 = torch.cat([torch.spmm(a1, r0), torch.spmm(a2, r0)], dim=1)
        r2 = torch.cat([torch.spmm(a1, r1), torch.spmm(a2, r1)], dim=1)
        r_final = torch.cat([r2, r0, r1], dim=1)
        r_final = F.dropout(r_final, p=self.dropout, training=self.training)
        return torch.mm(r_final, self.w_classify).log_softmax(dim=1)


class GPSModel(nn.Module):
    """GPS graph transformer (Rampášek et al., NeurIPS 2022, via PyG GPSConv)."""
    def __init__(self, in_ch, hid, out_ch, rwse_dim=16, num_layers=2,
                 heads=4, dropout=0.5):
        super().__init__()
        self.feat_enc = nn.Linear(in_ch, hid)
        self.pe_enc = nn.Linear(rwse_dim, hid)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            local_conv = GCNConv(hid, hid, cached=False)
            self.layers.append(GPSConv(
                channels=hid, conv=local_conv, heads=heads,
                dropout=dropout, norm='layer_norm'))
        self.head = nn.Linear(hid, out_ch)
        self.dropout = dropout

    def reset_parameters(self):
        self.feat_enc.reset_parameters()
        self.pe_enc.reset_parameters()
        self.head.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index=None, rwse=None, **kw):
        h = self.feat_enc(x) + self.pe_enc(rwse)
        for layer in self.layers:
            h = layer(h, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).log_softmax(dim=-1)


class SGFormerWrapper(nn.Module):
    """SGFormer (Wu et al., NeurIPS 2023, via PyG SGFormer)."""
    def __init__(self, in_ch, hid, out_ch, trans_num_layers=1,
                 trans_num_heads=4, gnn_num_layers=2,
                 graph_weight=0.5, dropout=0.5):
        super().__init__()
        self.model = SGFormer(
            in_channels=in_ch, hidden_channels=hid, out_channels=out_ch,
            trans_num_layers=trans_num_layers, trans_num_heads=trans_num_heads,
            trans_dropout=dropout, gnn_num_layers=gnn_num_layers,
            gnn_dropout=dropout, graph_weight=graph_weight, aggregate='add')
        # Replace plain float with nn.Parameter so autograd tracks it
        self.model.graph_weight = nn.Parameter(torch.tensor(float(graph_weight)))

    def reset_parameters(self):
        self.model.reset_parameters()

    def forward(self, x, edge_index=None, **kw):
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.model(x, edge_index, batch=batch)


class APPNPNet(nn.Module):
    """APPNP (Klicpera et al., ICLR 2019, via PyG APPNP)."""
    def __init__(self, in_ch, hid, out_ch, K=10, alpha=0.1, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_ch, hid)
        self.lin2 = nn.Linear(hid, out_ch)
        self.prop = APPNP_Prop(K=K, alpha=alpha)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index=None, **kw):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return x.log_softmax(dim=-1)


class GraphSAGE(nn.Module):
    """GraphSAGE (Hamilton et al., NeurIPS 2017, via PyG SAGEConv)."""
    def __init__(self, in_ch, hid, out_ch, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hid)
        self.conv2 = SAGEConv(hid, out_ch)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index=None, **kw):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index).log_softmax(dim=-1)


class SGCNet(nn.Module):
    """SGC (Wu et al., ICML 2019, via PyG SGConv)."""
    def __init__(self, in_ch, hid, out_ch, K=2, dropout=0.0):
        super().__init__()
        self.conv = SGConv(in_ch, out_ch, K=K)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index=None, **kw):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv(x, edge_index).log_softmax(dim=-1)


# ================================================================== #
#  Metric
# ================================================================== #
def compute_metric(y_true, logits, metric="macro"):
    """Compute normalized balanced accuracy or ROC-AUC from logit array.

    macro: balanced_accuracy → (score - 1/C) / (1 - 1/C)
    auc:   ROC-AUC (no normalization)
    acc:   micro-accuracy (legacy, no normalization)
    """
    if metric == "auc":
        probs = np.exp(logits) if logits.min() < 0 else logits
        row_s = probs.sum(1, keepdims=True)
        probs = probs / np.where(row_s > 0, row_s, 1.0)
        if probs.shape[1] == 2:
            return roc_auc_score(y_true, probs[:, 1])
        return roc_auc_score(y_true, probs, multi_class="ovr")
    C = logits.shape[1]
    preds = logits.argmax(1)
    if metric == "macro":
        return normalize_score(balanced_accuracy_score(y_true, preds), C)
    return float((preds == y_true).sum()) / len(y_true)


# ================================================================== #
#  Factory
# ================================================================== #
def make_model(name: str, in_ch: int, num_cls: int,
               precomp: dict = None, hidden: int = HIDDEN,
               dropout: float = 0.5) -> nn.Module:
    """Create a model instance by name."""
    precomp = precomp or {}
    if name == "MLP":
        return MLP2(in_ch, hidden, num_cls, dropout=dropout)
    if name == "GCN":
        return GCN2(in_ch, hidden, num_cls, dropout=dropout)
    if name == "H2GCN":
        return H2GCN(in_ch, hidden, num_cls, k=H2GCN_K, dropout=dropout)
    if name == "GPS":
        rwse_dim = precomp["rwse"].size(1) if "rwse" in precomp else GPS_RWSE_DIM
        return GPSModel(in_ch, hidden, num_cls, rwse_dim=rwse_dim,
                        num_layers=GPS_LAYERS, heads=GPS_HEADS, dropout=dropout)
    if name == "SGFormer":
        return SGFormerWrapper(
            in_ch, hidden, num_cls,
            trans_num_layers=SGFORMER_TRANS_LAYERS,
            trans_num_heads=SGFORMER_HEADS,
            gnn_num_layers=SGFORMER_GNN_LAYERS,
            graph_weight=SGFORMER_GRAPH_WT,
            dropout=dropout)
    if name == "APPNP":
        return APPNPNet(in_ch, hidden, num_cls,
                        K=APPNP_K, alpha=APPNP_ALPHA, dropout=dropout)
    if name == "SAGE":
        return GraphSAGE(in_ch, hidden, num_cls, dropout=dropout)
    if name == "SGC":
        return SGCNet(in_ch, hidden, num_cls, K=SGC_K)
    raise ValueError(f"Unknown model: {name}")


# ================================================================== #
#  Single training run
# ================================================================== #
def train_model(model, mname, data, tr, va, te, precomp, metric, device,
                lr=0.01, wd=5e-4, max_ep=MAX_EPOCHS, patience=PATIENCE):
    """Train model with early stopping. Returns (val_score, test_score)."""
    model = model.to(device)
    y = data.y.to(device)
    ei = data.edge_index.to(device)
    tr_d, va_d, te_d = tr.to(device), va.to(device), te.to(device)

    if mname in ("GCN", "H2GCN", "SAGE"):
        x = precomp["x_normed"].to(device)
    else:
        x = data.x.to(device)

    extra = {}
    if mname == "GPS":
        extra["rwse"] = precomp["rwse"].to(device)
    if mname == "H2GCN":
        extra["a1"] = precomp["a1"]
        extra["a2"] = precomp["a2"]

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_st, pat = -1e9, None, 0

    def _forward():
        if mname == "MLP":
            return model(x)
        if mname in ("GCN", "SGFormer", "APPNP", "SAGE", "SGC"):
            return model(x, edge_index=ei)
        if mname == "GPS":
            return model(x, edge_index=ei, rwse=extra["rwse"])
        if mname == "H2GCN":
            return model(x, a1=extra["a1"], a2=extra["a2"])
        raise ValueError(mname)

    for _ in range(max_ep):
        model.train()
        opt.zero_grad()
        out = _forward()
        F.nll_loss(out[tr_d], y[tr_d]).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = _forward()
            vs = compute_metric(y[va_d].cpu().numpy(),
                                out[va_d].cpu().numpy(), metric)
        if vs > best_val:
            best_val = vs
            pat = 0
            best_st = copy.deepcopy(model.state_dict())
        else:
            pat += 1
            if pat >= patience:
                break

    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        out = _forward()
        ts = compute_metric(y[te_d].cpu().numpy(),
                            out[te_d].cpu().numpy(), metric)
    return best_val, ts


def evaluate(model, mname, data, te, precomp, metric, device):
    """Evaluate a trained model on the test split."""
    model.eval()
    model = model.to(device)
    x = (precomp.get("x_normed", data.x)
         if mname in ("GCN", "H2GCN", "SAGE") else data.x).to(device)
    ei = data.edge_index.to(device)
    te_d = te.to(device)

    with torch.no_grad():
        if mname == "MLP":
            out = model(x)
        elif mname in ("GCN", "SGFormer", "APPNP", "SAGE", "SGC"):
            out = model(x, edge_index=ei)
        elif mname == "GPS":
            out = model(x, edge_index=ei, rwse=precomp["rwse"].to(device))
        elif mname == "H2GCN":
            out = model(x, a1=precomp["a1"], a2=precomp["a2"])
        else:
            raise ValueError(mname)

    return compute_metric(data.y[te_d].cpu().numpy(),
                          out[te_d].cpu().numpy(), metric)


# ================================================================== #
#  Grid search — neural models
# ================================================================== #
def grid_search_neural(mname, data, tr, va, te, num_cls, precomp, metric,
                       device, hidden=HIDDEN):
    """Val-based HP grid search for neural models."""
    best_val, best_test, best_hp = -1e9, 0.0, {}
    in_ch = data.x.size(1)
    grid = list(itertools.product(LR_GRID, WD_GRID, DROPOUT_GRID))

    for lr, wd, dp in grid:
        m = make_model(mname, in_ch, num_cls, precomp, hidden, dropout=dp)
        m.reset_parameters()
        try:
            vs, ts = train_model(m, mname, data, tr, va, te,
                                 precomp, metric, device, lr=lr, wd=wd)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                continue
            raise
        if vs > best_val:
            best_val, best_test = vs, ts
            best_hp = {"lr": lr, "wd": wd, "dropout": dp}

    return best_test, best_hp


# ================================================================== #
#  Grid search — LP
# ================================================================== #
def grid_search_lp(data, tr, va, te, num_cls, metric):
    """Grid search over LP alpha using PyG LabelPropagation."""
    N = data.num_nodes
    y = data.y
    ei = data.edge_index

    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[tr] = True

    best_val, best_test, best_alpha = -1e9, 0.0, 0.5

    for alpha in LP_ALPHA_GRID:
        lp = LabelPropagation(num_layers=LP_ITERS, alpha=alpha)
        Yh = lp(y, ei, mask=train_mask)
        Yh_np = Yh.numpy()
        va_np = va.numpy()
        te_np = te.numpy()
        y_np = y.numpy()

        if metric == "auc":
            row_s = Yh_np.sum(1, keepdims=True)
            probs = Yh_np / np.where(row_s > 0, row_s, 1.0)
            if probs.shape[1] == 2:
                vs = roc_auc_score(y_np[va_np], probs[va_np, 1])
                ts = roc_auc_score(y_np[te_np], probs[te_np, 1])
            else:
                vs = roc_auc_score(y_np[va_np], probs[va_np],
                                   multi_class="ovr")
                ts = roc_auc_score(y_np[te_np], probs[te_np],
                                   multi_class="ovr")
        else:
            preds = Yh_np.argmax(1)
            if metric == "macro":
                vs = normalize_score(
                    balanced_accuracy_score(y_np[va_np], preds[va_np]), num_cls)
                ts = normalize_score(
                    balanced_accuracy_score(y_np[te_np], preds[te_np]), num_cls)
            else:
                vs = float((preds[va_np] == y_np[va_np]).sum()) / len(va_np)
                ts = float((preds[te_np] == y_np[te_np]).sum()) / len(te_np)

        if vs > best_val:
            best_val, best_test, best_alpha = vs, ts, alpha

    return best_test, {"alpha": best_alpha}


# ================================================================== #
#  Precompute shared structures for a dataset
# ================================================================== #
def precompute_dataset(data: Data, N: int,
                       device: str = 'cpu',
                       models: list = None) -> dict:
    """Precompute RWSE, H2GCN adj, and row-normalized features."""
    if models is None:
        models = MODEL_NAMES
    precomp = {}
    adj = build_adj(data.edge_index, N)

    if 'GPS' in models:
        precomp['rwse'] = compute_rwse(adj, N, walk_length=GPS_RWSE_DIM)

    if 'H2GCN' in models:
        precomp['a1'], precomp['a2'] = h2gcn_precompute_adj(
            data.edge_index, N, device=device)

    if any(m in models for m in ('GCN', 'H2GCN', 'SAGE')):
        precomp['x_normed'] = row_normalize_features(data.x)

    return precomp


# ================================================================== #
#  High-level runner: all models on one dataset / one seed
# ================================================================== #
def run_all_models(edge_index: np.ndarray,
                   X: np.ndarray,
                   y: np.ndarray,
                   train_mask: np.ndarray,
                   val_mask: np.ndarray,
                   test_mask: np.ndarray,
                   seed: int = 42,
                   device: str = 'cpu',
                   models: list = None,
                   metric: str = 'acc') -> dict:
    """
    Run all baseline models and return {model_name: test_score}.

    Args:
        edge_index:     (2, E) int64 numpy array
        X:              (N, d) float32 numpy array
        y:              (N,)   int numpy array
        train/val/test_mask: boolean numpy masks
        seed:           random seed for this run
        device:         torch device string
        models:         list of model names (None = all 9)
        metric:         'macro' (balanced acc), 'acc' (micro), or 'auc'
    """
    if models is None:
        models = MODEL_NAMES

    data = numpy_to_pyg(edge_index, X, y)
    N = X.shape[0]
    num_cls = int(y.max()) + 1

    tr = torch.where(torch.tensor(train_mask))[0]
    va = torch.where(torch.tensor(val_mask))[0]
    te = torch.where(torch.tensor(test_mask))[0]

    set_seed(seed)
    precomp = precompute_dataset(data, N, device=device, models=models)

    accs = {}
    for mname in models:
        try:
            if mname == 'LP':
                sc, _ = grid_search_lp(data, tr, va, te, num_cls, metric)

            else:
                sc, _ = grid_search_neural(
                    mname, data, tr, va, te, num_cls,
                    precomp, metric, device)

            accs[mname] = sc

        except Exception as e:
            warnings.warn(f"run_all_models: {mname} failed: {e}")
            accs[mname] = float('nan')

    return accs
