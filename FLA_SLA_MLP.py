import os
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph
from logger import *
from modules import *
from models import MLP
from data_loader import *


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Custom 60/20/20 stratified split
# =========================================================
def stratified_split(y, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Creates class-wise stratified train/val/test masks.
    """
    set_seed(seed)
    num_nodes = y.size(0)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    classes = torch.unique(y)
    for c in classes:
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))]

        n = idx.size(0)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    return train_mask, val_mask, test_mask

# =========================================================
# Train / Eval
# =========================================================
# def train(model, x, y, train_mask, optimizer):
#     model.train()
#     optimizer.zero_grad()
#     out = model(x)
#     loss = F.cross_entropy(out[train_mask], y[train_mask])
#     loss.backward()
#     optimizer.step()
#     return loss.item()
#
#
# @torch.no_grad()
# def evaluate(model, x, y, train_mask, val_mask, test_mask):
#     model.eval()
#     out = model(x)
#     pred = out.argmax(dim=-1)
#
#     train_acc = (pred[train_mask] == y[train_mask]).float().mean().item()
#     val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
#     test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()
#
#     return train_acc, val_acc, test_acc
def train(model, x,y, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(x)[train_idx]
    loss = F.nll_loss(out, y.squeeze()[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


def ACC(prediction, label):
    correct = prediction.eq(label).sum().item()
    total = len(label)
    return correct / total
@torch.no_grad()
def evaluate(model, x,y, train_idx, valid_idx, test_idx, metric='accuracy'):
    model.eval()
    out = model(x)  # raw logits
    y_true = y.squeeze()  # ensure shape consistency

    if metric == 'accuracy':
        y_pred = out.argmax(dim=-1)
        train_score = ACC(y_pred[train_idx], y_true[train_idx])
        valid_score = ACC(y_pred[valid_idx], y_true[valid_idx])
        test_score = ACC(y_pred[test_idx], y_true[test_idx])

    elif metric == 'roc_auc':
        # Assume binary classification and get probability of class 1
        probs = F.softmax(out, dim=-1)[:, 1]  # get prob for class 1

        train_score = roc_auc_score(y_true[train_idx].cpu(), probs[train_idx].cpu())
        valid_score = roc_auc_score(y_true[valid_idx].cpu(), probs[valid_idx].cpu())
        test_score = roc_auc_score(y_true[test_idx].cpu(), probs[test_idx].cpu())

    else:
        raise ValueError("Unsupported metric: choose 'accuracy' or 'roc_auc'")

    return train_score, valid_score, test_score



# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description='experiment')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--Score_type', type=str, default='SLA')
    parser.add_argument('--dataset_name', type=str, default='actor')
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()

    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load dataset
    #data_loaded = Planetoid(root='/tmp/cora', name='Cora', split='geom-gcn')
    # data_loaded = load_data(args.dataset_name)
    # data = data_loaded[0].to(device)
    if args.dataset_name in ['chameleon', 'squirrel']:
        data = load_Sq_Cha_filterred(args.dataset_name)
        num_classes=5
    else:
        dataset = load_data(args.dataset_name)
        data = dataset[0]
        num_classes = dataset.num_classes

    # -----------------------------------------------------
    # Precompute 2-hop ego mean representations
    # -----------------------------------------------------
    if args.Score_type== 'FLA':
        print("Computing 2-hop ego-graph mean representations...")
        ego_x = compute_ego_mean_features(data, num_hops=2, device=device)

        # Optional normalization
        ego_x = (ego_x - ego_x.mean(dim=0, keepdim=True)) / (ego_x.std(dim=0, keepdim=True) + 1e-8)
    elif args.Score_type== 'SLA':
        print("Precomputing 2-hop ego structural features...")
        ego_x = compute_all_structural_features(data, radius=2, degree_bins=10)
    else:
        print("Score is not defined")


    y = data.y
    num_features = ego_x.size(1)
    #num_classes = data_loaded.num_classes
    logger = Logger(args.runs)

    for run in range(args.runs):
        print(f"\n===== Run {run + 1} / {args.runs} =====")
        seed = run + 42
        set_seed(seed)

        # Create custom 60/20/20 split
        train_mask, val_mask, test_mask = stratified_split(
            y.cpu(), train_ratio=0.6, val_ratio=0.2, seed=seed
        )
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)

        model = MLP(
            in_channels=num_features,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            num_layers=2,
            dropout=args.dropout
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        for epoch in range(1, args.epochs + 1):
            loss = train(model, ego_x, y, train_mask, optimizer)
            result = evaluate(model, ego_x, y, train_mask, val_mask, test_mask, metric='accuracy')
            logger.add_result(run, result)

            if epoch % 20 == 0 or epoch == 1:
                train_acc, val_acc, test_acc = result
                print(
                    f"Epoch: {epoch:03d}, "
                    f"Loss: {loss:.4f}, "
                    f"Train: {100 * train_acc:.2f}%, "
                    f"Val: {100 * val_acc:.2f}%, "
                    f"Test: {100 * test_acc:.2f}%"
                )

        logger.print_statistics(run)

    print("\n================ Final Results ================")
    logger.print_statistics()
    final_stats = get_final_mean_results(logger)
    final_test_mean = final_stats['final_test_mean']/100

    print(f"\nSaved Mean Test Accuracy: {final_test_mean:.2f}")

    save_final_mean_result(
        filepath='SLA_results.csv',
        dataset_name=args.dataset_name,
        score_type=args.Score_type,
        final_test_mean=final_test_mean
    )


if __name__ == "__main__":
    main()