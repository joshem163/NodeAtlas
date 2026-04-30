import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import random
import warnings

from torch_geometric.datasets import Planetoid
from models import GCN
from logger import Logger
from modules import *
from data_loader import *
from torch_sparse import SparseTensor

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(y, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Creates class-wise stratified train/val/test index splits.
    Returns:
        train_idx, val_idx, test_idx
    """
    set_seed(seed)

    train_indices = []
    val_indices = []
    test_indices = []

    classes = torch.unique(y)

    for c in classes:
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))]

        n = idx.size(0)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_idx_c = idx[:n_train]
        val_idx_c = idx[n_train:n_train + n_val]
        test_idx_c = idx[n_train + n_val:]

        train_indices.append(train_idx_c)
        val_indices.append(val_idx_c)
        test_indices.append(test_idx_c)

    train_idx = torch.cat(train_indices, dim=0)
    val_idx = torch.cat(val_indices, dim=0)
    test_idx = torch.cat(test_indices, dim=0)

    # Shuffle each split so classes are mixed
    train_idx = train_idx[torch.randperm(train_idx.size(0))]
    val_idx = val_idx[torch.randperm(val_idx.size(0))]
    test_idx = test_idx[torch.randperm(test_idx.size(0))]

    return train_idx, val_idx, test_idx


def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.adj_t)
    loss = F.nll_loss(out[train_idx], data.y[train_idx])

    loss.backward()
    optimizer.step()

    return loss.item()


def ACC(prediction, label):
    correct = prediction.view(-1).eq(label.view(-1)).sum().item()
    total = label.numel()
    return correct / total


@torch.no_grad()
def test(model, data, train_idx, valid_idx, test_idx):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1)

    train_acc = ACC(y_pred[train_idx], data.y[train_idx])
    valid_acc = ACC(y_pred[valid_idx], data.y[valid_idx])
    test_acc = ACC(y_pred[test_idx], data.y[test_idx])

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='experiment')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--Score_type', type=str, default='GCN')
    parser.add_argument('--dataset_name', type=str, default='chameleon')
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

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = load_data(args.dataset_name)
    #
    # data = dataset[0].to(device)
    if args.dataset_name in ['chameleon', 'squirrel']:
        data = load_Sq_Cha_filterred(args.dataset_name)
        num_classes=5
    else:
        dataset = load_data(args.dataset_name)
        data = dataset[0]
        num_classes = dataset.num_classes

    row, col = data.edge_index
    data.adj_t = SparseTensor(row=row, col=col,
                              sparse_sizes=(data.num_nodes, data.num_nodes)).t()
    print(data)

    model = GCN(
        data.num_features,
        args.hidden_channels,
        num_classes,
        args.num_layers,
        args.dropout
    ).to(device)

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        seed = run + 42
        set_seed(seed)

        train_idx, valid_idx, test_idx = stratified_split(
            data.y.cpu(),
            train_ratio=0.6,
            val_ratio=0.2,
            seed=seed
        )

        train_idx = train_idx.to(device)
        valid_idx = valid_idx.to(device)
        test_idx = test_idx.to(device)

        model.reset_parameters()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        for epoch in range(1, args.epochs + 1):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, train_idx, valid_idx, test_idx)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(
                    f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:03d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}%, '
                    f'Test: {100 * test_acc:.2f}%'
                )

        logger.print_statistics(run)

    logger.print_statistics()
    final_stats = get_final_mean_results(logger)
    final_test_mean = final_stats['final_test_mean']/100

    print(f"\nSaved Mean Test Accuracy: {final_test_mean:.2f}")

    save_final_mean_result(
        filepath='GCN_results.csv',
        dataset_name=args.dataset_name,
        score_type=args.Score_type,
        final_test_mean=final_test_mean
    )


if __name__ == "__main__":
    main()