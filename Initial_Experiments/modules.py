import os
import random
import numpy as np
import torch

from torch_geometric.utils import k_hop_subgraph

import random
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
def compute_ego_mean_features(data, num_hops=2, device='cpu'):
    """
    For each node i:
      - extract num_hops ego graph around i
      - mean pool the original node features in that ego graph
    Returns:
      ego_x: [num_nodes, num_features]
    """
    edge_index = data.edge_index.to(device)
    x = data.x.to(device)
    num_nodes = data.num_nodes

    ego_reps = []

    for node_idx in range(num_nodes):
        subset, _, _, _ = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=num_hops,
            edge_index=edge_index,
            relabel_nodes=False
        )
        ego_feat = x[subset].mean(dim=0)
        ego_reps.append(ego_feat)

    ego_x = torch.stack(ego_reps, dim=0)
    return ego_x

def degree_histogram_features(degrees, max_bin=10):
    """
    Fixed-length degree histogram:
      bins for degree 0,1,2,...,max_bin-2, and one final bin for >= max_bin-1
    Example max_bin=10 -> bins: 0,1,2,3,4,5,6,7,8,>=9
    """
    hist = np.zeros(max_bin, dtype=np.float32)
    for d in degrees:
        if d >= max_bin - 1:
            hist[max_bin - 1] += 1.0
        else:
            hist[d] += 1.0

    if len(degrees) > 0:
        hist /= len(degrees)
    return hist


def extract_ego_structural_features(G, center_node, radius=2, degree_bins=10):
    """
    Compute simple structural features from the radius-hop ego graph around center_node.
    Returns a 1D numpy array.
    """
    ego = nx.ego_graph(G, center_node, radius=radius, undirected=True)

    num_nodes = ego.number_of_nodes()
    num_edges = ego.number_of_edges()

    degrees_dict = dict(ego.degree())
    degrees = np.array(list(degrees_dict.values()), dtype=np.float32)

    center_degree = float(degrees_dict[center_node])

    density = nx.density(ego) if num_nodes > 1 else 0.0

    avg_degree = float(degrees.mean()) if num_nodes > 0 else 0.0
    max_degree = float(degrees.max()) if num_nodes > 0 else 0.0
    min_degree = float(degrees.min()) if num_nodes > 0 else 0.0
    std_degree = float(degrees.std()) if num_nodes > 0 else 0.0

    clustering_dict = nx.clustering(ego)
    center_clustering = float(clustering_dict.get(center_node, 0.0))
    avg_clustering = float(np.mean(list(clustering_dict.values()))) if num_nodes > 0 else 0.0

    transitivity = float(nx.transitivity(ego)) if num_nodes > 2 else 0.0

    triangles_dict = nx.triangles(ego)
    num_triangles = float(sum(triangles_dict.values()) / 3.0)

    num_components = float(nx.number_connected_components(ego))

    # distance counts from center: how many nodes at distance 0,1,2
    sp = nx.single_source_shortest_path_length(ego, center_node, cutoff=radius)
    dist_counts = np.zeros(radius + 1, dtype=np.float32)
    for _, d in sp.items():
        dist_counts[d] += 1.0
    dist_counts /= max(1, num_nodes)

    # optional radius/diameter on largest connected component
    # ego graph around center is connected to center, but keep this robust
    if num_nodes > 1:
        largest_cc_nodes = max(nx.connected_components(ego), key=len)
        ego_lcc = ego.subgraph(largest_cc_nodes).copy()
        if ego_lcc.number_of_nodes() > 1:
            ego_radius = float(nx.radius(ego_lcc))
            ego_diameter = float(nx.diameter(ego_lcc))
        else:
            ego_radius = 0.0
            ego_diameter = 0.0
    else:
        ego_radius = 0.0
        ego_diameter = 0.0

    deg_hist = degree_histogram_features(degrees.astype(int), max_bin=degree_bins)

    # scalar features
    scalar_features = np.array([
        num_nodes,
        num_edges,
        density,
        avg_degree,
        max_degree,
        min_degree,
        std_degree,
        center_degree,
        center_clustering,
        avg_clustering,
        transitivity,
        num_triangles,
        num_components,
        ego_radius,
        ego_diameter,
    ], dtype=np.float32)

    features = np.concatenate([scalar_features, dist_counts, deg_hist], axis=0)
    return features


def compute_all_structural_features(data, radius=2, degree_bins=10):
    """
    Precompute structural ego-graph features for all nodes.
    """
    # Convert PyG graph to undirected NetworkX graph
    G = to_networkx(data, to_undirected=True, remove_self_loops=True)

    feats = []
    for node in range(data.num_nodes):
        f = extract_ego_structural_features(G, node, radius=radius, degree_bins=degree_bins)
        feats.append(f)

    feats = np.stack(feats, axis=0)
    return torch.tensor(feats, dtype=torch.float)

import csv

def get_final_mean_results(logger):
    """
    Returns the same final aggregated results that logger.print_statistics()
    prints, but as numbers.
    """
    result = 100 * torch.tensor(logger.results)

    best_results = []
    for r in result:
        train1 = r[:, 0].max().item()
        valid = r[:, 1].max().item()
        train2 = r[r[:, 1].argmax(), 0].item()
        test = r[r[:, 1].argmax(), 2].item()
        best_results.append((train1, valid, train2, test))

    best_result = torch.tensor(best_results)

    final_stats = {
        'highest_train_mean': best_result[:, 0].mean().item(),
        'highest_valid_mean': best_result[:, 1].mean().item(),
        'final_train_mean': best_result[:, 2].mean().item(),
        'final_test_mean': best_result[:, 3].mean().item(),
    }
    return final_stats


def save_final_mean_result(filepath, dataset_name, score_type, final_test_mean):
    """
    Save only the final mean test accuracy to CSV.
    """
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(['dataset', 'score_type', 'mean_accuracy'])

        writer.writerow([dataset_name, score_type, final_test_mean])