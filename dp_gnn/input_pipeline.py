"""Input pipeline for DP-GNN training.

Loads a graph dataset, adds reverse edges, subsamples with degree bounds,
computes split masks, converts to PyG Data, adds self-loops, and normalizes.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from dp_gnn import dataset_readers
from dp_gnn import normalizations
from dp_gnn import sampler


def add_reverse_edges(graph: dataset_readers.Dataset) -> dataset_readers.Dataset:
    """Add reverse edges to the graph."""
    senders = np.concatenate([graph.senders, graph.receivers])
    receivers = np.concatenate([graph.receivers, graph.senders])
    graph.senders = senders
    graph.receivers = receivers
    return graph


def subsample_graph(
    graph: dataset_readers.Dataset,
    max_degree: int,
    rng: torch.Generator,
) -> dataset_readers.Dataset:
    """Subsamples the undirected graph with bounded in-degree."""
    edges = sampler.get_adjacency_lists(graph.senders, graph.receivers,
                                        graph.num_nodes())
    edges = sampler.sample_adjacency_lists(edges, graph.train_nodes,
                                           max_degree, rng)
    senders = []
    receivers = []
    for u in edges:
        for v in edges[u]:
            senders.append(u)
            receivers.append(v)
    graph.senders = np.array(senders, dtype=np.int64)
    graph.receivers = np.array(receivers, dtype=np.int64)
    return graph


def compute_masks_for_splits(
    graph: dataset_readers.Dataset,
) -> Dict[str, np.ndarray]:
    """Boolean masks for train/validation/test splits."""
    masks = {}
    num_nodes = graph.num_nodes()
    for split_name, split_nodes in [
        ('train', graph.train_nodes),
        ('validation', graph.validation_nodes),
        ('test', graph.test_nodes),
    ]:
        mask = np.zeros(num_nodes, dtype=bool)
        mask[split_nodes] = True
        masks[split_name] = mask
    return masks


def convert_to_pyg_data(
    graph: dataset_readers.Dataset,
) -> Tuple[Data, np.ndarray]:
    """Converts a Dataset to a PyG Data object + labels array."""
    edge_index = torch.tensor(
        np.stack([graph.senders, graph.receivers]), dtype=torch.long)
    x = torch.tensor(graph.node_features, dtype=torch.float32)
    num_edges = edge_index.size(1)
    edge_attr = torch.ones(num_edges, 1, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = graph.num_nodes()
    return data, np.asarray(graph.node_labels)


def add_self_loops(data: Data) -> Data:
    """Adds self-loops to the graph."""
    num_nodes = data.num_nodes
    self_loop_idx = torch.arange(num_nodes, dtype=torch.long)
    new_senders = torch.cat([self_loop_idx, data.edge_index[0]])
    new_receivers = torch.cat([self_loop_idx, data.edge_index[1]])
    new_edge_index = torch.stack([new_senders, new_receivers])
    new_edge_attr = torch.ones(new_edge_index.size(1), 1, dtype=torch.float32)
    data = data.clone()
    data.edge_index = new_edge_index
    data.edge_attr = new_edge_attr
    return data


def get_dataset(
    config,
    rng: torch.Generator,
) -> Tuple[Data, torch.Tensor, Dict[str, torch.Tensor]]:
    """Load and preprocess graph dataset.

    Returns (data, labels, masks) where masks is a dict of boolean tensors.
    """
    graph = dataset_readers.get_dataset(config.dataset, config.dataset_path)
    graph = add_reverse_edges(graph)
    graph = subsample_graph(graph, config.max_degree, rng)
    masks_np = compute_masks_for_splits(graph)
    data, labels_np = convert_to_pyg_data(graph)
    data = add_self_loops(data)
    data = normalizations.normalize_edges_with_mask(
        data, mask=None, adjacency_normalization=config.adjacency_normalization)
    labels = torch.tensor(labels_np, dtype=torch.long)
    masks = {k: torch.tensor(v) for k, v in masks_np.items()}
    return data, labels, masks
