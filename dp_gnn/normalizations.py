"""Edge weight (adjacency) normalization methods.

Reproduces the normalization logic from the reference JAX/jraph implementation
using PyTorch Geometric primitives.
"""

from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree


def normalize_edges_with_mask(
    data: Data,
    mask: Optional[torch.Tensor],
    adjacency_normalization: Optional[str],
) -> Data:
    """Normalizes edge weights with a boolean mask indicating valid edges.

    Args:
        data: PyG Data object with edge_index and edge_attr.
        mask: Boolean tensor of shape [num_edges] indicating valid edges,
              or None (treat all edges as valid).
        adjacency_normalization: One of None, 'inverse-degree',
                                 'inverse-sqrt-degree'.

    Returns:
        Data object with updated edge_attr.
    """
    num_edges = data.edge_index.size(1)
    if mask is None:
        mask = torch.ones(num_edges, dtype=torch.bool, device=data.edge_index.device)

    if adjacency_normalization is None:
        edge_weights = _masked_no_normalization(data, mask)
    elif adjacency_normalization == 'inverse-degree':
        edge_weights = _masked_inverse_degree_normalization(data, mask)
    elif adjacency_normalization == 'inverse-sqrt-degree':
        edge_weights = _masked_inverse_sqrt_degree_normalization(data, mask)
    else:
        raise ValueError(f'Unsupported normalization: {adjacency_normalization}')

    data = data.clone()
    data.edge_attr = edge_weights
    return data


def _masked_no_normalization(data: Data, mask: torch.Tensor) -> torch.Tensor:
    edges = torch.ones(data.edge_index.size(1), dtype=torch.float32,
                       device=data.edge_index.device)
    edges = torch.where(mask, edges, torch.zeros_like(edges))
    return edges.unsqueeze(-1)


def _masked_inverse_degree_normalization(
    data: Data, mask: torch.Tensor,
) -> torch.Tensor:
    num_nodes = data.num_nodes
    senders = data.edge_index[0]

    sender_degree = degree(senders, num_nodes=num_nodes,
                           dtype=torch.float32)
    valid_sender_counts = torch.zeros(num_nodes, dtype=torch.float32,
                                      device=data.edge_index.device)
    valid_sender_counts.scatter_add_(0, senders, mask.float())
    sender_coeffs = 1.0 / torch.clamp(valid_sender_counts, min=1.0)
    edges = sender_coeffs[senders]
    edges = torch.where(mask, edges, torch.zeros_like(edges))
    return edges.unsqueeze(-1)


def _masked_inverse_sqrt_degree_normalization(
    data: Data, mask: torch.Tensor,
) -> torch.Tensor:
    num_nodes = data.num_nodes
    senders = data.edge_index[0]
    receivers = data.edge_index[1]

    sender_counts = torch.zeros(num_nodes, dtype=torch.float32,
                                device=data.edge_index.device)
    sender_counts.scatter_add_(0, senders, mask.float())

    receiver_counts = torch.zeros(num_nodes, dtype=torch.float32,
                                  device=data.edge_index.device)
    receiver_counts.scatter_add_(0, receivers, mask.float())

    sender_coeffs = 1.0 / torch.sqrt(torch.clamp(sender_counts, min=1.0))
    receiver_coeffs = 1.0 / torch.sqrt(torch.clamp(receiver_counts, min=1.0))

    edges = sender_coeffs[senders] * receiver_coeffs[receivers]
    edges = torch.where(mask, edges, torch.zeros_like(edges))
    return edges.unsqueeze(-1)
