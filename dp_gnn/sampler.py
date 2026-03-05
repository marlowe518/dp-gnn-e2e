"""In-degree bounded sampling for DP-GNN.

Implements Bernoulli-based edge sampling that bounds the in-degree
from training nodes, reproducing the reference JAX implementation.
"""

from typing import Dict, List, Sequence, Set, Union

import numpy as np
import torch

Node = Union[int, str]
AdjacencyDict = Dict[Node, List[Node]]


def reverse_edges(edges: AdjacencyDict) -> AdjacencyDict:
    """Reverses an edgelist to obtain incoming edges for each node."""
    reversed_edges: AdjacencyDict = {u: [] for u in edges}
    for u, u_neighbors in edges.items():
        for v in u_neighbors:
            reversed_edges[v].append(u)
    return reversed_edges


def get_adjacency_lists(senders, receivers, num_nodes: int) -> AdjacencyDict:
    """Returns a dict of adjacency lists from sender/receiver arrays."""
    if len(senders) != len(receivers):
        raise ValueError('Senders and receivers should be of the same length.')
    edges: AdjacencyDict = {u: [] for u in range(num_nodes)}
    for u, v in zip(senders, receivers):
        edges[int(u)].append(int(v))
    return edges


def sample_adjacency_lists(
    edges: AdjacencyDict,
    train_nodes: Sequence[int],
    max_degree: int,
    rng: torch.Generator,
) -> AdjacencyDict:
    """Statelessly samples adjacency lists with in-degree constraints.

    Bernoulli sampling over edges so that no training node appears in more
    than max_degree incoming training-edge lists.

    Non-train nodes keep their full edgelists.
    """
    train_nodes_set: Set[int] = set(int(n) for n in train_nodes)
    all_nodes = edges.keys()

    reversed_edges = reverse_edges(edges)
    sampled_reversed_edges: AdjacencyDict = {u: [] for u in all_nodes}

    dropped_count = 0
    for u in all_nodes:
        incoming_edges = reversed_edges[u]
        incoming_train_edges = [v for v in incoming_edges if v in train_nodes_set]
        if not incoming_train_edges:
            continue

        in_degree = len(incoming_train_edges)
        sampling_prob = max_degree / (2.0 * in_degree)

        # Deterministic seeding per node (mirrors jax.random.fold_in)
        node_gen = torch.Generator()
        # Combine base seed with node id
        base_seed = torch.randint(0, 2**62, (1,), generator=rng).item()
        node_gen.manual_seed(base_seed ^ u)

        sampling_mask = torch.rand(in_degree, generator=node_gen) <= sampling_prob
        incoming_train_edges_arr = np.array(incoming_train_edges)
        sampled = incoming_train_edges_arr[sampling_mask.numpy()]
        unique_sampled = np.unique(sampled)

        if len(unique_sampled) <= max_degree:
            sampled_reversed_edges[u] = unique_sampled.tolist()
        else:
            dropped_count += 1

    sampled_edges = reverse_edges(sampled_reversed_edges)

    for u in all_nodes:
        if u not in train_nodes_set:
            sampled_edges[u] = edges[u]

    return sampled_edges
