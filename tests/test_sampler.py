"""Tests for sampler module.

Mirrors the reference tests: verifies occurrence constraints for 1-hop and
2-hop (disjoint) subgraphs under various graph sizes and max_degree settings.
"""

import pytest
import numpy as np
import torch

from dp_gnn.sampler import get_adjacency_lists, sample_adjacency_lists


def _erdos_renyi_edges(num_nodes, edge_probability, seed=42):
    """Create adjacency dict from random graph."""
    rng = np.random.RandomState(seed)
    edges = {u: [] for u in range(num_nodes)}
    for u in range(num_nodes):
        for v in range(num_nodes):
            if u != v and rng.random() < edge_probability:
                edges[u].append(v)
    return edges


def _sample_subgraphs_1hop(edges):
    return edges


def _sample_subgraphs_2hop(edges):
    subgraphs = {}
    for root, neighbors in edges.items():
        subgraphs[root] = {}
        for nbr in neighbors:
            subgraphs[root][nbr] = edges[nbr]
    return subgraphs


def _flatten_subgraphs(subgraphs):
    def _flatten(node, sg):
        if isinstance(sg, list):
            return [node] + sg
        flat = []
        for nbr, nbr_sg in sg.items():
            flat.extend(_flatten(nbr, nbr_sg))
        return flat
    return {n: _flatten(n, s) for n, s in subgraphs.items()}


class TestSamplerOneHop:
    @pytest.mark.parametrize("num_nodes", [10, 20, 50])
    @pytest.mark.parametrize("edge_probability", [0.1, 0.3, 0.5, 0.8, 1.0])
    @pytest.mark.parametrize("max_degree", [1, 2, 5, 10, 20])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_occurrence_constraints_one_hop(
        self, num_nodes, edge_probability, max_degree, seed
    ):
        edges = _erdos_renyi_edges(num_nodes, edge_probability, seed=seed)
        train_nodes = list(range(0, num_nodes, 2))

        rng = torch.Generator()
        rng.manual_seed(seed)
        sampled_edges = sample_adjacency_lists(edges, train_nodes, max_degree, rng)

        sampled_subgraphs = _sample_subgraphs_1hop(sampled_edges)
        flat = _flatten_subgraphs(sampled_subgraphs)

        train_set = set(train_nodes)
        occurrence_counts = {n: 0 for n in sampled_edges}
        for root, subgraph in flat.items():
            if root in train_set:
                for node in subgraph:
                    occurrence_counts[node] += 1

        assert len(sampled_edges) == num_nodes
        for count in occurrence_counts.values():
            assert count <= max_degree + 1


class TestSamplerTwoHopDisjoint:
    @pytest.mark.parametrize("num_nodes", [10, 20, 50])
    @pytest.mark.parametrize("edge_probability", [0.1, 0.3, 0.5, 0.8, 1.0])
    @pytest.mark.parametrize("max_degree", [1, 2, 5, 10, 20])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_occurrence_constraints_two_hop_disjoint(
        self, num_nodes, edge_probability, max_degree, seed
    ):
        num_train = num_nodes // 2
        # Disjoint: train subgraph and non-train subgraph have no cross-edges
        edges_train = _erdos_renyi_edges(num_train, edge_probability, seed=seed)
        edges_nontr = _erdos_renyi_edges(num_nodes - num_train, edge_probability,
                                         seed=seed + 100)

        edges = {u: [] for u in range(num_nodes)}
        for u in edges_train:
            edges[u] = edges_train[u]
        for u in edges_nontr:
            edges[u + num_train] = [v + num_train for v in edges_nontr[u]]

        train_nodes = list(range(num_train))

        rng = torch.Generator()
        rng.manual_seed(seed)
        sampled_edges = sample_adjacency_lists(edges, train_nodes, max_degree, rng)

        sampled_subgraphs = _sample_subgraphs_2hop(sampled_edges)
        flat = _flatten_subgraphs(sampled_subgraphs)

        train_set = set(train_nodes)
        occurrence_counts = {n: 0 for n in sampled_edges}
        for root, subgraph in flat.items():
            if root in train_set:
                for node in subgraph:
                    occurrence_counts[node] += 1

        assert len(sampled_edges) == num_nodes
        for count in occurrence_counts.values():
            assert count <= max_degree ** 2 + max_degree + 1


class TestGetAdjacencyLists:
    def test_basic(self):
        senders = [0, 1, 2]
        receivers = [1, 2, 0]
        adj = get_adjacency_lists(senders, receivers, 3)
        assert adj == {0: [1], 1: [2], 2: [0]}

    def test_mismatch_raises(self):
        with pytest.raises(ValueError):
            get_adjacency_lists([0, 1], [1], 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
