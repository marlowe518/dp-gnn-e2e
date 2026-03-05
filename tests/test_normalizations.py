"""Tests for normalizations module."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from dp_gnn.normalizations import normalize_edges_with_mask


def _make_graph(senders, receivers, num_nodes, node_feat_dim=1):
    """Build a PyG Data object with unit edge weights."""
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    num_edges = edge_index.size(1)
    x = torch.ones(num_nodes, node_feat_dim, dtype=torch.float32)
    edge_attr = torch.ones(num_edges, 1, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = num_nodes
    return data


def _build_adjacency(data):
    """Dense adjacency matrix from edge_index."""
    n = data.num_nodes
    adj = np.zeros((n, n))
    s = data.edge_index[0].numpy()
    r = data.edge_index[1].numpy()
    for u, v in zip(s, r):
        adj[u, v] = 1
    return adj


def _normalize_adj_matrix(adj, mode):
    if mode is None:
        return adj
    if mode == 'inverse-degree':
        d = np.sum(adj, axis=1)
        d_inv = np.diag(1.0 / np.maximum(d, 1.0))
        return d_inv @ adj
    if mode == 'inverse-sqrt-degree':
        d_row = np.sum(adj, axis=1)
        d_col = np.sum(adj, axis=0)
        d_row_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(d_row, 1.0)))
        d_col_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(d_col, 1.0)))
        return d_row_inv_sqrt @ adj @ d_col_inv_sqrt
    raise ValueError(f'Unknown mode: {mode}')


class TestNormalizations:

    # 3-node graph: 0->1, 2->1 (same as reference test)
    @pytest.mark.parametrize("adjacency_normalization", [
        None, 'inverse-degree', 'inverse-sqrt-degree',
    ])
    @pytest.mark.parametrize("add_self_loops", [True, False])
    @pytest.mark.parametrize("symmetrize", [True, False])
    def test_normalization_matches_dense(
        self, adjacency_normalization, add_self_loops, symmetrize
    ):
        senders = [0, 2]
        receivers = [1, 1]
        num_nodes = 3

        if symmetrize:
            senders = senders + receivers
            receivers = receivers + [0, 2]
        if add_self_loops:
            senders = list(range(num_nodes)) + senders
            receivers = list(range(num_nodes)) + receivers

        data = _make_graph(senders, receivers, num_nodes)
        adj = _build_adjacency(data)
        expected_adj = _normalize_adj_matrix(adj, adjacency_normalization)

        result = normalize_edges_with_mask(data, None, adjacency_normalization)
        edge_weights = result.edge_attr.squeeze(-1).numpy()

        # Reconstruct weighted adjacency from edge weights
        reconstructed = np.zeros((num_nodes, num_nodes))
        src = result.edge_index[0].numpy()
        dst = result.edge_index[1].numpy()
        for i, (u, v) in enumerate(zip(src, dst)):
            reconstructed[u, v] = edge_weights[i]

        np.testing.assert_allclose(reconstructed, expected_adj, atol=1e-6)

    def test_mask_zeros_out_edges(self):
        data = _make_graph([0, 1, 2], [1, 2, 0], num_nodes=3)
        mask = torch.tensor([True, False, True])
        result = normalize_edges_with_mask(data, mask, None)
        weights = result.edge_attr.squeeze(-1).numpy()
        assert weights[1] == 0.0
        assert weights[0] == 1.0
        assert weights[2] == 1.0

    def test_none_normalization_unit_weights(self):
        data = _make_graph([0, 1], [1, 0], num_nodes=2)
        result = normalize_edges_with_mask(data, None, None)
        weights = result.edge_attr.squeeze(-1).numpy()
        np.testing.assert_allclose(weights, [1.0, 1.0])

    def test_output_shape(self):
        data = _make_graph([0, 1, 2, 0], [1, 2, 0, 2], num_nodes=3)
        result = normalize_edges_with_mask(data, None, 'inverse-degree')
        assert result.edge_attr.shape == (4, 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
