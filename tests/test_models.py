"""Tests for models module.

Mirrors the reference test: builds a small graph, runs 1-hop GCN with identity
update, and checks that the output equals normalized_adj @ node_features.
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from dp_gnn.models import (
    MultiLayerPerceptron,
    GraphMultiLayerPerceptron,
    OneHopGraphConvolution,
    GraphConvolutionalNetwork,
)
from dp_gnn.normalizations import normalize_edges_with_mask


def _make_graph(senders, receivers, node_features, adjacency_normalization):
    num_nodes = node_features.shape[0]
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    num_edges = edge_index.size(1)
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_attr = torch.ones(num_edges, 1, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = num_nodes
    data = normalize_edges_with_mask(data, None, adjacency_normalization)
    return data


def _get_adjacency_matrix(data):
    n = data.num_nodes
    adj = np.zeros((n, n))
    for u, v in zip(data.edge_index[0].numpy(), data.edge_index[1].numpy()):
        adj[u, v] = 1
    return adj


def _normalize_adjacency(adj, mode):
    if mode is None:
        return adj
    if mode == 'inverse-sqrt-degree':
        sd = np.sum(adj, axis=1)
        sd = np.maximum(sd, 1.0)
        rd = np.sum(adj, axis=0)
        rd = np.maximum(rd, 1.0)
        return np.diag(1.0 / np.sqrt(sd)) @ adj @ np.diag(1.0 / np.sqrt(rd))
    if mode == 'inverse-degree':
        sd = np.sum(adj, axis=1)
        sd = np.maximum(sd, 1.0)
        return np.diag(1.0 / sd) @ adj
    raise ValueError(mode)


class TestOneHopGraphConvolution:
    @pytest.mark.parametrize("add_self_loops", [True, False])
    @pytest.mark.parametrize("symmetrize", [True, False])
    @pytest.mark.parametrize("adjacency_normalization", [
        None, 'inverse-degree', 'inverse-sqrt-degree',
    ])
    def test_matches_dense_matmul(self, add_self_loops, symmetrize,
                                  adjacency_normalization):
        senders = [0, 2]
        receivers = [1, 1]
        num_nodes = 3
        node_features = np.array([[2.0], [1.0], [1.0]], dtype=np.float32)

        if symmetrize:
            senders = senders + [1, 1]
            receivers = receivers + [0, 2]
        if add_self_loops:
            senders = list(range(num_nodes)) + senders
            receivers = list(range(num_nodes)) + receivers

        data = _make_graph(senders, receivers, node_features,
                           adjacency_normalization)

        identity_update = torch.nn.Identity()
        model = OneHopGraphConvolution(update_fn=identity_update)
        with torch.no_grad():
            out_data = model(data)

        adj = _get_adjacency_matrix(data)
        norm_adj = _normalize_adjacency(adj, adjacency_normalization)
        expected = norm_adj @ node_features

        np.testing.assert_allclose(
            out_data.x.numpy(), expected, atol=1e-5)


class TestMLP:
    def test_output_shape(self):
        mlp = MultiLayerPerceptron([16, 8], activation=torch.relu, input_dim=5)
        x = torch.randn(10, 5)
        y = mlp(x)
        assert y.shape == (10, 8)

    def test_skip_connections(self):
        mlp = MultiLayerPerceptron([5, 5], activation=torch.relu, input_dim=5,
                                   skip_connections=True, activate_final=True)
        x = torch.randn(4, 5)
        y = mlp(x)
        assert y.shape == (4, 5)

    def test_no_activation(self):
        mlp = MultiLayerPerceptron([3], activation=None, input_dim=5)
        x = torch.randn(4, 5)
        y = mlp(x)
        assert y.shape == (4, 3)


class TestGraphMLP:
    def test_forward(self):
        data = Data(
            x=torch.randn(6, 5),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.ones(2, 1),
        )
        data.num_nodes = 6
        model = GraphMultiLayerPerceptron([16, 3], activation=torch.relu,
                                          input_dim=5)
        with torch.no_grad():
            out = model(data)
        assert out.x.shape == (6, 3)


class TestGCN:
    def test_forward_shape(self):
        num_nodes = 8
        senders = [0, 1, 2, 3, 4, 5, 6, 7]
        receivers = [1, 2, 3, 4, 5, 6, 7, 0]
        edge_index = torch.tensor([senders, receivers], dtype=torch.long)
        x = torch.randn(num_nodes, 10)
        edge_attr = torch.ones(len(senders), 1)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = num_nodes

        model = GraphConvolutionalNetwork(
            latent_size=16,
            num_classes=5,
            num_message_passing_steps=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=10,
        )
        with torch.no_grad():
            out = model(data)
        assert out.x.shape == (num_nodes, 5)

    def test_loss_finite(self):
        num_nodes = 4
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        x = torch.randn(num_nodes, 5)
        edge_attr = torch.ones(4, 1)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = num_nodes

        model = GraphConvolutionalNetwork(
            latent_size=8, num_classes=3,
            num_message_passing_steps=1,
            num_encoder_layers=1, num_decoder_layers=1,
            activation=torch.tanh, input_dim=5,
        )
        out = model(data)
        labels = torch.randint(0, 3, (num_nodes,))
        loss = torch.nn.functional.cross_entropy(out.x, labels)
        assert torch.isfinite(loss)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
