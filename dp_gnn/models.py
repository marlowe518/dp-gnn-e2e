"""GNN and MLP models for node-level DP-GNN.

Reproduces the JAX/Flax models from the reference implementation using
PyTorch / PyG.  The key components are:

- MultiLayerPerceptron: standard MLP with optional skip connections.
- GraphMultiLayerPerceptron: applies an MLP to node features (ignores edges).
- OneHopGraphConvolution: weighted edge message-passing + node update.
- GraphConvolutionalNetwork: encoder -> message-passing core -> decoder.
"""

from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import degree


class MultiLayerPerceptron(nn.Module):
    """A multi-layer perceptron with optional skip connections."""

    def __init__(
        self,
        latent_sizes: Sequence[int],
        activation: Optional[Callable[..., torch.Tensor]],
        input_dim: int,
        skip_connections: bool = False,
        activate_final: bool = False,
    ):
        super().__init__()
        self.activation = activation
        self.skip_connections = skip_connections
        self.activate_final = activate_final

        dims = [input_dim] + list(latent_sizes)
        self.layers = nn.ModuleList()
        for i in range(len(latent_sizes)):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            next_x = layer(x)
            if index != len(self.layers) - 1 or self.activate_final:
                if self.activation is not None:
                    next_x = self.activation(next_x)
            if self.skip_connections and next_x.shape == x.shape:
                next_x = next_x + x
            x = next_x
        return x


class GraphMultiLayerPerceptron(nn.Module):
    """MLP applied to node features, ignoring graph structure."""

    def __init__(
        self,
        dimensions: Sequence[int],
        activation: Callable[..., torch.Tensor],
        input_dim: int,
    ):
        super().__init__()
        self.mlp = MultiLayerPerceptron(
            dimensions,
            activation,
            input_dim=input_dim,
            skip_connections=False,
            activate_final=False,
        )

    def forward(self, data: Data) -> Data:
        data = data.clone()
        data.x = self.mlp(data.x)
        return data


class OneHopGraphConvolution(nn.Module):
    """One-hop graph convolution with weighted edges.

    Message-passing occurs against the direction of input edges
    (same convention as the reference: senders/receivers are swapped).
    """

    def __init__(self, update_fn: nn.Module):
        super().__init__()
        self.update_fn = update_fn

    def forward(self, data: Data) -> Data:
        # In the reference, senders & receivers are swapped for MP direction.
        senders = data.edge_index[1]
        receivers = data.edge_index[0]

        edge_weights = data.edge_attr  # shape [E, 1]
        weighted_messages = edge_weights * data.x[senders]  # [E, F]

        num_nodes = data.x.size(0)
        convolved = torch.zeros_like(data.x)
        convolved.scatter_add_(0, receivers.unsqueeze(-1).expand_as(weighted_messages),
                               weighted_messages)

        convolved = self.update_fn(convolved)

        data = data.clone()
        data.x = convolved
        return data


class GraphConvolutionalNetwork(nn.Module):
    """GCN: encoder MLP -> message-passing hops -> decoder MLP."""

    def __init__(
        self,
        latent_size: int,
        num_classes: int,
        num_message_passing_steps: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        activation: Callable[..., torch.Tensor],
        input_dim: int,
    ):
        super().__init__()

        self.encoder = MultiLayerPerceptron(
            [latent_size] * num_encoder_layers,
            activation,
            input_dim=input_dim,
            skip_connections=False,
            activate_final=True,
        )

        self.core_hops = nn.ModuleList()
        for _ in range(num_message_passing_steps):
            node_update = MultiLayerPerceptron(
                [latent_size],
                activation,
                input_dim=latent_size,
                skip_connections=True,
                activate_final=True,
            )
            self.core_hops.append(OneHopGraphConvolution(update_fn=node_update))

        decoder_dims = [latent_size] * (num_decoder_layers - 1) + [num_classes]
        self.decoder = MultiLayerPerceptron(
            decoder_dims,
            activation,
            input_dim=latent_size,
            skip_connections=False,
            activate_final=False,
        )

    def forward(self, data: Data) -> Data:
        data = data.clone()
        data.x = self.encoder(data.x)
        for hop in self.core_hops:
            data = hop(data)
        data.x = self.decoder(data.x)
        return data
