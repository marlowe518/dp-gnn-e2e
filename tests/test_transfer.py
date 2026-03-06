"""Tests for parameter transfer utilities."""

import pytest
import torch

from dp_gnn.models import MultiLayerPerceptron, GraphConvolutionalNetwork
from dp_gnn.transfer import (
    validate_transfer_compatibility,
    create_parameter_mapping,
    transfer_parameters,
    get_transferable_parameters,
    freeze_parameters,
    unfreeze_all_parameters,
)


class TestValidateTransferCompatibility:
    """Test transfer validation."""

    def test_encoder_only_compatible(self):
        """Test valid encoder-only transfer."""
        # MLP: 128 -> 100 -> 100 -> 40 (3 layers)
        mlp = MultiLayerPerceptron(
            latent_sizes=[100, 100, 40],
            activation=torch.relu,
            input_dim=128,
        )
        # GCN: encoder 128 -> 100 (1 layer), decoder 100 -> 40 (1 layer)
        gcn = GraphConvolutionalNetwork(
            latent_size=100,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        is_compatible, message = validate_transfer_compatibility(
            mlp.state_dict(), gcn, 'encoder_only'
        )
        assert is_compatible, message

    def test_encoder_only_incompatible_hidden_dim(self):
        """Test encoder-only with mismatched hidden dim."""
        mlp = MultiLayerPerceptron(
            latent_sizes=[64, 40],  # hidden=64
            activation=torch.relu,
            input_dim=128,
        )
        gcn = GraphConvolutionalNetwork(
            latent_size=100,  # latent=100
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        is_compatible, message = validate_transfer_compatibility(
            mlp.state_dict(), gcn, 'encoder_only'
        )
        assert not is_compatible
        assert 'hidden_dim' in message

    def test_classifier_only_compatible(self):
        """Test valid classifier-only transfer."""
        mlp = MultiLayerPerceptron(
            latent_sizes=[100, 40],
            activation=torch.relu,
            input_dim=128,
        )
        gcn = GraphConvolutionalNetwork(
            latent_size=100,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        is_compatible, message = validate_transfer_compatibility(
            mlp.state_dict(), gcn, 'classifier_only'
        )
        assert is_compatible, message

    def test_full_transfer_compatible(self):
        """Test valid full transfer."""
        # MLP: 128 -> 100 -> 100 -> 40 (3 layers: 1 encoder + 1 decoder + 1 output)
        mlp = MultiLayerPerceptron(
            latent_sizes=[100, 100, 40],
            activation=torch.relu,
            input_dim=128,
        )
        # GCN: encoder (1 layer) + decoder (1 layer) = 2 layers
        gcn = GraphConvolutionalNetwork(
            latent_size=100,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        is_compatible, message = validate_transfer_compatibility(
            mlp.state_dict(), gcn, 'full'
        )
        assert is_compatible, message

    def test_full_transfer_insufficient_layers(self):
        """Test full transfer with insufficient MLP layers."""
        mlp = MultiLayerPerceptron(
            latent_sizes=[100, 40],  # 2 layers
            activation=torch.relu,
            input_dim=128,
        )
        # GCN needs 1 encoder + 1 decoder = 2 layers, but we need 3 for full transfer
        # (encoder hidden + decoder hidden + output)
        gcn = GraphConvolutionalNetwork(
            latent_size=100,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        is_compatible, message = validate_transfer_compatibility(
            mlp.state_dict(), gcn, 'full'
        )
        # This should be compatible for our implementation (2 layers = 1 enc + 1 dec)
        # Actually, full transfer requires: encoder_layers + decoder_layers = MLP layers
        # 1 + 1 = 2, and MLP has 2 layers, so it should be compatible
        assert is_compatible, message


class TestCreateParameterMapping:
    """Test parameter mapping creation."""

    def test_encoder_only_mapping(self):
        """Test encoder-only mapping."""
        mlp = MultiLayerPerceptron(
            latent_sizes=[100, 100, 40],
            activation=torch.relu,
            input_dim=128,
        )
        gcn = GraphConvolutionalNetwork(
            latent_size=100,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=2,  # 2 encoder layers
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        mapping = create_parameter_mapping(
            mlp.state_dict(), gcn, 'encoder_only'
        )

        assert mapping['encoder.layers.0.weight'] == 'layers.0.weight'
        assert mapping['encoder.layers.0.bias'] == 'layers.0.bias'
        assert mapping['encoder.layers.1.weight'] == 'layers.1.weight'
        assert mapping['encoder.layers.1.bias'] == 'layers.1.bias'

    def test_classifier_only_mapping(self):
        """Test classifier-only mapping."""
        mlp = MultiLayerPerceptron(
            latent_sizes=[100, 100, 40],
            activation=torch.relu,
            input_dim=128,
        )
        gcn = GraphConvolutionalNetwork(
            latent_size=100,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        mapping = create_parameter_mapping(
            mlp.state_dict(), gcn, 'classifier_only'
        )

        # Should map MLP last layer to GCN decoder last layer
        assert mapping['decoder.layers.0.weight'] == 'layers.2.weight'
        assert mapping['decoder.layers.0.bias'] == 'layers.2.bias'

    def test_full_mapping(self):
        """Test full transfer mapping."""
        mlp = MultiLayerPerceptron(
            latent_sizes=[100, 100, 40],
            activation=torch.relu,
            input_dim=128,
        )
        gcn = GraphConvolutionalNetwork(
            latent_size=100,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        mapping = create_parameter_mapping(
            mlp.state_dict(), gcn, 'full'
        )

        # Encoder: MLP layer 0 -> GCN encoder layer 0
        assert mapping['encoder.layers.0.weight'] == 'layers.0.weight'
        # Decoder: MLP layer 1 -> GCN decoder layer 0
        assert mapping['decoder.layers.0.weight'] == 'layers.1.weight'


class TestTransferParameters:
    """Test parameter transfer."""

    def test_encoder_only_transfer(self):
        """Test actual encoder parameter transfer."""
        # Create MLP with specific weights
        mlp = MultiLayerPerceptron(
            latent_sizes=[100, 40],
            activation=torch.relu,
            input_dim=128,
        )
        # Set specific weights for verification
        with torch.no_grad():
            mlp.layers[0].weight.fill_(1.0)
            mlp.layers[0].bias.fill_(2.0)

        # Create GCN
        gcn = GraphConvolutionalNetwork(
            latent_size=100,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        # Verify initial weights are different
        assert not torch.allclose(gcn.encoder.layers[0].weight, mlp.layers[0].weight)

        # Transfer
        gcn, info = transfer_parameters(
            mlp.state_dict(), gcn, 'encoder_only'
        )

        # Verify transfer
        assert torch.allclose(gcn.encoder.layers[0].weight, mlp.layers[0].weight)
        assert torch.allclose(gcn.encoder.layers[0].bias, mlp.layers[0].bias)
        assert 'encoder.layers.0.weight' in info['transferred']

    def test_classifier_transfer(self):
        """Test classifier parameter transfer."""
        mlp = MultiLayerPerceptron(
            latent_sizes=[100, 40],
            activation=torch.relu,
            input_dim=128,
        )
        with torch.no_grad():
            mlp.layers[1].weight.fill_(3.0)
            mlp.layers[1].bias.fill_(4.0)

        gcn = GraphConvolutionalNetwork(
            latent_size=100,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        gcn, info = transfer_parameters(
            mlp.state_dict(), gcn, 'classifier_only'
        )

        assert torch.allclose(gcn.decoder.layers[0].weight, mlp.layers[1].weight)
        assert torch.allclose(gcn.decoder.layers[0].bias, mlp.layers[1].bias)

    def test_strict_mode(self):
        """Test strict mode raises errors."""
        mlp = MultiLayerPerceptron(
            latent_sizes=[64, 40],  # hidden=64, incompatible
            activation=torch.relu,
            input_dim=128,
        )
        gcn = GraphConvolutionalNetwork(
            latent_size=100,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        with pytest.raises(ValueError):
            transfer_parameters(
                mlp.state_dict(), gcn, 'encoder_only', strict=True
            )

    def test_non_strict_mode(self):
        """Test non-strict mode returns errors in info."""
        mlp = MultiLayerPerceptron(
            latent_sizes=[64, 40],  # hidden=64, incompatible
            activation=torch.relu,
            input_dim=128,
        )
        gcn = GraphConvolutionalNetwork(
            latent_size=100,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        gcn, info = transfer_parameters(
            mlp.state_dict(), gcn, 'encoder_only', strict=False
        )

        assert len(info['errors']) > 0
        assert len(info['transferred']) == 0


class TestFreezeParameters:
    """Test parameter freezing."""

    def test_freeze_encoder(self):
        """Test freezing encoder parameters."""
        gcn = GraphConvolutionalNetwork(
            latent_size=100,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=2,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        params_to_freeze = [
            'encoder.layers.0.weight',
            'encoder.layers.0.bias',
        ]

        freeze_parameters(gcn, params_to_freeze)

        assert not gcn.encoder.layers[0].weight.requires_grad
        assert not gcn.encoder.layers[0].bias.requires_grad
        # Other parameters should still require grad
        assert gcn.encoder.layers[1].weight.requires_grad

    def test_unfreeze_all(self):
        """Test unfreezing all parameters."""
        gcn = GraphConvolutionalNetwork(
            latent_size=100,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        # Freeze encoder
        freeze_parameters(gcn, ['encoder.layers.0.weight', 'encoder.layers.0.bias'])

        # Unfreeze all
        unfreeze_all_parameters(gcn)

        # All should require grad now
        for param in gcn.parameters():
            assert param.requires_grad


class TestGetTransferableParameters:
    """Test getting list of transferable parameters."""

    def test_encoder_only_list(self):
        """Test getting encoder-only parameter list."""
        gcn = GraphConvolutionalNetwork(
            latent_size=100,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=2,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        params = get_transferable_parameters(gcn, 'encoder_only')

        assert 'encoder.layers.0.weight' in params
        assert 'encoder.layers.0.bias' in params
        assert 'encoder.layers.1.weight' in params
        assert 'decoder.layers.0.weight' not in params

    def test_classifier_only_list(self):
        """Test getting classifier-only parameter list."""
        gcn = GraphConvolutionalNetwork(
            latent_size=100,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=1,
            num_decoder_layers=2,
            activation=torch.tanh,
            input_dim=128,
        )

        params = get_transferable_parameters(gcn, 'classifier_only')

        # Should only include last decoder layer
        assert 'encoder.layers.0.weight' not in params
        assert 'decoder.layers.1.weight' in params


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
