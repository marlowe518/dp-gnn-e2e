"""Tests for checkpoint utilities."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from dp_gnn.checkpoint_utils import (
    save_checkpoint,
    load_checkpoint,
    load_model_state,
    get_checkpoint_metadata,
    list_checkpoints,
)
from dp_gnn.models import MultiLayerPerceptron, GraphConvolutionalNetwork


class TestCheckpointUtils:
    """Test suite for checkpoint utilities."""

    def test_save_load_mlp(self):
        """Test saving and loading an MLP model."""
        # Create model
        model = MultiLayerPerceptron(
            latent_sizes=[64, 40],
            activation=torch.relu,
            input_dim=128,
        )

        # Get original weights
        original_weights = {
            name: param.clone() for name, param in model.named_parameters()
        }

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'mlp_checkpoint.pt')
            save_checkpoint(model, path, metadata={'test': True, 'epoch': 5})

            # Create new model and load
            new_model = MultiLayerPerceptron(
                latent_sizes=[64, 40],
                activation=torch.relu,
                input_dim=128,
            )
            load_checkpoint(path, model=new_model)

            # Verify weights match
            for (name1, param1), (name2, param2) in zip(
                model.named_parameters(), new_model.named_parameters()
            ):
                assert torch.allclose(param1, param2), f"Mismatch in {name1}"

    def test_save_load_gcn(self):
        """Test saving and loading a GCN model."""
        model = GraphConvolutionalNetwork(
            latent_size=64,
            num_classes=40,
            num_message_passing_steps=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=128,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'gcn_checkpoint.pt')
            save_checkpoint(model, path)

            new_model = GraphConvolutionalNetwork(
                latent_size=64,
                num_classes=40,
                num_message_passing_steps=1,
                num_encoder_layers=1,
                num_decoder_layers=1,
                activation=torch.tanh,
                input_dim=128,
            )
            load_checkpoint(path, model=new_model)

            for (name1, param1), (name2, param2) in zip(
                model.named_parameters(), new_model.named_parameters()
            ):
                assert torch.allclose(param1, param2), f"Mismatch in {name1}"

    def test_metadata_roundtrip(self):
        """Test metadata is preserved round-trip."""
        model = MultiLayerPerceptron(
            latent_sizes=[32, 16],
            activation=torch.relu,
            input_dim=10,
        )

        metadata = {
            'epoch': 10,
            'best_accuracy': 0.95,
            'config': {'lr': 0.001, 'batch_size': 128},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.pt')
            save_checkpoint(model, path, metadata=metadata)

            # Load full checkpoint
            checkpoint = load_checkpoint(path)
            assert checkpoint['metadata'] == metadata

            # Get metadata only
            meta_only = get_checkpoint_metadata(path)
            assert meta_only == metadata

    def test_optimizer_state(self):
        """Test saving and loading optimizer state."""
        model = MultiLayerPerceptron(
            latent_sizes=[32],
            activation=torch.relu,
            input_dim=10,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Do a training step to change optimizer state
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.pt')
            save_checkpoint(model, path, optimizer=optimizer, epoch=5)

            # Create new model and optimizer
            new_model = MultiLayerPerceptron(
                latent_sizes=[32],
                activation=torch.relu,
                input_dim=10,
            )
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

            # Load checkpoint
            checkpoint = load_checkpoint(path, model=new_model, optimizer=new_optimizer)

            assert checkpoint['epoch'] == 5
            # Optimizer state should be loaded
            assert 'optimizer_state_dict' in checkpoint

    def test_load_model_state_only(self):
        """Test loading only model state without instantiating model."""
        model = MultiLayerPerceptron(
            latent_sizes=[64, 40],
            activation=torch.relu,
            input_dim=128,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.pt')
            save_checkpoint(model, path)

            # Load just state dict
            state_dict = load_model_state(path)

            assert 'layers.0.weight' in state_dict
            assert 'layers.0.bias' in state_dict
            assert 'layers.1.weight' in state_dict

    def test_list_checkpoints(self):
        """Test listing checkpoint files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some checkpoint files
            for i in range(3):
                model = MultiLayerPerceptron(
                    latent_sizes=[16],
                    activation=torch.relu,
                    input_dim=10,
                )
                path = os.path.join(tmpdir, f'checkpoint_{i}.pt')
                save_checkpoint(model, path)

            # Create a non-checkpoint file
            with open(os.path.join(tmpdir, 'readme.txt'), 'w') as f:
                f.write('test')

            # List checkpoints
            checkpoints = list_checkpoints(tmpdir)
            assert len(checkpoints) == 3
            assert all(path.endswith('.pt') for path in checkpoints)

    def test_device_loading(self):
        """Test loading to specific device."""
        model = MultiLayerPerceptron(
            latent_sizes=[32],
            activation=torch.relu,
            input_dim=10,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.pt')
            save_checkpoint(model, path)

            # Load to CPU (should work regardless of GPU availability)
            state_dict = load_model_state(path, device='cpu')
            assert state_dict['layers.0.weight'].device.type == 'cpu'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
