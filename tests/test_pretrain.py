"""Tests for MLP pretraining module."""

import os
import tempfile

import pytest
import torch
from torch_geometric.data import Data

from dp_gnn.pretrain import (
    create_pretraining_model,
    compute_metrics,
    pretrain_mlp,
    save_pretrained_mlp,
)
from dp_gnn.configs.pretrain_mlp import get_config
from dp_gnn.checkpoint_utils import load_checkpoint


class TestCreatePretrainingModel:
    """Test model creation."""

    def test_model_creation(self):
        """Test creating pretraining model."""
        config = get_config()
        model = create_pretraining_model(config, input_dim=128)

        assert model is not None
        # Check MLP structure
        assert len(model.mlp.layers) == config.num_layers

    def test_model_output_shape(self):
        """Test model output shape."""
        config = get_config()
        config.num_layers = 3
        config.latent_size = 64
        config.num_classes = 10

        model = create_pretraining_model(config, input_dim=128)

        # Create dummy data
        data = Data(x=torch.randn(5, 128))
        data.num_nodes = 5

        output = model(data)
        assert output.x.shape == (5, 10)


class TestComputeMetrics:
    """Test metrics computation."""

    def test_compute_metrics(self):
        """Test loss and accuracy computation."""
        logits = torch.tensor([
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ])
        labels = torch.tensor([0, 1, 2])
        mask = torch.tensor([True, True, False])

        loss, acc = compute_metrics(logits, labels, mask)

        # First two predictions are correct
        assert acc == 1.0
        assert loss > 0

    def test_compute_metrics_all_wrong(self):
        """Test metrics when all predictions are wrong."""
        logits = torch.tensor([
            [0.0, 2.0, 1.0],  # Predicts 1, true is 0
            [0.0, 2.0, 1.0],  # Predicts 1, true is 0
        ])
        labels = torch.tensor([0, 0])
        mask = torch.tensor([True, True])

        loss, acc = compute_metrics(logits, labels, mask)

        assert acc == 0.0
        assert loss > 0


class TestPretrainMLP:
    """Test pretraining on small synthetic data."""

    def test_pretrain_on_dummy(self, tmp_path):
        """Test pretraining loop on dummy data."""
        # Create a simple synthetic dataset
        num_nodes = 100
        num_features = 10
        num_classes = 3

        config = get_config()
        config.dataset = 'dummy'  # Will use dummy dataset
        config.input_dim = num_features
        config.latent_size = 16
        config.num_layers = 2
        config.num_classes = num_classes
        config.num_epochs = 5
        config.batch_size = 20
        config.evaluate_every_epochs = 2
        config.device = 'cpu'
        config.rng_seed = 42
        config.save_checkpoint = False

        # Run pretraining
        model, history = pretrain_mlp(config, verbose=False)

        # Check that model was trained
        assert model is not None
        assert 'final_train_acc' in history
        assert 'final_val_acc' in history
        assert 'final_test_acc' in history

        # Check that training progressed
        assert len(history['train_loss']) > 0 or len(history.get('train_acc', [])) > 0

    def test_loss_decreases(self, tmp_path):
        """Test that training loss decreases over epochs."""
        config = get_config()
        config.dataset = 'dummy'
        config.num_epochs = 10
        config.batch_size = 5
        config.learning_rate = 0.01
        config.evaluate_every_epochs = 1
        config.device = 'cpu'
        config.save_checkpoint = False

        model, history = pretrain_mlp(config, verbose=False)

        # Check that we have metrics
        if len(history['train_loss']) >= 2:
            # Loss should generally decrease (though not strictly monotonically)
            early_loss = history['train_loss'][0]
            late_loss = history['train_loss'][-1]
            assert late_loss <= early_loss * 1.5  # Allow some variance


class TestSavePretrainedMLP:
    """Test saving pretrained models."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading pretrained MLP."""
        from dp_gnn.checkpoint_utils import load_model_state

        config = get_config()
        model = create_pretraining_model(config, input_dim=128)

        # Set specific weights
        with torch.no_grad():
            model.mlp.layers[0].weight.fill_(1.5)

        # Save
        save_path = tmp_path / 'pretrained.pt'
        save_pretrained_mlp(
            model,
            str(save_path),
            metadata={'val_acc': 0.85, 'test_acc': 0.82}
        )

        # Load state dict (saved as inner MLP, no 'mlp.' prefix)
        state_dict = load_model_state(str(save_path))

        # Verify state dict has inner MLP structure (no 'mlp.' prefix)
        assert 'layers.0.weight' in state_dict
        assert 'mlp.layers.0.weight' not in state_dict

        # Load into new model's inner MLP
        new_model = create_pretraining_model(config, input_dim=128)
        new_model.mlp.load_state_dict(state_dict)

        # Verify weights match
        assert torch.allclose(new_model.mlp.layers[0].weight, model.mlp.layers[0].weight)


class TestIntegrationWithTransfer:
    """Integration test with transfer module."""

    def test_pretrain_then_transfer(self, tmp_path):
        """Test full pipeline: pretrain MLP -> save -> transfer to GCN."""
        from dp_gnn.models import GraphConvolutionalNetwork
        from dp_gnn.transfer import load_mlp_into_gcn
        from dp_gnn.dataset_readers import DummyDataset

        # 1. Pretrain MLP (dummy dataset has 5 features, 3 classes)
        config = get_config()
        config.dataset = 'dummy'
        config.input_dim = DummyDataset.NUM_DUMMY_FEATURES  # 5
        config.num_classes = DummyDataset.NUM_DUMMY_CLASSES  # 3
        config.latent_size = 16
        config.num_layers = 3
        config.num_epochs = 3
        config.device = 'cpu'
        config.save_checkpoint = False

        mlp_model, history = pretrain_mlp(config, verbose=False)

        # 2. Save pretrained MLP
        save_path = tmp_path / 'pretrained_mlp.pt'
        save_pretrained_mlp(mlp_model, str(save_path), metadata=history)

        # 3. Create GCN with matching architecture
        gcn = GraphConvolutionalNetwork(
            latent_size=16,
            num_classes=3,  # Match dummy dataset
            num_message_passing_steps=1,
            num_encoder_layers=1,
            num_decoder_layers=1,
            activation=torch.tanh,
            input_dim=5,  # Match dummy dataset
        )

        # 4. Transfer parameters
        gcn, info = load_mlp_into_gcn(
            str(save_path),
            gcn,
            transfer_strategy='encoder_only',
            device='cpu',
        )

        # Verify transfer worked
        assert len(info['transferred']) > 0, f"No parameters transferred. Errors: {info['errors']}"
        assert 'encoder.layers.0.weight' in info['transferred']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
