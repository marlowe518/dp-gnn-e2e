"""Tests for DP-MLP pretraining module."""

import sys
sys.path.insert(0, '.')

import torch
from types import SimpleNamespace

# pytest is optional
try:
    import pytest
except ImportError:
    pytest = None

from dp_gnn import dp_pretrain
from dp_gnn import models


class TestDPPretainCore:
    """Test core DP-MLP pretraining functionality."""

    def test_create_pretraining_model(self):
        """Test MLP model creation for pretraining."""
        config = SimpleNamespace(
            latent_size=64,
            num_layers=3,
            activation_fn='tanh',
            num_classes=40,
        )
        model = dp_pretrain.create_pretraining_model(config, input_dim=128)

        assert isinstance(model, models.GraphMultiLayerPerceptron)
        assert model.mlp.layers[0].in_features == 128
        assert model.mlp.layers[0].out_features == 64
        assert model.mlp.layers[-1].out_features == 40

    def test_compute_metrics(self):
        """Test metric computation."""
        logits = torch.randn(100, 10)
        labels = torch.randint(0, 10, (100,))
        mask = torch.rand(100) > 0.5

        loss, acc = dp_pretrain.compute_metrics(logits, labels, mask)

        assert 0 <= loss
        assert 0 <= acc <= 1

    def test_compute_per_example_grads_mlp_vmap(self):
        """Test per-example gradient computation."""
        device = torch.device('cpu')
        config = SimpleNamespace(
            latent_size=32,
            num_layers=2,
            activation_fn='tanh',
            num_classes=5,
        )
        model = dp_pretrain.create_pretraining_model(config, input_dim=16).to(device)

        data_x = torch.randn(50, 16, device=device)
        labels = torch.randint(0, 5, (50,), device=device)
        node_indices = torch.arange(0, 10, device=device)

        per_eg_grads = dp_pretrain._compute_per_example_grads_mlp_vmap(
            model, data_x, labels, node_indices, chunk_size=5)

        # Check structure
        assert 'mlp.layers.0.weight' in per_eg_grads
        assert 'mlp.layers.0.bias' in per_eg_grads
        assert per_eg_grads['mlp.layers.0.weight'].shape[0] == 10  # Batch size

    def test_clip_and_accumulate_gradients(self):
        """Test gradient clipping and accumulation."""
        device = torch.device('cpu')
        config = SimpleNamespace(
            latent_size=32,
            num_layers=2,
            activation_fn='tanh',
            num_classes=5,
        )
        model = dp_pretrain.create_pretraining_model(config, input_dim=16).to(device)

        data_x = torch.randn(50, 16, device=device)
        labels = torch.randint(0, 5, (50,), device=device)
        node_indices = torch.arange(0, 10, device=device)

        # Define thresholds
        thresholds = {
            'mlp.layers.0.weight': 1.0,
            'mlp.layers.0.bias': 1.0,
            'mlp.layers.1.weight': 1.0,
            'mlp.layers.1.bias': 1.0,
        }

        clipped_sum = dp_pretrain._clip_and_accumulate_gradients(
            model, data_x, labels, node_indices, thresholds, chunk_size=5)

        # Check structure - should be summed, so no batch dim
        assert 'mlp.layers.0.weight' in clipped_sum
        assert clipped_sum['mlp.layers.0.weight'].shape == model.mlp.layers[0].weight.shape

    def test_estimate_clipping_thresholds(self):
        """Test threshold estimation."""
        device = torch.device('cpu')
        config = SimpleNamespace(
            latent_size=32,
            num_layers=2,
            activation_fn='tanh',
            num_classes=5,
        )
        model = dp_pretrain.create_pretraining_model(config, input_dim=16).to(device)

        data_x = torch.randn(50, 16, device=device)
        labels = torch.randint(0, 5, (50,), device=device)
        estimation_indices = torch.arange(0, 20, device=device)

        thresholds = dp_pretrain._estimate_clipping_thresholds(
            model, data_x, labels, estimation_indices, l2_norm_clip_percentile=75.0)

        # Check all parameters have thresholds
        for name, param in model.named_parameters():
            assert name in thresholds
            assert thresholds[name] > 0


class TestDPPretainIntegration:
    """Integration tests for DP-MLP pretraining."""

    def test_pretrain_mlp_dp_smoke_test(self):
        """Smoke test for full DP pretraining loop with DummyDataset."""
        config = SimpleNamespace(
            # Dataset
            dataset='dummy',
            dataset_path='',
            use_train_nodes_only=True,  # DP-safe
            max_degree=5,
            adjacency_normalization='inverse-degree',
            multilabel=False,

            # Model
            latent_size=16,
            num_layers=2,
            activation_fn='tanh',
            num_classes=3,

            # Training (small for testing)
            num_epochs=2,
            batch_size=3,
            optimizer='adam',
            learning_rate=0.01,
            weight_decay=0.0,

            # DP settings
            noise_multiplier=1.0,
            l2_norm_clip_percentile=75.0,
            num_estimation_samples=3,
            max_epsilon=None,

            # Evaluation
            evaluate_every_epochs=1,

            # Reproducibility
            rng_seed=42,
            device='cpu',
        )

        model, history = dp_pretrain.pretrain_mlp_dp(config, verbose=False)

        # Check model is trained
        assert isinstance(model, models.GraphMultiLayerPerceptron)

        # Check history has required fields
        assert 'final_train_acc' in history
        assert 'final_val_acc' in history
        assert 'final_test_acc' in history
        assert 'final_epsilon' in history
        assert 'steps_taken' in history

        # Check epsilon was computed
        assert history['final_epsilon'] > 0
        assert history['steps_taken'] > 0

    def test_pretrain_mlp_dp_with_max_epsilon(self):
        """Test that max_epsilon stops training early."""
        config = SimpleNamespace(
            dataset='dummy',
            dataset_path='',
            use_train_nodes_only=True,
            max_degree=5,
            adjacency_normalization='inverse-degree',
            multilabel=False,

            latent_size=16,
            num_layers=2,
            activation_fn='tanh',
            num_classes=3,

            num_epochs=10,  # Would run many steps without max_epsilon
            batch_size=3,
            optimizer='adam',
            learning_rate=0.01,
            weight_decay=0.0,

            noise_multiplier=10.0,  # High noise = higher epsilon per step
            l2_norm_clip_percentile=75.0,
            num_estimation_samples=3,
            max_epsilon=0.5,  # Stop early

            evaluate_every_epochs=1,
            rng_seed=42,
            device='cpu',
        )

        model, history = dp_pretrain.pretrain_mlp_dp(config, verbose=False)

        # Should have stopped early due to epsilon
        assert history['final_epsilon'] <= 0.5 + 0.1  # Allow small tolerance
        assert history['steps_taken'] < 10 * 3  # Less than full epochs

    def test_privacy_budget_increases_with_steps(self):
        """Verify that epsilon increases as training progresses."""
        config = SimpleNamespace(
            dataset='dummy',
            dataset_path='',
            use_train_nodes_only=True,
            max_degree=5,
            adjacency_normalization='inverse-degree',
            multilabel=False,

            latent_size=16,
            num_layers=2,
            activation_fn='tanh',
            num_classes=3,

            num_epochs=3,
            batch_size=3,
            optimizer='adam',
            learning_rate=0.01,
            weight_decay=0.0,

            noise_multiplier=1.0,
            l2_norm_clip_percentile=75.0,
            num_estimation_samples=3,
            max_epsilon=None,

            evaluate_every_epochs=1,
            rng_seed=42,
            device='cpu',
        )

        model, history = dp_pretrain.pretrain_mlp_dp(config, verbose=False)

        # Check epsilon increases over time
        epsilons = history['epsilon']
        for i in range(1, len(epsilons)):
            assert epsilons[i] > epsilons[i-1], "Epsilon should increase with steps"


if __name__ == '__main__':
    # Run tests
    import sys

    test_classes = [TestDPPretainCore, TestDPPretainIntegration]

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                print(f"  {method_name}...", end=' ')
                try:
                    getattr(instance, method_name)()
                    print("PASS")
                except Exception as e:
                    print(f"FAIL: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.exit(1)

    print(f"\n{'='*60}")
    print("All tests passed!")
    print('='*60)
