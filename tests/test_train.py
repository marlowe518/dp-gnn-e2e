"""Tests for train module.

Mirrors reference tests:
  1. Per-example gradients (DP) should match batched gradients (summed).
  2. train_and_evaluate should complete without errors for all config variants.
"""

import copy
import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from dp_gnn import input_pipeline
from dp_gnn import train
from dp_gnn.dataset_readers import DummyDataset


def _make_dummy_config(model_type, dp=False):
    """Build a dummy config for testing."""
    base = SimpleNamespace(
        dataset='dummy',
        dataset_path='',
        batch_size=max(1, DummyDataset.NUM_DUMMY_TRAINING_SAMPLES // 2),
        max_degree=2,
        num_training_steps=10,
        num_classes=DummyDataset.NUM_DUMMY_CLASSES,
        evaluate_every_steps=5,
        checkpoint_every_steps=5,
        adjacency_normalization='inverse-degree',
        rng_seed=0,
        optimizer='adam',
        learning_rate=3e-3,
        differentially_private_training=dp,
    )
    if model_type == 'mlp':
        base.model = 'mlp'
        base.latent_size = 16
        base.num_layers = 1
        base.activation_fn = 'tanh' if dp else 'relu'
    elif model_type == 'gcn':
        base.model = 'gcn'
        base.latent_size = 16
        base.num_encoder_layers = 1
        base.num_message_passing_steps = 1
        base.num_decoder_layers = 1
        base.activation_fn = 'tanh' if dp else 'relu'
    if dp:
        base.pad_subgraphs_to = 10
        base.num_estimation_samples = 2
        base.l2_norm_clip_percentile = 75
        base.training_noise_multiplier = 2.0
        base.max_training_epsilon = None
    return base


class TestPerExampleGradients:
    @pytest.mark.parametrize("model_type", ['mlp', 'gcn'])
    @pytest.mark.parametrize("max_degree", [0, 1, 2])
    def test_per_example_matches_batched(self, model_type, max_degree):
        config = _make_dummy_config(model_type, dp=True)
        config.max_degree = max_degree

        torch.manual_seed(0)
        rng = torch.Generator()
        rng.manual_seed(0)

        data, labels_int, _ = input_pipeline.get_dataset(config, rng)
        labels = F.one_hot(labels_int, config.num_classes).float()
        num_nodes = labels.shape[0]

        subgraphs = train.get_subgraphs(data, config.pad_subgraphs_to)
        input_dim = data.x.shape[1]
        model = train.create_model(config, input_dim)

        batch_idx = torch.randint(num_nodes, (config.batch_size,))

        sub_weights = None
        if model_type == 'gcn':
            sub_weights = train._precompute_subgraph_weights(
                subgraphs, config.adjacency_normalization)
        per_eg_grads = train.compute_updates_for_dp(
            model, data, labels, subgraphs, batch_idx,
            sub_weights=sub_weights)
        per_eg_summed = {name: g.sum(dim=0) for name, g in per_eg_grads.items()}

        batched_grads = train.compute_updates(model, data, labels, batch_idx)

        for name in batched_grads:
            assert name in per_eg_summed, f"Missing grad for {name}"
            torch.testing.assert_close(
                batched_grads[name], per_eg_summed[name],
                atol=1e-3, rtol=1e-3,
            )


class TestTrainAndEvaluate:
    @pytest.mark.parametrize("model_type,dp", [
        ('gcn', False), ('mlp', False), ('gcn', True), ('mlp', True),
    ])
    def test_runs_without_error(self, model_type, dp):
        config = _make_dummy_config(model_type, dp)
        model = train.train_and_evaluate(config, workdir='/tmp/dp_gnn_test')
        assert model is not None


class TestSubgraphs:
    def test_subgraph_shape(self):
        rng = torch.Generator()
        rng.manual_seed(0)
        config = SimpleNamespace(
            dataset='dummy', dataset_path='', max_degree=2,
            adjacency_normalization='inverse-degree',
        )
        data, _, _ = input_pipeline.get_dataset(config, rng)
        sg = train.get_subgraphs(data, pad_to=5)
        assert sg.shape == (data.num_nodes, 5)
        for i in range(data.num_nodes):
            assert sg[i, 0].item() == i

    def test_make_subgraph(self):
        rng = torch.Generator()
        rng.manual_seed(0)
        config = SimpleNamespace(
            dataset='dummy', dataset_path='', max_degree=2,
            adjacency_normalization='inverse-degree',
        )
        data, _, _ = input_pipeline.get_dataset(config, rng)
        sg = train.get_subgraphs(data, pad_to=5)
        sub_data = train.make_subgraph_from_indices(data, sg[0], 'inverse-degree')
        assert sub_data.x is not None
        assert sub_data.edge_index is not None


class TestSensitivity:
    def test_mlp_sensitivity(self):
        config = SimpleNamespace(model='mlp', max_degree=5, num_message_passing_steps=1)
        assert train.compute_base_sensitivity(config) == 1.0
        assert train.compute_max_terms_per_node(config) == 1

    def test_gcn_1hop_sensitivity(self):
        config = SimpleNamespace(model='gcn', max_degree=5, num_message_passing_steps=1)
        assert train.compute_base_sensitivity(config) == 12.0
        assert train.compute_max_terms_per_node(config) == 6

    def test_gcn_2hop_sensitivity(self):
        config = SimpleNamespace(model='gcn', max_degree=3, num_message_passing_steps=2)
        assert train.compute_base_sensitivity(config) == 26.0
        assert train.compute_max_terms_per_node(config) == 13


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
