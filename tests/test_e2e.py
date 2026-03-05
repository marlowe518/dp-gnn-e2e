"""End-to-end pipeline smoke test.

Runs the full train + eval pipeline for all 4 configurations (GCN, MLP,
DP-GCN, DP-MLP) on the DummyDataset and validates:
  1. No crashes.
  2. Model is returned.
  3. Final logits are finite.
  4. Metrics (loss, accuracy) are in valid ranges.
"""

import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from dp_gnn import input_pipeline
from dp_gnn import train
from dp_gnn.dataset_readers import DummyDataset


def _make_dummy_config(model_type, dp=False):
    base = SimpleNamespace(
        dataset='dummy',
        dataset_path='',
        batch_size=max(1, DummyDataset.NUM_DUMMY_TRAINING_SAMPLES // 2),
        max_degree=2,
        num_training_steps=20,
        num_classes=DummyDataset.NUM_DUMMY_CLASSES,
        evaluate_every_steps=10,
        checkpoint_every_steps=10,
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


@pytest.mark.parametrize("model_type,dp", [
    ('gcn', False), ('mlp', False), ('gcn', True), ('mlp', True),
])
def test_e2e_pipeline(model_type, dp):
    config = _make_dummy_config(model_type, dp)
    model = train.train_and_evaluate(config, workdir='/tmp/dp_gnn_e2e')

    assert model is not None

    rng = torch.Generator()
    rng.manual_seed(config.rng_seed)
    data, labels_int, masks = input_pipeline.get_dataset(config, rng)
    labels = F.one_hot(labels_int, config.num_classes).float()

    model.eval()
    with torch.no_grad():
        logits = train.compute_logits(model, data)

    assert torch.all(torch.isfinite(logits)), "Logits contain NaN/Inf"

    metrics = train.compute_metrics(logits, labels, masks)
    for key in ['train_loss', 'val_loss', 'test_loss']:
        assert 0 <= metrics[key] < 100, f"{key} out of range: {metrics[key]}"
    for key in ['train_accuracy', 'val_accuracy', 'test_accuracy']:
        assert 0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
