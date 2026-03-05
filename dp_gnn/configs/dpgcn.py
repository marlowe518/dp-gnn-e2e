"""DP-GCN hyperparameter configuration (matches reference)."""

from types import SimpleNamespace


def get_config():
    return SimpleNamespace(
        dataset='ogbn-arxiv-disjoint',
        dataset_path='datasets/',
        pad_subgraphs_to=100,
        multilabel=False,
        adjacency_normalization='inverse-degree',
        model='gcn',
        latent_size=100,
        num_encoder_layers=1,
        num_message_passing_steps=1,
        num_decoder_layers=1,
        activation_fn='tanh',
        num_classes=40,
        max_degree=5,
        differentially_private_training=True,
        num_estimation_samples=10000,
        l2_norm_clip_percentile=75,
        training_noise_multiplier=2.0,
        num_training_steps=3000,
        max_training_epsilon=None,
        evaluate_every_steps=50,
        checkpoint_every_steps=50,
        rng_seed=0,
        optimizer='adam',
        learning_rate=3e-3,
        batch_size=10000,
    )
