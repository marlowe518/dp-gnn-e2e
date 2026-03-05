"""DP-MLP hyperparameter configuration (matches reference)."""

from types import SimpleNamespace


def get_config():
    return SimpleNamespace(
        dataset='ogbn-arxiv',
        dataset_path='datasets/',
        pad_subgraphs_to=1,
        multilabel=False,
        adjacency_normalization='inverse-degree',
        model='mlp',
        latent_size=256,
        num_layers=1,
        activation_fn='tanh',
        num_classes=40,
        max_degree=1,
        differentially_private_training=True,
        num_estimation_samples=10,
        l2_norm_clip_percentile=75,
        training_noise_multiplier=3.0,
        num_training_steps=500,
        max_training_epsilon=10,
        evaluate_every_steps=50,
        checkpoint_every_steps=50,
        rng_seed=0,
        optimizer='adam',
        learning_rate=0.003,
        batch_size=10000,
    )
