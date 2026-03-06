"""Configuration for MLP pretraining."""

from types import SimpleNamespace


def get_config():
    """Returns default configuration for MLP pretraining on ogbn-arxiv.

    Pretraining uses the full graph (not just train split) to maximize
    signal for learning node feature representations.
    """
    return SimpleNamespace(
        # Dataset
        dataset='ogbn-arxiv',
        dataset_path='datasets/',
        use_full_graph=True,  # Use all nodes for pretraining, not just train split
        max_degree=5,  # For graph subsampling (matches DP-GCN setting)
        adjacency_normalization='inverse-degree',
        multilabel=False,

        # Model architecture
        model='mlp',
        input_dim=128,  # ogbn-arxiv feature dimension
        latent_size=256,
        num_layers=3,  # Can be deeper than GCN encoder
        activation_fn='tanh',  # Match GCN activation
        num_classes=40,  # ogbn-arxiv num classes

        # Training settings (non-DP for pretraining)
        num_epochs=100,
        batch_size=10000,
        optimizer='adam',
        learning_rate=0.003,
        weight_decay=0.0,

        # Evaluation
        evaluate_every_epochs=10,

        # Checkpointing
        save_checkpoint=True,
        checkpoint_dir='checkpoints/pretrain',
        checkpoint_every_epochs=50,

        # Reproducibility
        rng_seed=0,
        device='cuda',
    )


def get_config_for_dpgcn_init():
    """Returns config that matches DP-GCN architecture for direct transfer.

    This config creates an MLP that can fully initialize a DP-GCN with:
    - latent_size=100 (matches DP-GCN)
    - num_layers=3 (1 encoder + 1 decoder hidden + 1 output)
    """
    config = get_config()
    config.latent_size = 100
    config.num_layers = 3
    config.activation_fn = 'tanh'
    config.learning_rate = 0.003
    config.num_epochs = 200
    return config


def get_deeper_config():
    """Returns config for deeper MLP pretraining.

    Deeper MLP may learn better feature representations.
    """
    config = get_config()
    config.latent_size = 256
    config.num_layers = 5  # Deeper network
    config.learning_rate = 0.001  # Lower LR for deeper network
    config.num_epochs = 150
    return config
