"""Configuration for DP-MLP pretraining."""

from types import SimpleNamespace


def get_config():
    """Returns default configuration for DP-MLP pretraining on ogbn-arxiv.

    DP-MLP pretraining uses only training nodes (privacy-safe) and
    applies DP-SGD for privacy guarantees.
    """
    return SimpleNamespace(
        # Dataset
        dataset='ogbn-arxiv',
        dataset_path='datasets/',
        use_train_nodes_only=True,  # CRITICAL: Only use train nodes for DP pretraining
        max_degree=5,
        adjacency_normalization='inverse-degree',
        multilabel=False,

        # Model architecture (should match target GCN for transfer)
        model='mlp',
        input_dim=128,
        latent_size=100,  # Matches DP-GCN latent size
        num_layers=3,  # 1 encoder + 1 hidden + 1 output
        activation_fn='tanh',
        num_classes=40,

        # DP-SGD settings
        noise_multiplier=1.0,  # Controls privacy/utility tradeoff
        l2_norm_clip_percentile=75.0,  # Percentile for gradient clipping
        num_estimation_samples=10000,  # Samples for estimating clip thresholds
        max_epsilon=None,  # Stop if epsilon exceeds this (None = no limit)

        # Training settings
        num_epochs=100,
        batch_size=10000,
        optimizer='adam',
        learning_rate=0.003,
        weight_decay=0.0,

        # Evaluation
        evaluate_every_epochs=10,

        # Checkpointing
        save_checkpoint=True,
        checkpoint_dir='checkpoints/pretrain_dp',
        checkpoint_every_epochs=50,

        # Reproducibility
        rng_seed=0,
        device='cuda',
    )


def get_config_for_epsilon(epsilon: float):
    """Returns config tuned for a target epsilon value.

    Uses grid search results to set appropriate noise_multiplier
    for the target epsilon budget.

    Args:
        epsilon: Target privacy budget (e.g., 1.0, 5.0, 10.0).

    Returns:
        Config with appropriate noise_multiplier.
    """
    config = get_config()

    # Rough mapping based on experiments:
    # Higher noise = higher epsilon for same steps (but worse utility)
    # Lower noise = lower epsilon (better utility but runs out of budget faster)
    if epsilon <= 1.0:
        config.noise_multiplier = 5.0
    elif epsilon <= 5.0:
        config.noise_multiplier = 2.0
    elif epsilon <= 10.0:
        config.noise_multiplier = 1.0
    else:
        config.noise_multiplier = 0.5

    config.max_epsilon = epsilon
    return config


def get_transductive_config():
    """Returns config for transductive setting.

    In transductive setting:
    - Full graph structure is visible
    - Only train node labels are used for training
    - Standard ogbn-arxiv dataset
    """
    config = get_config()
    config.dataset = 'ogbn-arxiv'
    config.use_train_nodes_only = True
    return config


def get_inductive_config():
    """Returns config for inductive setting.

    In inductive setting:
    - Uses ogbn-arxiv-disjoint (no inter-split edges)
    - Train and test graphs are effectively separated
    - Only train node labels are used
    """
    config = get_config()
    config.dataset = 'ogbn-arxiv-disjoint'
    config.use_train_nodes_only = True
    return config


def get_sweep_configs(epsilon_values=[0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]):
    """Returns list of configs for epsilon sweep experiment.

    Args:
        epsilon_values: List of target epsilon values.
            Use float('inf') for non-DP baseline.

    Returns:
        List of (config, label) tuples.
    """
    configs = []

    for eps in epsilon_values:
        if eps == float('inf'):
            # Non-DP baseline
            config = get_config()
            config.noise_multiplier = 0.0  # No noise
            label = 'non_dp'
        else:
            config = get_config_for_epsilon(eps)
            label = f'eps_{eps}'

        configs.append((config, label))

    return configs
