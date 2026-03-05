"""GCN hyperparameter configuration (matches reference)."""

from types import SimpleNamespace


def get_config():
    return SimpleNamespace(
        dataset='ogbn-arxiv',
        dataset_path='datasets/',
        multilabel=False,
        adjacency_normalization='inverse-degree',
        model='gcn',
        latent_size=256,
        num_encoder_layers=2,
        num_message_passing_steps=1,
        num_decoder_layers=2,
        activation_fn='relu',
        num_classes=40,
        max_degree=30,
        differentially_private_training=False,
        num_training_steps=1000,
        evaluate_every_steps=50,
        checkpoint_every_steps=50,
        rng_seed=0,
        optimizer='adam',
        learning_rate=0.002,
        batch_size=1000,
    )
