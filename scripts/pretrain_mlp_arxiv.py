"""Pretrain MLP on ogbn-arxiv for GCN initialization.

Usage:
    python scripts/pretrain_mlp_arxiv.py --config dpgcn_match
    python scripts/pretrain_mlp_arxiv.py --config deeper --epochs 150
"""

import argparse
import os
import sys
import time

sys.path.insert(0, '.')

from dp_gnn.configs.pretrain_mlp import get_config, get_config_for_dpgcn_init, get_deeper_config
from dp_gnn.pretrain import pretrain_mlp, save_pretrained_mlp


def main():
    parser = argparse.ArgumentParser(description='Pretrain MLP on ogbn-arxiv')
    parser.add_argument('--config', type=str, default='dpgcn_match',
                        choices=['dpgcn_match', 'deeper', 'default'],
                        help='Config preset to use')
    parser.add_argument('--latent-size', type=int, default=None,
                        help='Override latent size')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Override number of layers')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save pretrained model')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    args = parser.parse_args()

    # Get config
    if args.config == 'dpgcn_match':
        config = get_config_for_dpgcn_init()
    elif args.config == 'deeper':
        config = get_deeper_config()
    else:
        config = get_config()

    # Apply overrides
    config.device = args.device
    config.rng_seed = args.seed

    if args.latent_size is not None:
        config.latent_size = args.latent_size
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr

    # Default save path
    if args.save_path is None:
        os.makedirs('checkpoints/pretrain', exist_ok=True)
        args.save_path = f'checkpoints/pretrain/mlp_{args.config}_ls{config.latent_size}_nl{config.num_layers}_ep{config.num_epochs}.pt'

    print('=' * 60)
    print('MLP Pretraining on ogbn-arxiv')
    print('=' * 60)
    print(f'Config: {args.config}')
    print(f'Latent size: {config.latent_size}')
    print(f'Num layers: {config.num_layers}')
    print(f'Epochs: {config.num_epochs}')
    print(f'Learning rate: {config.learning_rate}')
    print(f'Batch size: {config.batch_size}')
    print(f'Device: {config.device}')
    print(f'Seed: {config.rng_seed}')
    print(f'Save path: {args.save_path}')
    print('=' * 60)

    # Run pretraining
    t0 = time.time()
    model, history = pretrain_mlp(config, verbose=True)
    elapsed = time.time() - t0

    print(f'\nPretraining completed in {elapsed:.1f}s ({elapsed/60:.1f}m)')

    # Save model
    save_pretrained_mlp(
        model,
        args.save_path,
        metadata={
            'config': args.config,
            'latent_size': config.latent_size,
            'num_layers': config.num_layers,
            'num_epochs': config.num_epochs,
            'learning_rate': config.learning_rate,
            'final_train_acc': history['final_train_acc'],
            'final_val_acc': history['final_val_acc'],
            'final_test_acc': history['final_test_acc'],
            'training_time': elapsed,
            'seed': config.rng_seed,
        }
    )

    print(f'\nFinal Results:')
    print(f'  Train Accuracy: {history["final_train_acc"]*100:.2f}%')
    print(f'  Val Accuracy:   {history["final_val_acc"]*100:.2f}%')
    print(f'  Test Accuracy:  {history["final_test_acc"]*100:.2f}%')
    print(f'\nModel saved to: {args.save_path}')


if __name__ == '__main__':
    main()
