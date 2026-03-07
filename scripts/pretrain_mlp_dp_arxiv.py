"""DP-MLP pretraining on ogbn-arxiv with epsilon sweep.

Usage:
    # Single epsilon value
    python scripts/pretrain_mlp_dp_arxiv.py --epsilon 5.0 --epochs 100

    # Inductive setting (disjoint)
    python scripts/pretrain_mlp_dp_arxiv.py --inductive --epsilon 5.0

    # Epsilon sweep (multiple runs)
    python scripts/pretrain_mlp_dp_arxiv.py --sweep --epsilons 0.5 1.0 2.0 5.0 10.0 inf

    # Non-DP baseline
    python scripts/pretrain_mlp_dp_arxiv.py --non-dp --epochs 100
"""

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

sys.path.insert(0, '.')

import torch

from dp_gnn.configs.dp_pretrain_mlp import (
    get_config,
    get_transductive_config,
    get_inductive_config,
    get_config_for_epsilon,
)
from dp_gnn.dp_pretrain import pretrain_mlp_dp, save_pretrained_mlp_dp


def run_single_experiment(
    config,
    label: str,
    output_dir: str,
    verbose: bool = True,
) -> dict:
    """Run a single DP-MLP pretraining experiment.

    Args:
        config: Configuration object.
        label: Label for this run (e.g., 'eps_5.0').
        output_dir: Directory to save results.
        verbose: Whether to print progress.

    Returns:
        Results dictionary.
    """
    if verbose:
        print(f'\n{"="*60}')
        print(f'Running: {label}')
        print(f'{"="*60}')
        print(f'Dataset: {config.dataset}')
        print(f'Noise multiplier: {config.noise_multiplier}')
        print(f'Max epsilon: {config.max_epsilon}')
        print(f'Epochs: {config.num_epochs}')
        print(f'Batch size: {config.batch_size}')
        print(f'Learning rate: {config.learning_rate}')
        print(f'Device: {config.device}')

    # Run pretraining
    t0 = time.time()
    model, history = pretrain_mlp_dp(config, verbose=verbose)
    elapsed = time.time() - t0

    # Prepare results
    results = {
        'label': label,
        'dataset': config.dataset,
        'noise_multiplier': config.noise_multiplier,
        'max_epsilon': config.max_epsilon,
        'final_epsilon': history['final_epsilon'],
        'steps_taken': history['steps_taken'],
        'final_train_acc': history['final_train_acc'],
        'final_val_acc': history['final_val_acc'],
        'final_test_acc': history['final_test_acc'],
        'training_time': elapsed,
        'config': {k: v for k, v in config.__dict__.items() if not callable(v)},
    }

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'mlp_{label}.pt')
    save_pretrained_mlp_dp(
        model,
        model_path,
        metadata={
            'label': label,
            'dataset': config.dataset,
            'noise_multiplier': config.noise_multiplier,
            'final_epsilon': history['final_epsilon'],
            'final_val_acc': history['final_val_acc'],
            'final_test_acc': history['final_test_acc'],
        }
    )

    # Save results
    results_path = os.path.join(output_dir, f'results_{label}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f'\nResults for {label}:')
        print(f'  Final epsilon: {history["final_epsilon"]:.2f}')
        print(f'  Train accuracy: {history["final_train_acc"]*100:.2f}%')
        print(f'  Val accuracy: {history["final_val_acc"]*100:.2f}%')
        print(f'  Test accuracy: {history["final_test_acc"]*100:.2f}%')
        print(f'  Training time: {elapsed:.1f}s')
        print(f'  Model saved: {model_path}')
        print(f'  Results saved: {results_path}')

    return results


def run_epsilon_sweep(
    epsilon_values: List[float],
    setting: str,
    output_dir: str,
    epochs: int,
    device: str,
    seed: int,
) -> List[dict]:
    """Run epsilon sweep experiment.

    Args:
        epsilon_values: List of epsilon values to test.
        setting: 'transductive' or 'inductive'.
        output_dir: Directory to save results.
        epochs: Number of training epochs.
        device: Device to use.
        seed: Random seed.

    Returns:
        List of results dictionaries.
    """
    all_results = []

    for eps in epsilon_values:
        # Create config
        if eps == float('inf'):
            config = get_transductive_config() if setting == 'transductive' else get_inductive_config()
            config.noise_multiplier = 0.0
            label = 'non_dp'
        else:
            config = get_config_for_epsilon(eps)
            if setting == 'inductive':
                config.dataset = 'ogbn-arxiv-disjoint'
            label = f'eps_{eps}'

        config.num_epochs = epochs
        config.device = device
        config.rng_seed = seed

        # Run experiment
        results = run_single_experiment(config, label, output_dir)
        all_results.append(results)

    # Save summary
    summary = {
        'setting': setting,
        'epochs': epochs,
        'seed': seed,
        'results': all_results,
    }
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n{"="*60}')
    print(f'Epsilon Sweep Summary ({setting})')
    print(f'{"="*60}')
    print(f'{"Epsilon":<12}{"Test Acc":<12}{"Time (s)":<12}')
    print('-' * 60)
    for r in all_results:
        eps_str = 'inf (non-DP)' if r['max_epsilon'] is None else f"{r['final_epsilon']:.2f}"
        print(f'{eps_str:<12}{r["final_test_acc"]*100:>6.2f}%     {r["training_time"]:>6.1f}')
    print(f'{"="*60}')
    print(f'Summary saved: {summary_path}')

    return all_results


def main():
    parser = argparse.ArgumentParser(description='DP-MLP pretraining on ogbn-arxiv')

    # Setting selection
    parser.add_argument('--transductive', action='store_true',
                        help='Use transductive setting (default)')
    parser.add_argument('--inductive', action='store_true',
                        help='Use inductive setting (disjoint graph)')

    # Epsilon configuration
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Target epsilon value (use inf for non-DP)')
    parser.add_argument('--noise-mult', type=float, default=None,
                        help='Override noise multiplier')
    parser.add_argument('--non-dp', action='store_true',
                        help='Run non-DP baseline')

    # Sweep configuration
    parser.add_argument('--sweep', action='store_true',
                        help='Run epsilon sweep')
    parser.add_argument('--epsilons', nargs='+', type=float,
                        default=[0.5, 1.0, 2.0, 5.0, 10.0, float('inf')],
                        help='Epsilon values for sweep')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--latent-size', type=int, default=None,
                        help='Latent dimension')

    # Other
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    # Determine setting
    setting = 'inductive' if args.inductive else 'transductive'

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = f'checkpoints/pretrain_dp_{setting}'

    print('=' * 60)
    print('DP-MLP Pretraining')
    print('=' * 60)
    print(f'Setting: {setting}')
    print(f'Epochs: {args.epochs}')
    print(f'Device: {args.device}')
    print(f'Seed: {args.seed}')
    print(f'Output dir: {args.output_dir}')
    print('=' * 60)

    if args.sweep:
        # Run epsilon sweep
        run_epsilon_sweep(
            epsilon_values=args.epsilons,
            setting=setting,
            output_dir=args.output_dir,
            epochs=args.epochs,
            device=args.device,
            seed=args.seed,
        )
    else:
        # Single run
        if args.non_dp:
            config = get_transductive_config() if setting == 'transductive' else get_inductive_config()
            config.noise_multiplier = 0.0
            config.max_epsilon = None
            label = 'non_dp'
        elif args.epsilon is not None:
            config = get_config_for_epsilon(args.epsilon)
            if setting == 'inductive':
                config.dataset = 'ogbn-arxiv-disjoint'
            label = f'eps_{args.epsilon}'
        else:
            # Default config
            config = get_transductive_config() if setting == 'transductive' else get_inductive_config()
            label = f'eps_auto'

        # Apply overrides
        config.num_epochs = args.epochs
        config.device = args.device
        config.rng_seed = args.seed
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.lr is not None:
            config.learning_rate = args.lr
        if args.latent_size is not None:
            config.latent_size = args.latent_size
        if args.noise_mult is not None:
            config.noise_multiplier = args.noise_mult

        run_single_experiment(config, label, args.output_dir)


if __name__ == '__main__':
    main()
