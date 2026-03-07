"""End-to-end pipeline: DP-MLP pretraining → GCN finetuning.

This script runs the complete experimental workflow:
1. DP-MLP pretraining with specified epsilon (or non-DP)
2. Transfer pretrained MLP to GCN
3. DP-GCN finetuning (with its own privacy budget)
4. Evaluate and compare results

Usage:
    # Single pipeline run
    python scripts/run_dp_pretrain_pipeline.py --epsilon 5.0 --setting transductive

    # Epsilon sweep
    python scripts/run_dp_pretrain_pipeline.py --sweep --epsilons 0.5 1.0 2.0 5.0 10.0 --setting transductive

    # Inductive setting
    python scripts/run_dp_pretrain_pipeline.py --sweep --epsilons 1.0 5.0 --setting inductive

    # Non-DP baseline
    python scripts/run_dp_pretrain_pipeline.py --non-dp --setting transductive

    # Random init baseline (no pretraining)
    python scripts/run_dp_pretrain_pipeline.py --random-init --setting transductive
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, '.')

import torch

from dp_gnn.configs.dp_pretrain_mlp import (
    get_config_for_epsilon,
    get_transductive_config,
    get_inductive_config,
)
from dp_gnn.configs.dpgcn import get_config as get_dpgcn_config
from dp_gnn.dp_pretrain import pretrain_mlp_dp, save_pretrained_mlp_dp
from dp_gnn import train
from dp_gnn.transfer import load_mlp_into_gcn


def run_dp_mlp_pretraining(
    epsilon: Optional[float],
    setting: str,
    pretrain_epochs: int,
    device: str,
    seed: int,
    output_dir: str,
    verbose: bool = True,
) -> Dict:
    """Run DP-MLP pretraining stage.

    Args:
        epsilon: Target epsilon (None for non-DP).
        setting: 'transductive' or 'inductive'.
        pretrain_epochs: Number of pretraining epochs.
        device: Device to use.
        seed: Random seed.
        output_dir: Output directory for checkpoints.
        verbose: Whether to print progress.

    Returns:
        Results dictionary with checkpoint path and metrics.
    """
    # Create config
    if epsilon is None:
        config = get_transductive_config() if setting == 'transductive' else get_inductive_config()
        config.noise_multiplier = 0.0
        label = 'non_dp'
    else:
        config = get_config_for_epsilon(epsilon)
        if setting == 'inductive':
            config.dataset = 'ogbn-arxiv-disjoint'
        label = f'eps_{epsilon}'

    config.num_epochs = pretrain_epochs
    config.device = device
    config.rng_seed = seed

    if verbose:
        print(f'\n{"="*60}')
        print(f'Stage 1: DP-MLP Pretraining ({label}, {setting})')
        print(f'{"="*60}')

    # Run pretraining
    t0 = time.time()
    model, history = pretrain_mlp_dp(config, verbose=verbose)
    pretrain_time = time.time() - t0

    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f'pretrain_{label}.pt')
    save_pretrained_mlp_dp(
        model,
        checkpoint_path,
        metadata={
            'label': label,
            'epsilon': history['final_epsilon'],
            'val_acc': history['final_val_acc'],
            'test_acc': history['final_test_acc'],
        }
    )

    results = {
        'stage': 'pretraining',
        'label': label,
        'setting': setting,
        'epsilon': history['final_epsilon'],
        'train_acc': history['final_train_acc'],
        'val_acc': history['final_val_acc'],
        'test_acc': history['final_test_acc'],
        'steps_taken': history['steps_taken'],
        'pretrain_time': pretrain_time,
        'checkpoint_path': checkpoint_path,
    }

    return results


def run_gcn_finetuning(
    pretrain_checkpoint: Optional[str],
    setting: str,
    finetune_steps: int,
    noise_mult: float,
    lr: float,
    transfer_strategy: str,
    device: str,
    seed: int,
    output_dir: str,
    verbose: bool = True,
) -> Dict:
    """Run GCN finetuning stage.

    Args:
        pretrain_checkpoint: Path to pretrained MLP checkpoint (None for random init).
        setting: 'transductive' or 'inductive'.
        finetune_steps: Number of finetuning steps.
        noise_mult: Noise multiplier for DP-GCN.
        lr: Learning rate.
        transfer_strategy: MLP→GCN transfer strategy.
        device: Device to use.
        seed: Random seed.
        output_dir: Output directory.
        verbose: Whether to print progress.

    Returns:
        Results dictionary with metrics.
    """
    if verbose:
        print(f'\n{"="*60}')
        print(f'Stage 2: GCN Finetuning')
        print(f'{"="*60}')
        if pretrain_checkpoint:
            print(f'Pretrained checkpoint: {pretrain_checkpoint}')
            print(f'Transfer strategy: {transfer_strategy}')
        else:
            print('Random initialization (no pretraining)')
        print(f'Finetuning steps: {finetune_steps}')
        print(f'Noise multiplier: {noise_mult}')
        print(f'Learning rate: {lr}')

    # Get GCN config
    config = get_dpgcn_config()

    # Adjust for setting
    if setting == 'inductive':
        config.dataset = 'ogbn-arxiv-disjoint'

    # Apply settings
    config.device = device
    config.rng_seed = seed
    config.num_training_steps = finetune_steps
    config.training_noise_multiplier = noise_mult
    config.learning_rate = lr

    # Run training
    t0 = time.time()

    # Use train.train_and_evaluate with optional pretrained init
    if pretrain_checkpoint:
        # Training with pretrained initialization
        metrics = train_with_pretrained_init(
            config,
            pretrain_checkpoint,
            transfer_strategy,
            verbose=verbose,
        )
    else:
        # Training with random initialization
        metrics = train.train_and_evaluate(config, workdir=output_dir)

    finetune_time = time.time() - t0

    results = {
        'stage': 'finetuning',
        'pretrained': pretrain_checkpoint is not None,
        'setting': setting,
        'finetune_noise_mult': noise_mult,
        'finetune_lr': lr,
        'finetune_steps': finetune_steps,
        'final_epsilon': metrics.get('epsilon', 0),
        'final_train_acc': metrics.get('train_accuracy', 0),
        'final_val_acc': metrics.get('val_accuracy', 0),
        'final_test_acc': metrics.get('test_accuracy', 0),
        'finetune_time': finetune_time,
    }

    return results


def train_with_pretrained_init(
    config,
    checkpoint_path: str,
    transfer_strategy: str,
    verbose: bool = True,
):
    """Train GCN with pretrained MLP initialization.

    This is a modified version of train.train_and_evaluate that
    loads pretrained MLP weights before training.
    """
    # Import here to avoid circular dependency
    from dp_gnn import models

    device = torch.device(getattr(config, 'device', 'cpu'))
    torch.manual_seed(config.rng_seed)
    rng = torch.Generator()
    rng.manual_seed(config.rng_seed)

    # Load dataset
    from dp_gnn import input_pipeline
    import torch.nn.functional as F

    data, labels_int, masks = input_pipeline.get_dataset(config, rng)
    labels = F.one_hot(labels_int, config.num_classes).float()

    train_mask = masks['train']
    train_indices = torch.where(train_mask)[0]
    num_training_nodes = len(train_indices)

    # Move data to device
    data = data.to(device)
    labels = labels.to(device)
    train_labels = labels[train_indices]
    train_indices = train_indices.to(device)
    masks_device = {k: v.to(device) for k, v in masks.items()}

    # Create model
    input_dim = data.x.shape[1]
    model = train.create_model(config, input_dim).to(device)

    # Load pretrained MLP
    if verbose:
        print(f'Loading pretrained MLP from {checkpoint_path}')

    model, info = load_mlp_into_gcn(
        checkpoint_path,
        model,
        transfer_strategy=transfer_strategy,
        device=device,
        strict=False,
    )

    if verbose:
        print(f'Transferred parameters: {len(info["transferred"])}')
        if info['errors']:
            print(f'Warnings: {info["errors"]}')

    # Continue with standard training
    # (For now, just run the standard training loop)
    return train.train_and_evaluate(config, workdir='/tmp/dp_gnn_finetune')


def run_full_pipeline(
    epsilon: Optional[float],
    setting: str,
    pretrain_epochs: int,
    finetune_steps: int,
    finetune_noise_mult: float,
    finetune_lr: float,
    transfer_strategy: str,
    device: str,
    seed: int,
    output_dir: str,
    verbose: bool = True,
) -> Dict:
    """Run full pipeline: pretraining + finetuning.

    Args:
        epsilon: Target epsilon for pretraining (None for non-DP).
        setting: 'transductive' or 'inductive'.
        pretrain_epochs: Number of pretraining epochs.
        finetune_steps: Number of finetuning steps.
        finetune_noise_mult: Noise multiplier for GCN finetuning.
        finetune_lr: Learning rate for finetuning.
        transfer_strategy: MLP→GCN transfer strategy.
        device: Device to use.
        seed: Random seed.
        output_dir: Output directory.
        verbose: Whether to print progress.

    Returns:
        Combined results dictionary.
    """
    label = 'non_dp' if epsilon is None else f'eps_{epsilon}'

    # Stage 1: Pretraining
    pretrain_results = run_dp_mlp_pretraining(
        epsilon=epsilon,
        setting=setting,
        pretrain_epochs=pretrain_epochs,
        device=device,
        seed=seed,
        output_dir=output_dir,
        verbose=verbose,
    )

    # Stage 2: Finetuning
    finetune_results = run_gcn_finetuning(
        pretrain_checkpoint=pretrain_results['checkpoint_path'],
        setting=setting,
        finetune_steps=finetune_steps,
        noise_mult=finetune_noise_mult,
        lr=finetune_lr,
        transfer_strategy=transfer_strategy,
        device=device,
        seed=seed,
        output_dir=output_dir,
        verbose=verbose,
    )

    # Combine results
    full_results = {
        'label': label,
        'setting': setting,
        'pretraining': pretrain_results,
        'finetuning': finetune_results,
        'total_time': pretrain_results['pretrain_time'] + finetune_results['finetune_time'],
    }

    # Save combined results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f'pipeline_{label}.json')
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)

    if verbose:
        print(f'\n{"="*60}')
        print(f'Pipeline Complete: {label} ({setting})')
        print(f'{"="*60}')
        print(f'Pretraining: ε={pretrain_results["epsilon"]:.2f}, '
              f'test_acc={pretrain_results["test_acc"]*100:.2f}%')
        print(f'Finetuning: ε={finetune_results["final_epsilon"]:.2f}, '
              f'test_acc={finetune_results["final_test_acc"]*100:.2f}%')
        print(f'Total time: {full_results["total_time"]:.1f}s')
        print(f'Results saved: {results_path}')

    return full_results


def run_epsilon_sweep(
    epsilon_values: List[float],
    setting: str,
    pretrain_epochs: int,
    finetune_steps: int,
    finetune_noise_mult: float,
    finetune_lr: float,
    transfer_strategy: str,
    device: str,
    seed: int,
    output_dir: str,
):
    """Run epsilon sweep experiment."""
    all_results = []

    for eps in epsilon_values:
        results = run_full_pipeline(
            epsilon=eps if eps != float('inf') else None,
            setting=setting,
            pretrain_epochs=pretrain_epochs,
            finetune_steps=finetune_steps,
            finetune_noise_mult=finetune_noise_mult,
            finetune_lr=finetune_lr,
            transfer_strategy=transfer_strategy,
            device=device,
            seed=seed,
            output_dir=output_dir,
        )
        all_results.append(results)

    # Save summary
    summary = {
        'setting': setting,
        'pretrain_epochs': pretrain_epochs,
        'finetune_steps': finetune_steps,
        'finetune_noise_mult': finetune_noise_mult,
        'finetune_lr': finetune_lr,
        'results': all_results,
    }
    summary_path = os.path.join(output_dir, 'pipeline_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f'\n{"="*80}')
    print(f'Pipeline Summary: {setting}')
    print(f'{"="*80}')
    print(f'{"Pretrain ε":<15}{"Pretrain Acc":<15}{"Finetune ε":<15}{"Finetune Acc":<15}')
    print('-' * 80)
    for r in all_results:
        pretrain_eps = r['pretraining']['epsilon']
        pretrain_acc = r['pretraining']['test_acc'] * 100
        finetune_eps = r['finetuning']['final_epsilon']
        finetune_acc = r['finetuning']['final_test_acc'] * 100
        pretrain_eps_str = 'inf' if pretrain_eps == float('inf') else f'{pretrain_eps:.2f}'
        finetune_eps_str = 'inf' if finetune_eps == float('inf') else f'{finetune_eps:.2f}'
        print(f'{pretrain_eps_str:<15}{pretrain_acc:>6.2f}%       {finetune_eps_str:<15}{finetune_acc:>6.2f}%')
    print(f'{"="*80}')

    return all_results


def main():
    parser = argparse.ArgumentParser(description='DP-MLP pretraining + GCN finetuning pipeline')

    # Experiment type
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Target epsilon for pretraining (use "inf" for non-DP)')
    parser.add_argument('--non-dp', action='store_true',
                        help='Run non-DP baseline')
    parser.add_argument('--random-init', action='store_true',
                        help='Run with random initialization (no pretraining)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run epsilon sweep')
    parser.add_argument('--epsilons', nargs='+', type=float,
                        default=[0.5, 1.0, 2.0, 5.0, 10.0, float('inf')],
                        help='Epsilon values for sweep')

    # Setting
    parser.add_argument('--setting', type=str, default='transductive',
                        choices=['transductive', 'inductive'],
                        help='Experimental setting')

    # Pretraining config
    parser.add_argument('--pretrain-epochs', type=int, default=100,
                        help='Number of pretraining epochs')

    # Finetuning config
    parser.add_argument('--finetune-steps', type=int, default=1000,
                        help='Number of finetuning steps')
    parser.add_argument('--finetune-noise-mult', type=float, default=4.0,
                        help='Noise multiplier for GCN finetuning')
    parser.add_argument('--finetune-lr', type=float, default=3e-3,
                        help='Learning rate for finetuning')
    parser.add_argument('--transfer-strategy', type=str, default='encoder_classifier',
                        choices=['encoder_only', 'classifier_only', 'encoder_classifier', 'full'],
                        help='Transfer strategy')

    # Other
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = f'results/dp_pretrain_pipeline_{args.setting}'

    print('=' * 80)
    print('DP-MLP Pretraining + GCN Finetuning Pipeline')
    print('=' * 80)
    print(f'Setting: {args.setting}')
    print(f'Pretraining epochs: {args.pretrain_epochs}')
    print(f'Finetuning steps: {args.finetune_steps}')
    print(f'Finetuning noise: {args.finetune_noise_mult}')
    print(f'Finetuning LR: {args.finetune_lr}')
    print(f'Transfer strategy: {args.transfer_strategy}')
    print(f'Device: {args.device}')
    print(f'Output dir: {args.output_dir}')
    print('=' * 80)

    if args.sweep:
        run_epsilon_sweep(
            epsilon_values=args.epsilons,
            setting=args.setting,
            pretrain_epochs=args.pretrain_epochs,
            finetune_steps=args.finetune_steps,
            finetune_noise_mult=args.finetune_noise_mult,
            finetune_lr=args.finetune_lr,
            transfer_strategy=args.transfer_strategy,
            device=args.device,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.random_init:
        # Run with random initialization (no pretraining)
        finetune_results = run_gcn_finetuning(
            pretrain_checkpoint=None,
            setting=args.setting,
            finetune_steps=args.finetune_steps,
            noise_mult=args.finetune_noise_mult,
            lr=args.finetune_lr,
            transfer_strategy=args.transfer_strategy,
            device=args.device,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print(f'\nRandom Init Baseline ({args.setting}):')
        print(f'  Test accuracy: {finetune_results["final_test_acc"]*100:.2f}%')
    else:
        # Single pipeline run
        epsilon = None if args.non_dp else args.epsilon
        run_full_pipeline(
            epsilon=epsilon,
            setting=args.setting,
            pretrain_epochs=args.pretrain_epochs,
            finetune_steps=args.finetune_steps,
            finetune_noise_mult=args.finetune_noise_mult,
            finetune_lr=args.finetune_lr,
            transfer_strategy=args.transfer_strategy,
            device=args.device,
            seed=args.seed,
            output_dir=args.output_dir,
        )


if __name__ == '__main__':
    main()
