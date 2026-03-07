"""Hyperparameter optimization for DP-MLP pretraining (transductive setting).

Usage:
    python scripts/hyperparam_search.py --search-type grid --max-epsilon 20
    python scripts/hyperparam_search.py --search-type random --n-trials 50
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple
import itertools

sys.path.insert(0, '.')

import torch
import numpy as np
from types import SimpleNamespace

from dp_gnn.dp_pretrain import pretrain_mlp_dp


def create_config(
    noise_mult: float,
    clip_percentile: float,
    lr: float,
    batch_size: int,
    epochs: int,
    max_epsilon: float = None,
) -> SimpleNamespace:
    """Create config with given hyperparameters."""
    return SimpleNamespace(
        dataset='ogbn-arxiv',
        dataset_path='datasets/',
        use_train_nodes_only=True,
        max_degree=5,
        adjacency_normalization='inverse-degree',
        multilabel=False,
        latent_size=100,
        num_layers=3,
        activation_fn='tanh',
        num_classes=40,
        num_epochs=epochs,
        batch_size=batch_size,
        optimizer='adam',
        learning_rate=lr,
        weight_decay=0.0,
        noise_multiplier=noise_mult,
        l2_norm_clip_percentile=clip_percentile,
        num_estimation_samples=10000,
        max_epsilon=max_epsilon,
        evaluate_every_epochs=5,
        rng_seed=0,
        device='cuda',
    )


def run_single_trial(
    config: SimpleNamespace,
    trial_id: int,
    verbose: bool = False
) -> Dict:
    """Run a single hyperparameter trial."""
    try:
        model, history = pretrain_mlp_dp(config, verbose=verbose)
        
        return {
            'trial_id': trial_id,
            'hyperparameters': {
                'noise_multiplier': config.noise_multiplier,
                'clip_percentile': config.l2_norm_clip_percentile,
                'learning_rate': config.learning_rate,
                'batch_size': config.batch_size,
                'num_epochs': config.num_epochs,
                'max_epsilon': config.max_epsilon,
            },
            'results': {
                'test_acc': history['final_test_acc'] * 100,
                'val_acc': history['final_val_acc'] * 100,
                'train_acc': history['final_train_acc'] * 100,
                'final_epsilon': history['final_epsilon'] if history['final_epsilon'] != float('inf') else None,
                'steps_taken': history['steps_taken'],
            },
            'success': True,
        }
    except Exception as e:
        return {
            'trial_id': trial_id,
            'hyperparameters': {
                'noise_multiplier': config.noise_multiplier,
                'clip_percentile': config.l2_norm_clip_percentile,
                'learning_rate': config.learning_rate,
                'batch_size': config.batch_size,
            },
            'error': str(e),
            'success': False,
        }


def grid_search(
    param_grid: Dict[str, List],
    max_epsilon: float,
    output_dir: str,
) -> List[Dict]:
    """Perform grid search over hyperparameters."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    all_results = []
    total_combinations = np.prod([len(v) for v in values])
    
    print(f"\n{'='*70}")
    print(f"GRID SEARCH: {total_combinations} combinations")
    print(f"Max Epsilon: {max_epsilon}")
    print(f"{'='*70}\n")
    
    for i, combination in enumerate(itertools.product(*values)):
        params = dict(zip(keys, combination))
        
        print(f"\n[{i+1}/{total_combinations}] Testing: {params}")
        
        config = create_config(
            noise_mult=params['noise_multiplier'],
            clip_percentile=params['clip_percentile'],
            lr=params['learning_rate'],
            batch_size=params['batch_size'],
            epochs=params['num_epochs'],
            max_epsilon=max_epsilon,
        )
        
        result = run_single_trial(config, trial_id=i, verbose=False)
        all_results.append(result)
        
        if result['success']:
            print(f"  ✓ Test Acc: {result['results']['test_acc']:.2f}%, "
                  f"ε: {result['results']['final_epsilon']:.2f}")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown')}")
    
    return all_results


def random_search(
    param_distributions: Dict[str, Tuple],
    n_trials: int,
    max_epsilon: float,
    output_dir: str,
) -> List[Dict]:
    """Perform random search over hyperparameters."""
    all_results = []
    
    print(f"\n{'='*70}")
    print(f"RANDOM SEARCH: {n_trials} trials")
    print(f"Max Epsilon: {max_epsilon}")
    print(f"{'='*70}\n")
    
    for i in range(n_trials):
        # Sample from distributions
        params = {
            'noise_multiplier': np.random.uniform(*param_distributions['noise_multiplier']),
            'clip_percentile': np.random.uniform(*param_distributions['clip_percentile']),
            'learning_rate': 10 ** np.random.uniform(*param_distributions['learning_rate_log']),
            'batch_size': np.random.choice(param_distributions['batch_size']),
            'num_epochs': np.random.choice(param_distributions['num_epochs']),
        }
        
        print(f"\n[{i+1}/{n_trials}] Testing: noise={params['noise_multiplier']:.4f}, "
              f"clip={params['clip_percentile']:.1f}, lr={params['learning_rate']:.4f}")
        
        config = create_config(
            noise_mult=params['noise_multiplier'],
            clip_percentile=params['clip_percentile'],
            lr=params['learning_rate'],
            batch_size=params['batch_size'],
            epochs=params['num_epochs'],
            max_epsilon=max_epsilon,
        )
        
        result = run_single_trial(config, trial_id=i, verbose=False)
        all_results.append(result)
        
        if result['success']:
            print(f"  ✓ Test Acc: {result['results']['test_acc']:.2f}%, "
                  f"ε: {result['results']['final_epsilon']:.2f}")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown')}")
    
    return all_results


def analyze_results(results: List[Dict], output_dir: str):
    """Analyze and report search results."""
    successful = [r for r in results if r['success']]
    
    if not successful:
        print("\n✗ No successful trials!")
        return
    
    # Sort by test accuracy
    successful.sort(key=lambda x: x['results']['test_acc'], reverse=True)
    
    print(f"\n{'='*70}")
    print("TOP 10 CONFIGURATIONS (by test accuracy)")
    print(f"{'='*70}")
    print(f"{'Rank':<6}{'Test Acc':<12}{'Val Acc':<12}{'Epsilon':<12}{'Noise':<10}{'LR':<10}")
    print("-"*70)
    
    for i, r in enumerate(successful[:10]):
        h = r['hyperparameters']
        res = r['results']
        print(f"{i+1:<6}"
              f"{res['test_acc']:<12.2f}"
              f"{res['val_acc']:<12.2f}"
              f"{res['final_epsilon']:<12.2f}"
              f"{h['noise_multiplier']:<10.4f}"
              f"{h['learning_rate']:<10.4f}")
    
    # Best config details
    best = successful[0]
    print(f"\n{'='*70}")
    print("BEST CONFIGURATION")
    print(f"{'='*70}")
    for k, v in best['hyperparameters'].items():
        print(f"  {k}: {v}")
    print(f"\nResults:")
    for k, v in best['results'].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'hyperparam_search_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'results': results,
            'best_config': best,
            'summary': {
                'total_trials': len(results),
                'successful_trials': len(successful),
                'best_test_acc': best['results']['test_acc'],
                'best_val_acc': best['results']['val_acc'],
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return best


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search for DP-MLP')
    parser.add_argument('--search-type', type=str, default='grid',
                        choices=['grid', 'random'],
                        help='Type of hyperparameter search')
    parser.add_argument('--max-epsilon', type=float, default=20.0,
                        help='Maximum privacy budget')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials for random search')
    parser.add_argument('--output-dir', type=str, default='results/hyperparam_search',
                        help='Output directory')
    
    args = parser.parse_args()
    
    if args.search_type == 'grid':
        # Define parameter grid
        param_grid = {
            'noise_multiplier': [0.001, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02],
            'clip_percentile': [25.0, 50.0, 75.0],
            'learning_rate': [0.003, 0.005, 0.01, 0.02],
            'batch_size': [10000],
            'num_epochs': [20],
        }
        
        results = grid_search(param_grid, args.max_epsilon, args.output_dir)
        
    else:  # random search
        # Define parameter distributions
        param_distributions = {
            'noise_multiplier': (0.001, 0.05),      # Uniform [0.001, 0.05]
            'clip_percentile': (25.0, 75.0),         # Uniform [25, 75]
            'learning_rate_log': (-3, -1.5),         # Log uniform [1e-3, ~0.03]
            'batch_size': [10000, 90941],            # Discrete choices
            'num_epochs': [20, 50],                  # Discrete choices
        }
        
        results = random_search(
            param_distributions, 
            args.n_trials, 
            args.max_epsilon, 
            args.output_dir
        )
    
    # Analyze results
    best = analyze_results(results, args.output_dir)
    
    print(f"\n{'='*70}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
