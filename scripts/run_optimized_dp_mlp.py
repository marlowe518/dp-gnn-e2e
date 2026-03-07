"""Run DP-MLP pretraining with optimized hyperparameters.

Usage:
    # Run recommended configuration (ε=20)
    python scripts/run_optimized_dp_mlp.py --privacy-level balanced

    # Run strict privacy (ε=5)
    python scripts/run_optimized_dp_mlp.py --privacy-level strict

    # Run relaxed privacy (ε=100)
    python scripts/run_optimized_dp_mlp.py --privacy-level relaxed

    # Non-DP baseline
    python scripts/run_optimized_dp_mlp.py --privacy-level non-dp
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, '.')

from types import SimpleNamespace
from dp_gnn.dp_pretrain import pretrain_mlp_dp, save_pretrained_mlp_dp


# Optimized configurations from hyperparameter search
OPTIMIZED_CONFIGS = {
    'strict': {
        'name': 'Strict Privacy (ε≈5)',
        'noise_multiplier': 0.003,
        'clip_percentile': 40.0,
        'learning_rate': 0.02,
        'batch_size': 90941,
        'num_epochs': 50,
        'expected_acc': '35-40%',
    },
    'balanced': {
        'name': 'Balanced (ε≈20) - RECOMMENDED',
        'noise_multiplier': 0.006,
        'clip_percentile': 50.0,
        'learning_rate': 0.015,
        'batch_size': 10000,
        'num_epochs': 20,
        'expected_acc': '44-46%',
    },
    'relaxed': {
        'name': 'Relaxed Privacy (ε≈100)',
        'noise_multiplier': 0.01,
        'clip_percentile': 50.0,
        'learning_rate': 0.01,
        'batch_size': 10000,
        'num_epochs': 20,
        'expected_acc': '42-44%',
    },
    'non-dp': {
        'name': 'Non-DP Baseline',
        'noise_multiplier': 0.0,
        'clip_percentile': 75.0,
        'learning_rate': 0.003,
        'batch_size': 10000,
        'num_epochs': 20,
        'expected_acc': '45-47%',
    },
}


def create_config(preset_name: str, device: str = 'cuda'):
    """Create config from preset."""
    preset = OPTIMIZED_CONFIGS[preset_name]
    
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
        num_epochs=preset['num_epochs'],
        batch_size=preset['batch_size'],
        optimizer='adam',
        learning_rate=preset['learning_rate'],
        weight_decay=0.0,
        noise_multiplier=preset['noise_multiplier'],
        l2_norm_clip_percentile=preset['clip_percentile'],
        num_estimation_samples=10000,
        max_epsilon=None,  # Train full epochs, report actual ε
        evaluate_every_epochs=5,
        rng_seed=0,
        device=device,
    )


def run_experiment(preset_name: str, output_dir: str, device: str = 'cuda'):
    """Run a single experiment."""
    preset = OPTIMIZED_CONFIGS[preset_name]
    
    print(f"\n{'='*70}")
    print(f"Running: {preset['name']}")
    print(f"{'='*70}")
    print(f"Expected accuracy: {preset['expected_acc']}")
    print(f"\nHyperparameters:")
    print(f"  noise_multiplier: {preset['noise_multiplier']}")
    print(f"  clip_percentile: {preset['clip_percentile']}")
    print(f"  learning_rate: {preset['learning_rate']}")
    print(f"  batch_size: {preset['batch_size']}")
    print(f"  num_epochs: {preset['num_epochs']}")
    print(f"{'='*70}\n")
    
    config = create_config(preset_name, device)
    
    t0 = time.time()
    model, history = pretrain_mlp_dp(config, verbose=True)
    elapsed = time.time() - t0
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'preset': preset_name,
        'config': {
            'noise_multiplier': preset['noise_multiplier'],
            'clip_percentile': preset['clip_percentile'],
            'learning_rate': preset['learning_rate'],
            'batch_size': preset['batch_size'],
            'num_epochs': preset['num_epochs'],
        },
        'results': {
            'test_acc': history['final_test_acc'],
            'val_acc': history['final_val_acc'],
            'train_acc': history['final_train_acc'],
            'final_epsilon': history['final_epsilon'],
            'steps_taken': history['steps_taken'],
            'training_time': elapsed,
        },
    }
    
    # Save JSON
    results_path = os.path.join(output_dir, f'results_{preset_name}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    model_path = os.path.join(output_dir, f'mlp_{preset_name}.pt')
    save_pretrained_mlp_dp(
        model,
        model_path,
        metadata={
            'preset': preset_name,
            'test_acc': history['final_test_acc'],
            'val_acc': history['final_val_acc'],
            'epsilon': history['final_epsilon'],
        }
    )
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {preset['name']}")
    print(f"{'='*70}")
    print(f"Test Accuracy: {history['final_test_acc']*100:.2f}%")
    print(f"Val Accuracy: {history['final_val_acc']*100:.2f}%")
    print(f"Train Accuracy: {history['final_train_acc']*100:.2f}%")
    print(f"Final Epsilon: {history['final_epsilon']:.2f}")
    print(f"Steps Taken: {history['steps_taken']}")
    print(f"Training Time: {elapsed:.1f}s")
    print(f"\nModel saved: {model_path}")
    print(f"Results saved: {results_path}")
    print(f"{'='*70}\n")
    
    return results


def run_all_experiments(output_dir: str, device: str = 'cuda'):
    """Run all preset configurations."""
    all_results = []
    
    for preset_name in ['non-dp', 'balanced', 'relaxed', 'strict']:
        result = run_experiment(preset_name, output_dir, device)
        all_results.append(result)
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Config':<25}{'Test Acc':<15}{'Val Acc':<15}{'Epsilon':<15}")
    print("-"*70)
    
    for r in all_results:
        preset = OPTIMIZED_CONFIGS[r['preset']]
        test_acc = r['results']['test_acc'] * 100
        val_acc = r['results']['val_acc'] * 100
        eps = r['results']['final_epsilon']
        eps_str = f"{eps:.2f}" if eps != float('inf') else "inf"
        print(f"{preset['name']:<25}{test_acc:<15.2f}{val_acc:<15.2f}{eps_str:<15}")
    
    print("="*70)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'summary_comparison.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'results': all_results,
            'configs': OPTIMIZED_CONFIGS,
        }, f, indent=2)
    
    print(f"\nSummary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Run optimized DP-MLP configs')
    parser.add_argument('--privacy-level', type=str, default='balanced',
                        choices=['strict', 'balanced', 'relaxed', 'non-dp', 'all'],
                        help='Privacy level preset to run')
    parser.add_argument('--output-dir', type=str, default='results/optimized_configs',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.privacy_level == 'all':
        run_all_experiments(args.output_dir, args.device)
    else:
        run_experiment(args.privacy_level, args.output_dir, args.device)


if __name__ == '__main__':
    main()
