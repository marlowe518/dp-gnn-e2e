"""End-to-end pipeline: Pretrain MLP -> Transfer -> DP-GCN Finetune.

This script runs the complete pipeline:
1. Pretrain MLP on ogbn-arxiv (or use existing checkpoint)
2. Transfer parameters to DP-GCN
3. Finetune DP-GCN with DP
4. Compare with baseline (no pretraining)

Usage:
    # Full pipeline with new pretraining
    python scripts/run_pretrain_finetune_pipeline.py --run-pretrain --run-baseline
    
    # Use existing pretrained checkpoint
    python scripts/run_pretrain_finetune_pipeline.py --pretrain-ckpt checkpoints/pretrain/mlp.pt --run-baseline
    
    # Just finetuning with existing checkpoints
    python scripts/run_pretrain_finetune_pipeline.py --pretrain-ckpt mlp.pt --skip-baseline
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, '.')

import torch


def run_pretrain(args):
    """Run MLP pretraining."""
    from dp_gnn.configs.pretrain_mlp import get_config_for_dpgcn_init
    from dp_gnn.pretrain import pretrain_mlp, save_pretrained_mlp

    config = get_config_for_dpgcn_init()
    config.device = args.device
    config.rng_seed = args.seed

    print('=' * 70)
    print('STAGE 1: MLP Pretraining')
    print('=' * 70)

    t0 = time.time()
    model, history = pretrain_mlp(config, verbose=True)
    elapsed = time.time() - t0

    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(
        args.checkpoint_dir,
        f'mlp_pretrain_ls{config.latent_size}_nl{config.num_layers}_{timestamp}.pt'
    )

    save_pretrained_mlp(
        model,
        save_path,
        metadata={
            'final_train_acc': history['final_train_acc'],
            'final_val_acc': history['final_val_acc'],
            'final_test_acc': history['final_test_acc'],
            'training_time': elapsed,
            'config': {k: v for k, v in config.__dict__.items()},
        }
    )

    print(f'\nPretrained model saved to: {save_path}')

    return save_path, history


def run_finetune(pretrain_ckpt, args, baseline=False):
    """Run DP-GCN finetuning."""
    from dp_gnn.configs.dpgcn import get_config
    from dp_gnn import train
    from dp_gnn.transfer import load_mlp_into_gcn, freeze_parameters, get_transferable_parameters
    from dp_gnn.train import (
        create_model, compute_logits, compute_metrics,
        estimate_clipping_thresholds, compute_base_sensitivity,
        compute_max_terms_per_node, _clip_and_sum_gcn_vmap,
        get_subgraphs, _precompute_subgraph_weights,
    )
    from dp_gnn import input_pipeline
    from dp_gnn import privacy_accountants
    import torch.nn.functional as F

    config = get_config()
    config.device = args.device
    config.rng_seed = args.seed

    # Apply overrides
    if args.noise_mult is not None:
        config.training_noise_multiplier = args.noise_mult
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.steps is not None:
        config.num_training_steps = args.steps

    label = 'BASELINE' if baseline else 'PRETRAINED'
    print('=' * 70)
    print(f'STAGE 2: DP-GCN Finetuning ({label})')
    print('=' * 70)
    print(f'Noise multiplier: {config.training_noise_multiplier}')
    print(f'Learning rate: {config.learning_rate}')
    print(f'Num steps: {config.num_training_steps}')
    print(f'Transfer strategy: {args.transfer}')

    device = torch.device(config.device)
    torch.manual_seed(config.rng_seed)
    rng = torch.Generator()
    rng.manual_seed(config.rng_seed)

    # Load dataset
    data, labels_int, masks = input_pipeline.get_dataset(config, rng)
    labels = F.one_hot(labels_int, config.num_classes).float()

    train_mask = masks['train']
    train_indices = torch.where(train_mask)[0]
    num_training_nodes = len(train_indices)
    train_labels = labels[train_indices]

    # Subgraphs
    subgraphs = get_subgraphs(data, pad_to=config.pad_subgraphs_to)
    train_subgraphs = subgraphs[train_indices]
    sub_weights = _precompute_subgraph_weights(
        train_subgraphs, config.adjacency_normalization)

    # Privacy accountant
    max_terms = compute_max_terms_per_node(config)
    training_privacy_accountant = privacy_accountants.get_training_privacy_accountant(
        config, num_training_nodes, max_terms)

    # Move to device
    data = data.to(device)
    labels = labels.to(device)
    train_labels = train_labels.to(device)
    train_indices = train_indices.to(device)
    masks = {k: v.to(device) for k, v in masks.items()}

    # Create model
    input_dim = data.x.shape[1]
    model = create_model(config, input_dim).to(device)

    # Load pretrained weights
    if not baseline and pretrain_ckpt:
        print(f'Loading pretrained MLP from {pretrain_ckpt}...')
        model, info = load_mlp_into_gcn(
            pretrain_ckpt, model,
            transfer_strategy=args.transfer,
            device=device,
            strict=True,
        )
        print(f'Transferred {len(info["transferred"])} parameters')

        if args.freeze_epochs > 0:
            frozen_params = get_transferable_parameters(model, args.transfer)
            freeze_parameters(model, frozen_params)
            print(f'Froze {len(frozen_params)} parameters for first {args.freeze_epochs} steps')

    # Estimate clipping thresholds
    dp_gen = torch.Generator()
    dp_gen.manual_seed(config.rng_seed + 1)

    estimation_indices = train_indices[:config.num_estimation_samples]
    l2_norms_threshold = estimate_clipping_thresholds(
        model, data, train_labels, train_subgraphs,
        estimation_indices, config.l2_norm_clip_percentile,
        sub_weights=sub_weights)
    base_sensitivity = compute_base_sensitivity(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_step = 0
    results_history = []

    t0 = time.time()

    for step in range(1, config.num_training_steps):
        model.train()

        # Unfreeze if needed
        if not baseline and args.freeze_epochs > 0 and step == args.freeze_epochs:
            for param in model.parameters():
                param.requires_grad = True
            print(f'Step {step}: Unfroze all parameters')

        # Sample batch
        step_gen = torch.Generator()
        step_gen.manual_seed(config.rng_seed * 1000 + step)
        batch_idx = torch.randint(
            num_training_nodes, (config.batch_size,), generator=step_gen)
        batch_idx = batch_idx.to(device)

        # DP training
        clipped_sum = _clip_and_sum_gcn_vmap(
            model, data, train_labels, train_subgraphs, sub_weights,
            batch_idx, l2_norms_threshold)

        noisy_grads = {}
        for name, summed in clipped_sum.items():
            noise_std = (l2_norms_threshold[name] * base_sensitivity
                         * config.training_noise_multiplier)
            if noise_std > 0 and noise_std != float('inf'):
                noise = torch.normal(
                    mean=0.0, std=noise_std, size=summed.shape,
                    generator=dp_gen, dtype=summed.dtype,
                ).to(summed.device)
                noisy_grads[name] = summed + noise
            else:
                noisy_grads[name] = summed

        optimizer.zero_grad()
        for name, p in model.named_parameters():
            if name in noisy_grads:
                p.grad = noisy_grads[name] / config.batch_size
        optimizer.step()

        # Evaluate
        is_last_step = (step == config.num_training_steps - 1)
        if step % config.evaluate_every_steps == 0 or is_last_step:
            training_epsilon = training_privacy_accountant(step + 1)

            model.eval()
            with torch.no_grad():
                logits = compute_logits(model, data)
            step_metrics = compute_metrics(logits, labels, masks)

            train_acc = step_metrics['train_accuracy'] * 100
            val_acc = step_metrics['val_accuracy'] * 100
            test_acc = step_metrics['test_accuracy'] * 100

            results_history.append({
                'step': step,
                'epsilon': training_epsilon,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc,
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_step = step

            if step % 100 == 0 or is_last_step:
                print(f'Step {step:4d}: ε={training_epsilon:.2f}, '
                      f'train={train_acc:.2f}%, val={val_acc:.2f}%, test={test_acc:.2f}%')

            # Check privacy budget
            if (getattr(config, 'max_training_epsilon', None) and
                training_epsilon >= config.max_training_epsilon):
                print(f'Privacy budget exhausted at step {step}')
                break

    elapsed = time.time() - t0

    print(f'\nBest: step={best_step}, val_acc={best_val_acc:.2f}%, test_acc={best_test_acc:.2f}%')

    return {
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'best_step': best_step,
        'final_epsilon': results_history[-1]['epsilon'] if results_history else 0,
        'training_time': elapsed,
        'history': results_history,
    }


def main():
    parser = argparse.ArgumentParser(description='Pretrain + DP-GCN Pipeline')
    parser.add_argument('--run-pretrain', action='store_true',
                        help='Run MLP pretraining')
    parser.add_argument('--pretrain-ckpt', type=str, default=None,
                        help='Path to existing pretrained MLP')
    parser.add_argument('--run-baseline', action='store_true',
                        help='Also run baseline (no pretraining)')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip baseline comparison')
    parser.add_argument('--transfer', type=str, default='encoder_only',
                        choices=['encoder_only', 'encoder_classifier', 'full'],
                        help='Transfer strategy')
    parser.add_argument('--freeze-epochs', type=int, default=0,
                        help='Epochs to freeze transferred params')
    parser.add_argument('--noise-mult', type=float, default=4.0,
                        help='Noise multiplier')
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Learning rate')
    parser.add_argument('--steps', type=int, default=3000,
                        help='Training steps')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/pretrain')
    parser.add_argument('--output', type=str, default='results/pretrain_experiment.json',
                        help='Output JSON file for results')

    args = parser.parse_args()

    # Validate args
    if not args.run_pretrain and not args.pretrain_ckpt:
        parser.error('Must specify --run-pretrain or --pretrain-ckpt')

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'pretrain': None,
        'baseline': None,
        'pretrained': None,
    }

    # Stage 1: Pretraining
    pretrain_ckpt = args.pretrain_ckpt
    if args.run_pretrain:
        pretrain_ckpt, pretrain_history = run_pretrain(args)
        results['pretrain'] = {
            'checkpoint': pretrain_ckpt,
            'history': pretrain_history,
        }

    if not pretrain_ckpt:
        print('No pretrained checkpoint available. Exiting.')
        return

    # Stage 2: Finetuning with pretrained init
    pretrained_results = run_finetune(pretrain_ckpt, args, baseline=False)
    results['pretrained'] = pretrained_results

    # Stage 3: Baseline (if requested)
    if args.run_baseline and not args.skip_baseline:
        baseline_results = run_finetune(None, args, baseline=True)
        results['baseline'] = baseline_results

        # Print comparison
        print('\n' + '=' * 70)
        print('COMPARISON')
        print('=' * 70)
        print(f'Baseline:   test_acc={baseline_results["best_test_acc"]:.2f}%')
        print(f'Pretrained: test_acc={pretrained_results["best_test_acc"]:.2f}%')
        improvement = pretrained_results['best_test_acc'] - baseline_results['best_test_acc']
        print(f'Improvement: {improvement:+.2f}%')
        print('=' * 70)

    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {args.output}')


if __name__ == '__main__':
    main()
