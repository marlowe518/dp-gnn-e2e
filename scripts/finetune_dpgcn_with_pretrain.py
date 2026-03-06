"""DP-GCN finetuning with pretrained MLP initialization.

Usage:
    # With pretrained checkpoint
    python scripts/finetune_dpgcn_with_pretrain.py --pretrain-ckpt checkpoints/pretrain/mlp.pt
    
    # Baseline (no pretraining)
    python scripts/finetune_dpgcn_with_pretrain.py --no-pretrain
    
    # Different transfer strategies
    python scripts/finetune_dpgcn_with_pretrain.py --pretrain-ckpt mlp.pt --transfer encoder_only
    python scripts/finetune_dpgcn_with_pretrain.py --pretrain-ckpt mlp.pt --transfer encoder_classifier
"""

import argparse
import os
import sys
import time

sys.path.insert(0, '.')

import torch

from dp_gnn.configs.dpgcn import get_config
from dp_gnn import train
from dp_gnn.transfer import load_mlp_into_gcn, freeze_parameters, get_transferable_parameters


def main():
    parser = argparse.ArgumentParser(description='DP-GCN finetuning with pretrained init')
    parser.add_argument('--pretrain-ckpt', type=str, default=None,
                        help='Path to pretrained MLP checkpoint')
    parser.add_argument('--no-pretrain', action='store_true',
                        help='Run baseline without pretraining')
    parser.add_argument('--transfer', type=str, default='encoder_only',
                        choices=['encoder_only', 'classifier_only', 'encoder_classifier', 'full'],
                        help='Parameter transfer strategy')
    parser.add_argument('--freeze-epochs', type=int, default=0,
                        help='Number of epochs to freeze transferred parameters')
    parser.add_argument('--noise-mult', type=float, default=None,
                        help='Noise multiplier for DP training')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate for finetuning')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of finetuning steps')
    parser.add_argument('--latent-size', type=int, default=100,
                        help='GCN latent size (must match pretrained MLP)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--workdir', type=str, default='/tmp/dp_gnn_pretrain',
                        help='Working directory for checkpoints')

    args = parser.parse_args()

    # Validate arguments
    if not args.no_pretrain and args.pretrain_ckpt is None:
        parser.error('Must provide --pretrain-ckpt or use --no-pretrain for baseline')

    # Get base config
    config = get_config()
    config.device = args.device
    config.rng_seed = args.seed
    config.latent_size = args.latent_size

    # Apply overrides
    if args.noise_mult is not None:
        config.training_noise_multiplier = args.noise_mult
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.epochs is not None:
        config.num_training_steps = args.epochs

    print('=' * 60)
    print('DP-GCN Finetuning' + (' (Baseline)' if args.no_pretrain else ' with Pretrained Init'))
    print('=' * 60)
    print(f'Latent size: {config.latent_size}')
    print(f'Num steps: {config.num_training_steps}')
    print(f'Batch size: {config.batch_size}')
    print(f'Learning rate: {config.learning_rate}')
    print(f'Noise multiplier: {config.training_noise_multiplier}')
    print(f'Device: {config.device}')
    print(f'Seed: {config.rng_seed}')

    if not args.no_pretrain:
        print(f'Pretrain checkpoint: {args.pretrain_ckpt}')
        print(f'Transfer strategy: {args.transfer}')
        print(f'Freeze epochs: {args.freeze_epochs}')
    print('=' * 60)

    # Run training
    t0 = time.time()

    # Use modified training that supports pretrained init
    model = train_with_pretrained_init(config, args)

    elapsed = time.time() - t0
    print(f'\nFinetuning completed in {elapsed:.1f}s ({elapsed/60:.1f}m)')


def train_with_pretrained_init(config, args):
    """Training loop with pretrained initialization support."""
    from dp_gnn.train import (
        create_model, compute_logits, compute_metrics,
        estimate_clipping_thresholds, compute_base_sensitivity,
        compute_max_terms_per_node, _clip_and_sum_gcn_vmap,
        get_subgraphs, _precompute_subgraph_weights,
    )
    from dp_gnn import input_pipeline
    from dp_gnn import privacy_accountants
    from dp_gnn import optimizers as dp_optimizers
    import torch.nn.functional as F

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

    print(f'Num training nodes: {num_training_nodes}')

    # Subgraphs for DP
    print('Building subgraphs...')
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

    # Load pretrained weights if provided
    frozen_params = []
    if not args.no_pretrain and args.pretrain_ckpt:
        print(f'Loading pretrained MLP from {args.pretrain_ckpt}...')
        model, info = load_mlp_into_gcn(
            args.pretrain_ckpt,
            model,
            transfer_strategy=args.transfer,
            device=device,
            strict=True,
        )
        print(f'Transferred {len(info["transferred"])} parameters')
        if info['errors']:
            print(f'Errors: {info["errors"]}')

        # Freeze transferred parameters if requested
        if args.freeze_epochs > 0:
            frozen_params = get_transferable_parameters(model, args.transfer)
            freeze_parameters(model, frozen_params)
            print(f'Froze {len(frozen_params)} parameters for first {args.freeze_epochs} steps')

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {num_params}')

    # Estimate clipping thresholds
    dp_gen = torch.Generator()
    dp_gen.manual_seed(config.rng_seed + 1)

    estimation_indices = train_indices[:config.num_estimation_samples]
    print(f'Estimating clipping thresholds from {len(estimation_indices)} samples...')
    l2_norms_threshold = estimate_clipping_thresholds(
        model, data, train_labels, train_subgraphs,
        estimation_indices, config.l2_norm_clip_percentile,
        sub_weights=sub_weights)
    base_sensitivity = compute_base_sensitivity(config)
    print(f'Clipping thresholds computed')
    print(f'Base sensitivity: {base_sensitivity}')

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Initial metrics
    model.eval()
    with torch.no_grad():
        logits = compute_logits(model, data)
    init_metrics = compute_metrics(logits, labels, masks)
    print(f'Initial: train_acc={init_metrics["train_accuracy"]*100:.2f}%, '
          f'val_acc={init_metrics["val_accuracy"]*100:.2f}%')

    # Training loop
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_step = 0

    for step in range(1, config.num_training_steps):
        model.train()

        # Unfreeze if freeze period is over
        if args.freeze_epochs > 0 and step == args.freeze_epochs:
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

        # Evaluate periodically
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

            print(f'Step {step:4d}: ε={training_epsilon:.2f}, '
                  f'train={train_acc:.2f}%, val={val_acc:.2f}%, test={test_acc:.2f}%')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_step = step

            # Check privacy budget
            if (getattr(config, 'max_training_epsilon', None) and
                training_epsilon >= config.max_training_epsilon):
                print(f'Privacy budget exhausted at step {step}')
                break

    print(f'\nBest: step={best_step}, val_acc={best_val_acc:.2f}%, test_acc={best_test_acc:.2f}%')

    return model


if __name__ == '__main__':
    main()
