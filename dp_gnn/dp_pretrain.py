"""DP-MLP pretraining module for node classification.

Provides differentially private training of MLP on node features using DP-SGD.
Supports privacy budget tracking and both transductive and inductive settings.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from torch_geometric.data import Data

from dp_gnn import input_pipeline
from dp_gnn import models
from dp_gnn import optimizers as dp_optimizers
from dp_gnn import privacy_accountants
from dp_gnn.checkpoint_utils import save_checkpoint


def create_pretraining_model(config, input_dim: int) -> models.GraphMultiLayerPerceptron:
    """Create an MLP model for pretraining.

    Args:
        config: Configuration object.
        input_dim: Input feature dimension.

    Returns:
        GraphMultiLayerPerceptron instance.
    """
    activation_map = {
        'relu': torch.relu,
        'tanh': torch.tanh,
        'selu': torch.selu,
    }
    activation = activation_map[config.activation_fn]

    dimensions = [config.latent_size] * (config.num_layers - 1) + [config.num_classes]

    return models.GraphMultiLayerPerceptron(
        dimensions=dimensions,
        activation=activation,
        input_dim=input_dim,
    )


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    """Compute loss and accuracy on masked subset.

    Args:
        logits: Model predictions [N, C].
        labels: Ground truth labels [N] (class indices).
        mask: Boolean mask [N].

    Returns:
        (loss, accuracy)
    """
    with torch.no_grad():
        loss_per_node = F.cross_entropy(logits, labels, reduction='none')
        loss_masked = torch.where(mask, loss_per_node, torch.zeros_like(loss_per_node))
        loss = loss_masked.sum() / mask.float().sum()

        preds = logits.argmax(dim=-1)
        correct = (preds == labels).float()
        correct_masked = torch.where(mask, correct, torch.zeros_like(correct))
        accuracy = correct_masked.sum() / mask.float().sum()

    return loss.item(), accuracy.item()


def _compute_per_example_grads_mlp_vmap(
    model: models.GraphMultiLayerPerceptron,
    data_x: torch.Tensor,
    labels: torch.Tensor,
    node_indices: torch.Tensor,
    chunk_size: int = 2000,
) -> Dict[str, torch.Tensor]:
    """Vectorized per-example gradients for MLP using torch.func.vmap.

    Args:
        model: MLP model.
        data_x: Node features [N, F].
        labels: Node labels [N].
        node_indices: Batch indices [B].
        chunk_size: Chunk size for memory efficiency.

    Returns:
        Dict mapping param name -> per-example gradients [B, *param_shape].
    """
    device = data_x.device
    batch_x = data_x[node_indices]
    batch_labels = labels[node_indices]

    mlp_module = model.mlp
    params = dict(mlp_module.named_parameters())

    def single_loss(params, x, label):
        logits = functional_call(mlp_module, params, (x,))
        return F.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0))

    ft_per_sample_grad = vmap(grad(single_loss), in_dims=(None, 0, 0))

    B = len(node_indices)
    chunks_grads = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk_eg = ft_per_sample_grad(
            params, batch_x[start:end], batch_labels[start:end])
        chunks_grads.append(chunk_eg)

    per_example_grads = {}
    for mlp_name in chunks_grads[0]:
        full_name = f'mlp.{mlp_name}'
        per_example_grads[full_name] = torch.cat(
            [c[mlp_name] for c in chunks_grads], dim=0)

    return per_example_grads


def _clip_and_accumulate_gradients(
    model: models.GraphMultiLayerPerceptron,
    data_x: torch.Tensor,
    labels: torch.Tensor,
    node_indices: torch.Tensor,
    l2_norms_threshold: Dict[str, float],
    chunk_size: int = 500,
) -> Dict[str, torch.Tensor]:
    """Compute per-example gradients, clip, and sum across batch.

    Args:
        model: MLP model.
        data_x: Node features [N, F].
        labels: Node labels [N].
        node_indices: Batch indices [B].
        l2_norms_threshold: Per-layer clipping thresholds.
        chunk_size: Chunk size for memory efficiency.

    Returns:
        Dict mapping param name -> summed clipped gradients [*param_shape].
    """
    device = data_x.device
    batch_x = data_x[node_indices]
    batch_labels = labels[node_indices]

    mlp_module = model.mlp
    params = dict(mlp_module.named_parameters())

    def single_loss(params, x, label):
        logits = functional_call(mlp_module, params, (x,))
        return F.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0))

    ft_per_sample_grad = vmap(grad(single_loss), in_dims=(None, 0, 0))

    B = len(node_indices)
    clipped_sum = None

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk_eg = ft_per_sample_grad(
            params, batch_x[start:end], batch_labels[start:end])

        chunk_dict = {f'mlp.{k}': v for k, v in chunk_eg.items()}
        chunk_clipped = dp_optimizers.clip_by_norm(chunk_dict, l2_norms_threshold)

        if clipped_sum is None:
            clipped_sum = {k: v.sum(dim=0) for k, v in chunk_clipped.items()}
        else:
            for k in clipped_sum:
                clipped_sum[k] += chunk_clipped[k].sum(dim=0)

    return clipped_sum


def _estimate_clipping_thresholds(
    model: models.GraphMultiLayerPerceptron,
    data_x: torch.Tensor,
    labels: torch.Tensor,
    estimation_indices: torch.Tensor,
    l2_norm_clip_percentile: float,
    chunk_size: int = 2000,
) -> Dict[str, float]:
    """Estimate per-layer gradient clipping thresholds from a sample.

    Args:
        model: MLP model.
        data_x: Node features [N, F].
        labels: Node labels [N].
        estimation_indices: Indices to use for estimation.
        l2_norm_clip_percentile: Percentile for threshold (e.g., 75.0).
        chunk_size: Chunk size for memory efficiency.

    Returns:
        Dict mapping param name -> clipping threshold.
    """
    per_eg_grads = _compute_per_example_grads_mlp_vmap(
        model, data_x, labels, estimation_indices, chunk_size)

    thresholds = {}
    for name, grad in per_eg_grads.items():
        B = grad.shape[0]
        flat = grad.reshape(B, -1)
        norms = torch.linalg.norm(flat, dim=1)
        threshold = float(np.percentile(norms.detach().cpu().numpy(),
                                        l2_norm_clip_percentile))
        thresholds[name] = max(threshold, 1e-8)
    return thresholds


def pretrain_mlp_dp(
    config,
    verbose: bool = True,
) -> Tuple[models.GraphMultiLayerPerceptron, Dict]:
    """Pretrain an MLP with DP-SGD on node features.

    Args:
        config: Configuration object with pretraining settings.
            Required fields:
                - dataset, dataset_path
                - latent_size, num_layers, activation_fn, num_classes
                - num_epochs, batch_size, learning_rate
                - noise_multiplier (for DP)
                - l2_norm_clip_percentile (default: 75.0)
                - num_estimation_samples (default: 10000)
                - max_epsilon (optional: stop if exceeded)
                - use_train_nodes_only (default: True for DP)
                - rng_seed, device
        verbose: Whether to print progress.

    Returns:
        (trained_model, history) where history contains training metrics.
    """
    device = torch.device(getattr(config, 'device', 'cpu'))
    if verbose:
        print(f'Using device: {device}')

    torch.manual_seed(config.rng_seed)

    # Load dataset
    rng = torch.Generator()
    rng.manual_seed(config.rng_seed)

    data, labels_int, masks = input_pipeline.get_dataset(config, rng)
    labels = labels_int  # Keep as class indices for pretraining

    num_nodes = data.num_nodes
    train_mask = masks['train']
    val_mask = masks['validation']
    test_mask = masks['test']

    # For DP pretraining, default to using only train nodes
    use_train_nodes_only = getattr(config, 'use_train_nodes_only', True)
    if use_train_nodes_only:
        pretrain_mask = train_mask
        if verbose:
            print(f'Using TRAIN nodes only for DP pretraining ({train_mask.sum().item()} nodes)')
    else:
        # Use all nodes (not recommended for DP)
        pretrain_mask = torch.ones(num_nodes, dtype=torch.bool)
        if verbose:
            print(f'WARNING: Using ALL nodes for pretraining (not privacy-safe)')

    pretrain_indices = torch.where(pretrain_mask)[0]
    num_pretrain_nodes = len(pretrain_indices)

    if verbose:
        print(f'Num pretraining nodes: {num_pretrain_nodes}')
        print(f'Num total nodes: {num_nodes}')
        print(f'Feature dim: {data.x.shape[1]}')

    # Move data to device
    data_x = data.x.to(device)
    labels = labels.to(device)
    pretrain_mask = pretrain_mask.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    pretrain_indices = pretrain_indices.to(device)

    # Create model
    input_dim = data_x.shape[1]
    model = create_pretraining_model(config, input_dim).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f'Model parameters: {num_params}')

    # DP settings
    noise_multiplier = getattr(config, 'noise_multiplier', 1.0)
    l2_norm_clip_percentile = getattr(config, 'l2_norm_clip_percentile', 75.0)
    num_estimation_samples = getattr(config, 'num_estimation_samples', 10000)
    max_epsilon = getattr(config, 'max_epsilon', None)
    target_delta = 1.0 / (10 * num_pretrain_nodes)

    if verbose:
        print(f'DP settings: noise_multiplier={noise_multiplier}, '
              f'clip_percentile={l2_norm_clip_percentile}, target_delta={target_delta:.2e}')

    # Estimate clipping thresholds
    dp_gen = torch.Generator()
    dp_gen.manual_seed(config.rng_seed + 1)

    num_estimation_samples = min(num_estimation_samples, num_pretrain_nodes)
    estimation_indices = pretrain_indices[:num_estimation_samples]
    if verbose:
        print(f'Estimating clipping thresholds from {len(estimation_indices)} samples...')

    l2_norms_threshold = _estimate_clipping_thresholds(
        model, data_x, labels, estimation_indices, l2_norm_clip_percentile)

    if verbose:
        print(f'Clipping thresholds: {l2_norms_threshold}')

    # Optimizer (standard optimizer, we'll manually apply noisy gradients)
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=getattr(config, 'weight_decay', 0.0),
        )
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=getattr(config, 'momentum', 0.9),
            weight_decay=getattr(config, 'weight_decay', 0.0),
        )
    else:
        raise ValueError(f'Unsupported optimizer: {config.optimizer}')

    # Privacy accountant (standard DP-SGD for MLP)
    sampling_probability = config.batch_size / num_pretrain_nodes
    privacy_accountant_fn = lambda num_steps: privacy_accountants.dpsgd_privacy_accountant(
        num_training_steps=num_steps,
        noise_multiplier=noise_multiplier,
        target_delta=target_delta,
        sampling_probability=sampling_probability,
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epsilon': [],
    }

    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    steps_taken = 0
    epsilon_exceeded = False

    for epoch in range(config.num_epochs):
        if epsilon_exceeded:
            break

        model.train()

        # Shuffle pretraining indices
        perm = torch.randperm(num_pretrain_nodes, device=device)
        epoch_indices = pretrain_indices[perm]

        epoch_loss = 0.0
        num_batches = 0

        # Mini-batch training with DP-SGD
        for i in range(0, num_pretrain_nodes, config.batch_size):
            batch_idx = epoch_indices[i:i + config.batch_size]

            # Compute clipped and summed gradients
            clipped_sum = _clip_and_accumulate_gradients(
                model, data_x, labels, batch_idx, l2_norms_threshold)

            # Add noise for DP
            noisy_grads = dp_optimizers.dp_aggregate(
                {k: v.unsqueeze(0) for k, v in clipped_sum.items()},  # Add batch dim for dp_aggregate
                l2_norms_threshold,
                base_sensitivity=1.0,  # MLP has sensitivity 1.0
                noise_multiplier=noise_multiplier,
                generator=dp_gen,
            )

            # Average over batch size
            batch_size_actual = len(batch_idx)
            averaged_grads = {k: v / batch_size_actual for k, v in noisy_grads.items()}

            # Apply gradients
            optimizer.zero_grad()
            for name, param in model.named_parameters():
                if name in averaged_grads:
                    param.grad = averaged_grads[name]
            optimizer.step()

            # Compute batch loss for logging (noisy, but approximate)
            with torch.no_grad():
                batch_data = Data(x=data_x[batch_idx], edge_index=None)
                batch_data.num_nodes = len(batch_idx)
                logits = model(batch_data).x
                loss = F.cross_entropy(logits, labels[batch_idx])
                epoch_loss += loss.item()

            num_batches += 1
            steps_taken += 1

            # Check privacy budget
            current_epsilon = privacy_accountant_fn(steps_taken)

            if max_epsilon is not None and current_epsilon > max_epsilon:
                if verbose:
                    print(f'\nStopping: epsilon ({current_epsilon:.2f}) exceeded max ({max_epsilon:.2f})')
                epsilon_exceeded = True
                break

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # Evaluation
        if (epoch + 1) % config.evaluate_every_epochs == 0 or epoch == config.num_epochs - 1 or epsilon_exceeded:
            model.eval()
            with torch.no_grad():
                # Full graph forward
                full_data = Data(x=data_x, edge_index=None)
                full_data.num_nodes = num_nodes
                logits = model(full_data).x

                # Compute metrics on all splits
                train_loss, train_acc = compute_metrics(logits, labels, train_mask)
                val_loss, val_acc = compute_metrics(logits, labels, val_mask)
                test_loss, test_acc = compute_metrics(logits, labels, test_mask)

                current_epsilon = privacy_accountant_fn(steps_taken)

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['test_loss'].append(test_loss)
                history['test_acc'].append(test_acc)
                history['epsilon'].append(current_epsilon)

                if verbose:
                    print(f'Epoch {epoch + 1}/{config.num_epochs} '
                          f'(steps={steps_taken}, ε={current_epsilon:.2f}): '
                          f'train_loss={avg_train_loss:.4f}, '
                          f'train_acc={train_acc*100:.2f}%, '
                          f'val_acc={val_acc*100:.2f}%, '
                          f'test_acc={test_acc*100:.2f}%')

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            final_epsilon = privacy_accountant_fn(steps_taken) if steps_taken > 0 else 0.0
            print(f'\nLoaded best model with val_acc={best_val_acc*100:.2f}%, ε={final_epsilon:.2f}')

    # Final evaluation
    model.eval()
    with torch.no_grad():
        full_data = Data(x=data_x, edge_index=None)
        full_data.num_nodes = num_nodes
        logits = model(full_data).x

        final_train_loss, final_train_acc = compute_metrics(logits, labels, train_mask)
        final_val_loss, final_val_acc = compute_metrics(logits, labels, val_mask)
        final_test_loss, final_test_acc = compute_metrics(logits, labels, test_mask)

    final_epsilon = privacy_accountant_fn(steps_taken) if steps_taken > 0 else 0.0

    if verbose:
        print(f'\nFinal Results (ε={final_epsilon:.2f}):')
        print(f'  Train: loss={final_train_loss:.4f}, acc={final_train_acc*100:.2f}%')
        print(f'  Val:   loss={final_val_loss:.4f}, acc={final_val_acc*100:.2f}%')
        print(f'  Test:  loss={final_test_loss:.4f}, acc={final_test_acc*100:.2f}%')

    history['final_train_acc'] = final_train_acc
    history['final_val_acc'] = final_val_acc
    history['final_test_acc'] = final_test_acc
    history['final_epsilon'] = final_epsilon
    history['steps_taken'] = steps_taken

    return model, history


def save_pretrained_mlp_dp(
    model: models.GraphMultiLayerPerceptron,
    path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """Save DP-pretrained MLP for later use in GCN initialization.

    Args:
        model: Trained MLP model.
        path: Save path.
        metadata: Optional metadata (e.g., performance metrics, epsilon).
    """
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    state_dict = model.mlp.state_dict()
    checkpoint = {
        'model_state_dict': state_dict,
        'metadata': metadata or {},
    }
    torch.save(checkpoint, path)
    print(f'Saved DP-pretrained MLP to {path}')
