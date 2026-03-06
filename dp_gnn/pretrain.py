"""MLP pretraining module for node classification.

Provides standard (non-DP) training of MLP on node features.
Can use full graph or just training split.
"""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from dp_gnn import input_pipeline
from dp_gnn import models
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


def pretrain_mlp(
    config,
    verbose: bool = True,
) -> Tuple[models.GraphMultiLayerPerceptron, Dict]:
    """Pretrain an MLP on node features.

    Args:
        config: Configuration object with pretraining settings.
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

    # For pretraining, optionally use full graph
    if getattr(config, 'use_full_graph', True):
        # Use all nodes for training
        pretrain_mask = torch.ones(num_nodes, dtype=torch.bool)
    else:
        pretrain_mask = train_mask

    pretrain_indices = torch.where(pretrain_mask)[0]
    num_pretrain_nodes = len(pretrain_indices)

    if verbose:
        print(f'Num pretraining nodes: {num_pretrain_nodes}')
        print(f'Num total nodes: {num_nodes}')
        print(f'Feature dim: {data.x.shape[1]}')

    # Move data to device
    data = data.to(device)
    labels = labels.to(device)
    pretrain_mask = pretrain_mask.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    pretrain_indices = pretrain_indices.to(device)

    # Create model
    input_dim = data.x.shape[1]
    model = create_pretraining_model(config, input_dim).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f'Model parameters: {num_params}')

    # Optimizer
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

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    # Training loop
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(config.num_epochs):
        model.train()

        # Shuffle pretraining indices
        perm = torch.randperm(num_pretrain_nodes, device=device)
        epoch_indices = pretrain_indices[perm]

        epoch_loss = 0.0
        num_batches = 0

        # Mini-batch training
        for i in range(0, num_pretrain_nodes, config.batch_size):
            batch_idx = epoch_indices[i:i + config.batch_size]

            # Forward pass
            batch_data = Data(x=data.x[batch_idx], edge_index=None)
            batch_data.num_nodes = len(batch_idx)
            logits = model(batch_data).x

            # Compute loss
            batch_labels = labels[batch_idx]
            loss = F.cross_entropy(logits, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # Evaluation
        if (epoch + 1) % config.evaluate_every_epochs == 0 or epoch == config.num_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Full graph forward
                full_data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
                full_data.num_nodes = num_nodes
                logits = model(full_data).x

                # Compute metrics on all splits
                train_loss, train_acc = compute_metrics(logits, labels, train_mask)
                val_loss, val_acc = compute_metrics(logits, labels, val_mask)
                test_loss, test_acc = compute_metrics(logits, labels, test_mask)

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['test_loss'].append(test_loss)
                history['test_acc'].append(test_acc)

                if verbose:
                    print(f'Epoch {epoch + 1}/{config.num_epochs}: '
                          f'train_loss={avg_train_loss:.4f}, '
                          f'train_acc={train_acc*100:.2f}%, '
                          f'val_acc={val_acc*100:.2f}%, '
                          f'test_acc={test_acc*100:.2f}%')

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Save checkpoint periodically
        if (getattr(config, 'save_checkpoint', False) and
            config.checkpoint_every_epochs > 0 and
            (epoch + 1) % config.checkpoint_every_epochs == 0):
            checkpoint_dir = getattr(config, 'checkpoint_dir', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'pretrain_epoch_{epoch+1}.pt')
            save_checkpoint(
                model,
                checkpoint_path,
                metadata={
                    'epoch': epoch + 1,
                    'val_acc': val_acc if 'val_acc' in dir() else 0.0,
                    'config': {k: v for k, v in config.__dict__.items()},
                },
                optimizer=optimizer,
                epoch=epoch + 1,
            )
            if verbose:
                print(f'  Saved checkpoint to {checkpoint_path}')

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f'\nLoaded best model with val_acc={best_val_acc*100:.2f}%')

    # Final evaluation
    model.eval()
    with torch.no_grad():
        full_data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        full_data.num_nodes = num_nodes
        logits = model(full_data).x

        final_train_loss, final_train_acc = compute_metrics(logits, labels, train_mask)
        final_val_loss, final_val_acc = compute_metrics(logits, labels, val_mask)
        final_test_loss, final_test_acc = compute_metrics(logits, labels, test_mask)

    if verbose:
        print(f'\nFinal Results:')
        print(f'  Train: loss={final_train_loss:.4f}, acc={final_train_acc*100:.2f}%')
        print(f'  Val:   loss={final_val_loss:.4f}, acc={final_val_acc*100:.2f}%')
        print(f'  Test:  loss={final_test_loss:.4f}, acc={final_test_acc*100:.2f}%')

    history['final_train_acc'] = final_train_acc
    history['final_val_acc'] = final_val_acc
    history['final_test_acc'] = final_test_acc

    return model, history


def save_pretrained_mlp(
    model: models.GraphMultiLayerPerceptron,
    path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """Save pretrained MLP for later use in GCN initialization.

    Args:
        model: Trained MLP model.
        path: Save path.
        metadata: Optional metadata (e.g., performance metrics).
    """
    # Save the inner MLP's state dict (without 'mlp.' prefix)
    # This makes transfer to GCN easier
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    state_dict = model.mlp.state_dict()
    checkpoint = {
        'model_state_dict': state_dict,
        'metadata': metadata or {},
    }
    torch.save(checkpoint, path)
    print(f'Saved pretrained MLP to {path}')
