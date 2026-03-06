"""Checkpoint utilities for saving and loading model state.

Provides functions to save/load model checkpoints with metadata,
enabling pretrain-finetune workflows.
"""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
) -> None:
    """Save a model checkpoint with optional metadata.

    Args:
        model: The model to save.
        path: Path to save the checkpoint (will create dir if needed).
        metadata: Optional dict with additional info (e.g., epoch, metrics).
        optimizer: Optional optimizer state to save.
        epoch: Optional epoch number.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {},
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Load a checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Optional model to load state into.
        optimizer: Optional optimizer to load state into.
        device: Device to load tensors to.

    Returns:
        Dict containing checkpoint data (state_dicts, metadata, epoch).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def load_model_state(
    path: str,
    device: str = 'cpu',
) -> Dict[str, torch.Tensor]:
    """Load only the model state_dict from a checkpoint.

    Args:
        path: Path to the checkpoint file.
        device: Device to load tensors to.

    Returns:
        Model state_dict.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint.get('model_state_dict', checkpoint)


def get_checkpoint_metadata(path: str) -> Optional[Dict[str, Any]]:
    """Get metadata from a checkpoint without loading the full model.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Metadata dict or None if not present.
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    return checkpoint.get('metadata')


def list_checkpoints(
    directory: str,
    pattern: str = '*.pt',
) -> list:
    """List checkpoint files in a directory.

    Args:
        directory: Directory to search.
        pattern: Glob pattern for checkpoint files.

    Returns:
        List of checkpoint file paths.
    """
    import glob
    return sorted(glob.glob(os.path.join(directory, pattern)))
