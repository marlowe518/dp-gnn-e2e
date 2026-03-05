"""DP-SGD / DP-Adam optimizers for per-example gradients.

Reproduces the reference JAX/optax implementation:
  1. clip_by_norm: per-example, per-layer L2 norm clipping
  2. dp_aggregate: clip -> sum -> add noise
  3. dpsgd / dpadam: dp_aggregate chained with SGD / Adam

In PyTorch we implement dp_aggregate as a callable transform on the
per-example gradient dict, then feed the noisy aggregated gradients
to a standard optimizer.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def clip_by_norm(
    per_example_grads: Dict[str, torch.Tensor],
    l2_norms_threshold: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """Clip each per-example gradient by its per-layer L2 norm threshold.

    Args:
        per_example_grads: dict mapping param name -> tensor of shape [B, *param_shape].
        l2_norms_threshold: dict mapping param name -> scalar clip threshold.

    Returns:
        Clipped per-example gradients (same structure).
    """
    clipped = {}
    for name, grad in per_example_grads.items():
        batch_size = grad.shape[0]
        flat = grad.reshape(batch_size, -1)
        norms = torch.linalg.norm(flat, dim=1)  # [B]
        clip_val = l2_norms_threshold[name]
        divisors = torch.clamp(norms / clip_val, min=1.0)  # [B]
        # Reshape divisors for broadcasting
        shape = [batch_size] + [1] * (grad.dim() - 1)
        clipped[name] = grad / divisors.reshape(shape)
    return clipped


def dp_aggregate(
    per_example_grads: Dict[str, torch.Tensor],
    l2_norms_threshold: Dict[str, float],
    base_sensitivity: float,
    noise_multiplier: float,
    generator: torch.Generator,
) -> Dict[str, torch.Tensor]:
    """Aggregate per-example gradients with DP guarantees.

    1. Clip per-example gradients.
    2. Sum across the batch.
    3. Add Gaussian noise scaled to clip * base_sensitivity * noise_multiplier.
    """
    clipped = clip_by_norm(per_example_grads, l2_norms_threshold)

    noisy_grads = {}
    for name, grad in clipped.items():
        summed = grad.sum(dim=0)
        noise_std = l2_norms_threshold[name] * base_sensitivity * noise_multiplier
        if noise_std > 0 and np.isfinite(noise_std):
            # Generator is always CPU; generate noise on CPU then move
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=summed.shape,
                generator=generator,
                dtype=summed.dtype,
            ).to(summed.device)
            noisy_grads[name] = summed + noise
        else:
            noisy_grads[name] = summed
    return noisy_grads
