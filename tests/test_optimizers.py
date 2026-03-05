"""Tests for optimizers module.

Mirrors the reference tests:
  1. No clipping + no noise should recover sum of gradients.
  2. Clipping norms should match analytical values.
  3. Noise std should match l2_clip * base_sensitivity * noise_multiplier.
"""

import pytest
import torch
import numpy as np

from dp_gnn.optimizers import clip_by_norm, dp_aggregate


def _build_per_example_grads(batch_size=8):
    """Example i's grads are all i (includes 0 to test no div-by-zero)."""
    params = {
        'param0.w': torch.zeros(2, 3, 4),
        'param0.b': torch.zeros(1,),   # scalar-like but with batch dim
        'param1': torch.zeros(6, 7),
    }
    a = torch.arange(batch_size, dtype=torch.float32)
    per_eg = {}
    for name, p in params.items():
        # shape: [B, *p.shape]
        per_eg[name] = (a.reshape(-1, *([1]*p.dim())) *
                        torch.ones(batch_size, *p.shape))
    return params, per_eg


class TestClipByNorm:
    @pytest.mark.parametrize("clip", [0.5, 10.0, 20.0, 40.0, 80.0])
    def test_norms_are_bounded(self, clip):
        _, per_eg = _build_per_example_grads()
        thresholds = {n: clip for n in per_eg}
        clipped = clip_by_norm(per_eg, thresholds)
        for name, g in clipped.items():
            B = g.shape[0]
            norms = torch.linalg.norm(g.reshape(B, -1), dim=1)
            assert torch.all(norms <= clip + 1e-5)

    def test_no_clip_when_below_threshold(self):
        _, per_eg = _build_per_example_grads()
        large_clip = 1e30
        thresholds = {n: large_clip for n in per_eg}
        clipped = clip_by_norm(per_eg, thresholds)
        for name in per_eg:
            assert torch.allclose(clipped[name], per_eg[name])


class TestDpAggregate:
    def test_no_privacy_recovers_sum(self):
        """No clipping + no noise => sum of per-example grads."""
        _, per_eg = _build_per_example_grads()
        thresholds = {n: float('inf') for n in per_eg}
        gen = torch.Generator()
        gen.manual_seed(42)
        result = dp_aggregate(per_eg, thresholds,
                              base_sensitivity=1.0,
                              noise_multiplier=0.0,
                              generator=gen)
        for name, g in per_eg.items():
            expected = g.sum(dim=0)
            assert torch.allclose(result[name], expected, atol=1e-5)

    @pytest.mark.parametrize("clip", [0.5, 10.0, 40.0])
    def test_clipping_values(self, clip):
        params, per_eg = _build_per_example_grads()
        thresholds = {n: clip for n in per_eg}
        gen = torch.Generator()
        gen.manual_seed(42)
        result = dp_aggregate(per_eg, thresholds,
                              base_sensitivity=1.0,
                              noise_multiplier=0.0,
                              generator=gen)

        batch_size = 8
        a = torch.arange(batch_size, dtype=torch.float32)
        for name, p_shape in [(n, per_eg[n].shape[1:]) for n in per_eg]:
            flat_dim = int(np.prod(p_shape))
            norms = a * np.sqrt(flat_dim)
            divisors = torch.clamp(norms / clip, min=1.0)
            expected_val = (a / divisors).sum()
            expected = expected_val * torch.ones(p_shape)
            assert torch.allclose(result[name], expected, rtol=1e-5, atol=1e-5), \
                f"Mismatch for {name}"

    @pytest.mark.parametrize("clip,base_sens,noise_mult", [
        (3.0, 1.0, 2.0),
        (1.0, 3.0, 5.0),
        (100.0, 2.0, 4.0),
        (1.0, 5.0, 90.0),
    ])
    def test_noise_std(self, clip, base_sens, noise_mult):
        """Noise std should be clip * base_sensitivity * noise_multiplier."""
        expected_std = clip * base_sens * noise_mult
        gen = torch.Generator()
        gen.manual_seed(42)

        per_eg = {'w': torch.ones(1, 100, 100)}  # batch=1
        thresholds = {'w': clip}

        all_updates = []
        for _ in range(3):
            result = dp_aggregate(per_eg, thresholds,
                                  base_sensitivity=base_sens,
                                  noise_multiplier=noise_mult,
                                  generator=gen)
            all_updates.append(result['w'])

        for upd in all_updates:
            actual_std = upd.std().item()
            assert abs(actual_std - expected_std) < 0.1 * expected_std, \
                f"Expected std ~{expected_std}, got {actual_std}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
