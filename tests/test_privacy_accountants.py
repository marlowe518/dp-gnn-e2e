"""Tests for privacy_accountants module.

Mirrors the reference tests:
  1. multiterm epsilon >= standard epsilon when max_terms_per_node=1
  2. Low noise multiplier returns inf.
"""

import pytest
import numpy as np

from dp_gnn.privacy_accountants import (
    dpsgd_privacy_accountant,
    multiterm_dpsgd_privacy_accountant,
)


class TestMultitermVsStandard:
    @pytest.mark.parametrize("num_training_steps", [1, 10])
    @pytest.mark.parametrize("noise_multiplier", [1])
    @pytest.mark.parametrize("target_delta", [1e-5])
    @pytest.mark.parametrize("batch_size", [10, 20])
    @pytest.mark.parametrize("num_samples", [1000, 2000])
    @pytest.mark.parametrize("max_terms_per_node", [1])
    def test_multiterm_geq_standard(
        self, num_training_steps, noise_multiplier,
        target_delta, batch_size, num_samples, max_terms_per_node
    ):
        multiterm_eps = multiterm_dpsgd_privacy_accountant(
            num_training_steps=num_training_steps,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            num_samples=num_samples,
            batch_size=batch_size,
            max_terms_per_node=max_terms_per_node,
        )
        standard_eps = dpsgd_privacy_accountant(
            num_training_steps=num_training_steps,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            sampling_probability=batch_size / num_samples,
        )
        assert standard_eps <= multiterm_eps


class TestLowNoiseMultiplier:
    @pytest.mark.parametrize("noise_multiplier", [-1, 0])
    @pytest.mark.parametrize("target_delta", [1e-5])
    @pytest.mark.parametrize("num_samples", [1000])
    @pytest.mark.parametrize("batch_size", [10])
    @pytest.mark.parametrize("max_terms_per_node", [1])
    def test_multiterm_returns_inf(
        self, noise_multiplier, target_delta, num_samples,
        batch_size, max_terms_per_node
    ):
        eps = multiterm_dpsgd_privacy_accountant(
            num_training_steps=10,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            num_samples=num_samples,
            batch_size=batch_size,
            max_terms_per_node=max_terms_per_node,
        )
        assert eps == np.inf

    @pytest.mark.parametrize("noise_multiplier", [-1, 0])
    @pytest.mark.parametrize("target_delta", [1e-5])
    @pytest.mark.parametrize("sampling_probability", [0.1])
    def test_standard_returns_inf(
        self, noise_multiplier, target_delta, sampling_probability
    ):
        eps = dpsgd_privacy_accountant(
            num_training_steps=10,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            sampling_probability=sampling_probability,
        )
        assert eps == np.inf


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
