"""Privacy accountants for different training schemes.

Uses the dp_accounting library (same as reference) to compute
epsilon given training parameters.
"""

import functools

import dp_accounting
import numpy as np
import scipy.special
import scipy.stats


def multiterm_dpsgd_privacy_accountant(
    num_training_steps: int,
    noise_multiplier: float,
    target_delta: float,
    num_samples: int,
    batch_size: int,
    max_terms_per_node: int,
) -> float:
    """Computes epsilon for DP-SGD with multi-term sensitivity amplification.

    Accounts for the exact hypergeometric distribution of affected terms
    in a minibatch (sampling without replacement).
    """
    if noise_multiplier < 1e-20:
        return np.inf

    terms_rv = scipy.stats.hypergeom(num_samples, max_terms_per_node, batch_size)
    terms_logprobs = [
        terms_rv.logpmf(i) for i in np.arange(max_terms_per_node + 1)
    ]

    orders = np.arange(1, 10, 0.1)[1:]

    accountant = dp_accounting.rdp.RdpAccountant(orders)
    accountant.compose(dp_accounting.GaussianDpEvent(noise_multiplier))
    unamplified_rdps = accountant._rdp  # pylint: disable=protected-access

    amplified_rdps = []
    for order, unamplified_rdp in zip(orders, unamplified_rdps):
        beta = unamplified_rdp * (order - 1)
        log_fs = beta * (
            np.square(np.arange(max_terms_per_node + 1) / max_terms_per_node)
        )
        amplified_rdp = scipy.special.logsumexp(terms_logprobs + log_fs) / (
            order - 1
        )
        amplified_rdps.append(amplified_rdp)

    amplified_rdps = np.asarray(amplified_rdps)
    if not np.all(
        unamplified_rdps * (batch_size / num_samples) ** 2
        <= amplified_rdps + 1e-6
    ):
        raise ValueError('The lower bound has been violated. Something is wrong.')

    amplified_rdps_total = amplified_rdps * num_training_steps

    return dp_accounting.rdp.compute_epsilon(
        orders, amplified_rdps_total, target_delta
    )[0]


def dpsgd_privacy_accountant(
    num_training_steps: int,
    noise_multiplier: float,
    target_delta: float,
    sampling_probability: float,
) -> float:
    """Computes epsilon for standard DP-SGD (single affected term per node)."""
    if noise_multiplier < 1e-20:
        return np.inf

    orders = np.arange(1, 200, 0.1)[1:]
    event = dp_accounting.PoissonSampledDpEvent(
        sampling_probability, dp_accounting.GaussianDpEvent(noise_multiplier)
    )
    accountant = dp_accounting.rdp.RdpAccountant(orders)
    accountant.compose(event, num_training_steps)
    return accountant.get_epsilon(target_delta)


def get_training_privacy_accountant(config, num_training_nodes, max_terms_per_node):
    """Returns a callable: num_training_steps -> epsilon."""
    if not config.differentially_private_training:
        return lambda num_training_steps: 0

    if config.model == 'mlp':
        return functools.partial(
            dpsgd_privacy_accountant,
            noise_multiplier=config.training_noise_multiplier,
            target_delta=1 / (10 * num_training_nodes),
            sampling_probability=config.batch_size / num_training_nodes,
        )
    if config.model == 'gcn':
        return functools.partial(
            multiterm_dpsgd_privacy_accountant,
            noise_multiplier=config.training_noise_multiplier,
            target_delta=1 / (10 * num_training_nodes),
            num_samples=num_training_nodes,
            batch_size=config.batch_size,
            max_terms_per_node=max_terms_per_node,
        )

    raise ValueError(f'Could not create privacy accountant for model: {config.model}.')
