"""
Trial-Based Data Simulators for SBI Tutorial
=============================================

This module contains simple simulators for learning about IID trial-based
inference (multiple observations per parameter set).

The simulators are intentionally simple (Linear Gaussian) so students can
focus on the inference methodology rather than the complexity of the simulator.
"""

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from sbi.utils import BoxUniform


# =============================================================================
# Trial-Based Linear Gaussian Simulator
# =============================================================================

def linear_gaussian_simulator(
    theta: Tensor,
    noise_std: float = 0.5,
) -> Tensor:
    """Simulate a single observation from a Linear Gaussian model.

    This is the simplest possible simulator: observations are drawn from
    a Gaussian centered on the parameters with fixed noise.

    x | θ ~ N(θ, σ²I)

    Args:
        theta: Parameters of shape (batch_size, dim) or (dim,)
        noise_std: Standard deviation of observation noise

    Returns:
        Observations of shape (batch_size, dim)
    """
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    noise = torch.randn_like(theta) * noise_std
    x = theta + noise

    return x


def create_trial_data_prior(dim: int = 2) -> BoxUniform:
    """Create a prior distribution for the Linear Gaussian model.

    Args:
        dim: Dimensionality of the parameter space

    Returns:
        BoxUniform prior over [-2, 2]^dim
    """
    lower_bound = torch.full((dim,), -2.0)
    upper_bound = torch.full((dim,), 2.0)
    return BoxUniform(low=lower_bound, high=upper_bound)


def simulate_iid_trials(
    theta: Tensor,
    num_trials: int,
    noise_std: float = 0.5,
) -> Tensor:
    """Simulate multiple IID trials for a single parameter set.

    Given parameters θ, generates num_trials independent observations:
    x_i | θ ~ N(θ, σ²I) for i = 1, ..., num_trials

    Args:
        theta: Parameters of shape (dim,) for a single parameter set
        num_trials: Number of IID trials to generate
        noise_std: Standard deviation of observation noise

    Returns:
        Observations of shape (num_trials, dim)
    """
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    # Repeat theta for each trial
    theta_repeated = theta.repeat(num_trials, 1)

    # Generate independent noise for each trial
    noise = torch.randn_like(theta_repeated) * noise_std
    x = theta_repeated + noise

    return x


def generate_trial_training_data(
    prior: BoxUniform,
    num_parameter_sets: int,
    max_num_trials: int,
    noise_std: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Generate training data for NPE with permutation-invariant embedding.

    For each parameter set, generates data with varying numbers of trials
    (from 1 to max_num_trials). Missing trials are padded with NaN.

    This is necessary for training NPE to be amortized over the number of trials.

    Args:
        prior: Prior distribution to sample parameters from
        num_parameter_sets: Number of unique parameter sets to sample
        max_num_trials: Maximum number of trials per parameter set
        noise_std: Standard deviation of observation noise

    Returns:
        Tuple of (theta, x) where:
            - theta: Parameters of shape (num_parameter_sets * max_num_trials, dim)
            - x: Observations of shape (num_parameter_sets * max_num_trials, max_num_trials, dim)
              with NaN padding for missing trials
    """
    dim = prior.sample().shape[0]
    theta_list = []
    x_list = []

    # Sample parameter sets
    theta_base = prior.sample((num_parameter_sets,))

    for i in range(num_parameter_sets):
        # Generate all trials for this parameter set
        all_trials = simulate_iid_trials(theta_base[i], max_num_trials, noise_std)

        # Create training examples with varying numbers of trials
        for num_trials in range(1, max_num_trials + 1):
            # Pad with NaN for missing trials
            x_padded = torch.ones(max_num_trials, dim) * float("nan")
            x_padded[:num_trials, :] = all_trials[:num_trials, :]

            theta_list.append(theta_base[i])
            x_list.append(x_padded)

    theta = torch.stack(theta_list)
    x = torch.stack(x_list)

    return theta, x


def true_posterior_linear_gaussian(
    x_o: Tensor,
    noise_std: float = 0.5,
    prior_mean: Tensor | None = None,
    prior_std: float = 2.0,
) -> MultivariateNormal:
    """Compute the analytical posterior for the Linear Gaussian model.

    For a Gaussian prior N(μ₀, Σ₀) and Gaussian likelihood N(θ, σ²I),
    the posterior given observations X = {x₁, ..., xₙ} is also Gaussian.

    Args:
        x_o: Observations of shape (num_trials, dim)
        noise_std: Standard deviation of observation noise
        prior_mean: Prior mean (default: zeros)
        prior_std: Prior standard deviation (same for all dims, uniform prior approximation)

    Returns:
        MultivariateNormal distribution representing the analytical posterior
    """
    if x_o.ndim == 1:
        x_o = x_o.unsqueeze(0)

    num_trials = x_o.shape[0]
    dim = x_o.shape[1]

    if prior_mean is None:
        prior_mean = torch.zeros(dim)

    # Posterior precision = prior precision + n * likelihood precision
    prior_precision = 1.0 / (prior_std ** 2)
    likelihood_precision = 1.0 / (noise_std ** 2)
    posterior_precision = prior_precision + num_trials * likelihood_precision
    posterior_variance = 1.0 / posterior_precision

    # Posterior mean = posterior_variance * (prior_precision * prior_mean + likelihood_precision * sum(x))
    data_sum = x_o.sum(dim=0)
    posterior_mean = posterior_variance * (
        prior_precision * prior_mean + likelihood_precision * data_sum
    )

    # Return as MultivariateNormal
    posterior_cov = torch.eye(dim) * posterior_variance
    return MultivariateNormal(loc=posterior_mean, covariance_matrix=posterior_cov)
