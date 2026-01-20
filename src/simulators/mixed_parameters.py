"""
Mixed Parameter Type Simulator for SBI Tutorial
================================================

This module contains a simple simulator for learning about MNPE
(Mixed Neural Posterior Estimation) which handles both continuous
and discrete parameters.
"""

import numpy as np
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, Bernoulli
from sbi.utils import MultipleIndependent


def create_mixed_prior(
    dim_continuous: int = 1,
    dim_discrete: int = 2,
) -> MultipleIndependent:
    """Create a prior with both continuous and discrete parameters.

    IMPORTANT: For MNPE, continuous parameters must come first!

    Args:
        dim_continuous: Number of continuous parameters
        dim_discrete: Number of binary discrete parameters

    Returns:
        MultipleIndependent prior combining Gaussian and Bernoulli distributions
    """
    continuous_prior = MultivariateNormal(
        torch.zeros(dim_continuous),
        torch.eye(dim_continuous),
    )

    discrete_priors = [Bernoulli(0.5 * torch.ones(1)) for _ in range(dim_discrete)]

    return MultipleIndependent([continuous_prior, *discrete_priors])


def mixed_simulator(
    theta: Tensor,
    sigma_continuous: float = 0.1,
    sigma_discrete: float = 0.01,
) -> Tensor:
    """Simulate observations from a mixed continuous-discrete model.

    The simulator generates 2 observations:
    - x₁: Continuous parameter + Gaussian noise
    - x₂: Sum of discrete parameters + small Gaussian noise

    This allows us to infer both types of parameters from the data.

    Args:
        theta: Parameters of shape (batch_size, dim_total) or (dim_total,)
               where dim_total = dim_continuous + dim_discrete
               Continuous parameters must come first!
        sigma_continuous: Noise std for continuous observation
        sigma_discrete: Noise std for discrete observation sum

    Returns:
        Observations of shape (batch_size, 2)
    """
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    batch_size = theta.shape[0]
    dim_continuous = 1  # We use 1 continuous parameter

    # Continuous part: x₁ = θ_continuous + noise
    x_continuous = (
        theta[:, :dim_continuous]
        + torch.randn(batch_size, dim_continuous) * sigma_continuous
    )

    # Discrete part: x₂ = sum of discrete parameters + small noise
    # This makes the sum of discrete parameters identifiable
    discrete_sum = theta[:, dim_continuous:].sum(dim=1, keepdim=True)
    x_discrete = discrete_sum + torch.randn(batch_size, 1) * sigma_discrete

    return torch.cat([x_continuous, x_discrete], dim=1)


def analytical_mixed_posterior(
    x_o: Tensor,
    sigma_continuous: float = 0.1,
    sigma_discrete: float = 0.01,
    num_samples: int = 1000,
) -> Tensor:
    """Sample from the analytical posterior for the mixed model.

    The posterior factorizes:
    - Continuous: Gaussian posterior (conjugate)
    - Discrete: Categorical over {(0,0), (0,1), (1,0), (1,1)} based on likelihood

    Args:
        x_o: Observation of shape (1, 2) or (2,)
        sigma_continuous: Noise std used in simulator
        sigma_discrete: Noise std used in simulator
        num_samples: Number of posterior samples to generate

    Returns:
        Posterior samples of shape (num_samples, 3)
    """
    if x_o.ndim == 1:
        x_o = x_o.unsqueeze(0)

    x1_obs = x_o[0, 0].item()  # Continuous observation
    x2_obs = x_o[0, 1].item()  # Discrete sum observation

    # Posterior for continuous parameter: Gaussian conjugate update
    # Prior: N(0, 1), Likelihood: N(θ, σ²)
    prior_precision = 1.0
    likelihood_precision = 1.0 / (sigma_continuous**2)
    posterior_precision = prior_precision + likelihood_precision
    posterior_variance = 1.0 / posterior_precision
    posterior_mean = posterior_variance * (likelihood_precision * x1_obs)

    theta1_samples = (
        torch.randn(num_samples) * np.sqrt(posterior_variance) + posterior_mean
    )

    # Posterior for discrete parameters: enumerate all combinations
    combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    log_probs = torch.tensor([
        -(x2_obs - (d1 + d2)) ** 2 / (2 * sigma_discrete**2)
        for d1, d2 in combinations
    ])
    probs = torch.softmax(log_probs, dim=0)

    # Sample discrete combinations
    indices = torch.distributions.Categorical(probs).sample((num_samples,))
    theta2_samples = torch.tensor(
        [combinations[idx][0] for idx in indices], dtype=torch.float32
    )
    theta3_samples = torch.tensor(
        [combinations[idx][1] for idx in indices], dtype=torch.float32
    )

    return torch.stack([theta1_samples, theta2_samples, theta3_samples], dim=1)
