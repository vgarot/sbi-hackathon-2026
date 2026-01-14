"""
Lotka-Volterra Predator-Prey Model for SBI Tutorial
====================================================

This simulator models the population dynamics of wolves (predators)
and deer (prey) for environmental monitoring.

The model uses the classic Lotka-Volterra differential equations:
- dDeer/dt = α * Deer - β * Deer * Wolves
- dWolves/dt = δ * Deer * Wolves - γ * Wolves

Parameters:
- α: Deer birth rate
- β: Predation rate
- δ: Wolf efficiency converting deer to wolves
- γ: Wolf death rate
"""

import numpy as np
import torch
from sbi.utils import BoxUniform
from scipy import stats


def lotka_volterra(
    y: np.ndarray, alpha: float, beta: float, delta: float, gamma: float
) -> np.ndarray:
    """Lotka-Volterra differential equations for deer-wolf dynamics."""
    deer, wolves = y
    ddeer_dt = alpha * deer - beta * deer * wolves
    dwolves_dt = delta * deer * wolves - gamma * wolves
    return np.asarray([ddeer_dt, dwolves_dt])


def simulate(parameters: np.ndarray, time_span: float = 200.0) -> np.ndarray:
    """Simulate deer-wolf population dynamics.

    Args:
        parameters: Array of [alpha, beta, delta, gamma] parameters
        time_span: Total simulation time in days (default: 200.0)

    Returns:
        Array of shape (timesteps, 2) with [deer, wolves] populations over time
    """
    alpha, beta, delta, gamma = parameters

    initial_populations = np.asarray([40.0, 9.0])  # [deer, wolves]
    dt = 0.1  # Time step

    timesteps = int(time_span / dt)
    populations = np.zeros((timesteps, 2))
    populations[0] = initial_populations

    for i in range(1, timesteps):
        populations[i] = (
            populations[i - 1]
            + lotka_volterra(populations[i - 1], alpha, beta, delta, gamma) * dt
        )

    return populations


def _get_stats(population: np.ndarray, use_autocorrelation: bool) -> np.ndarray:
    """Calculate summary statistics for a population time series.

    Args:
        population: 1D array of population values over time
        use_autocorrelation: Whether to include autocorrelation in the stats

    Returns:
        1D array of summary statistics
    """
    # 5 moments
    moments = np.array(
        [
            np.mean(population),
            np.std(population),
            np.max(population),
            stats.skew(population),
            stats.kurtosis(population),
        ]
    )

    # 5 normalized autocorrelation lags at specific, spaced-out intervals
    mean_centered_pop = population - np.mean(population)
    autocorr_full = np.correlate(mean_centered_pop, mean_centered_pop, mode="full")

    # The value at lag 0 is the variance of the series.
    lag_0_corr = autocorr_full[autocorr_full.size // 2]

    # Avoid division by zero for constant series.
    if lag_0_corr > 1e-6:
        # Get the second half, normalize by lag 0.
        normalized_autocorr = (autocorr_full / lag_0_corr)[autocorr_full.size // 2 :]

        # Take specific, spaced-out lags to capture longer-term dynamics.
        # These correspond to time delays of 1, 5, 10, 20, and 40 days.
        lags_to_take = [10, 50, 100, 200, 400]
        autocorr = normalized_autocorr[lags_to_take]
    else:
        # If variance is zero, autocorrelation is undefined, return zeros.
        autocorr = np.zeros(5)

    return moments if not use_autocorrelation else np.concatenate([moments, autocorr])


def summarize_simulation(
    simulation_result: np.ndarray, use_autocorrelation: bool = False
) -> np.ndarray:
    """
    Convert simulation to summary statistics with observation noise.

    Calculates stats for each population (deer and wolves):
    - 5 moments: mean, std, max, skewness, kurtosis (always included)
    - 5 autocorrelation lags (optional, controlled by use_autocorrelation)
    """
    # Add observation noise to simulate real-world measurement uncertainty
    noise = np.random.randn(*simulation_result.shape)
    noisy_populations = simulation_result + noise

    deer_pop = noisy_populations[:, 0]
    wolves_pop = noisy_populations[:, 1]

    # --- Calculate and combine stats for both populations ---
    deer_stats = _get_stats(deer_pop, use_autocorrelation)
    wolf_stats = _get_stats(wolves_pop, use_autocorrelation)

    summary = np.concatenate([deer_stats, wolf_stats])
    return summary


def _simulate_and_summarize(params, use_autocorrelation):
    """Helper function for parallel simulation processing."""
    simulation_result = simulate(params)
    summary_stats = summarize_simulation(
        simulation_result, use_autocorrelation=use_autocorrelation
    )
    return summary_stats


def lotka_volterra_simulator(
    params: torch.Tensor | np.ndarray,
    use_autocorrelation: bool = False,
) -> torch.Tensor:
    """SBI-compatible simulator that returns summary statistics.

    Args:
        params: Parameters for the Lotka-Volterra model. Can be:
            - 1D array/tensor of shape (4,) for single simulation (will be batched)
            - 2D array/tensor of shape (batch_size, 4) for batch simulation
        use_autocorrelation: Whether to include autocorrelation in summary statistics
        num_workers: Number of parallel workers for simulation. If 1, runs sequentially.

    Returns:
        Summary statistics tensor of shape (batch_size, n_summary_stats)
    """
    # Convert parameters to numpy array
    if isinstance(params, torch.Tensor):
        params_np = params.detach().cpu().numpy()
    else:
        params_np = np.array(params)

    # Ensure parameters are positive (ecological constraint)
    assert np.all(params_np >= 0), "All LV parameters must be non-negative."

    # Convert single parameter set to batch format
    if params_np.ndim == 1:
        params_np = params_np.reshape(1, -1)
    elif params_np.ndim != 2:
        raise ValueError(
            f"Parameters must be 1D or 2D array, got shape {params_np.shape}"
        )

    # Process batch of parameter sets
    batch_size = params_np.shape[0]

    try:
        batch_summaries = [
            _simulate_and_summarize(params_np[i], use_autocorrelation)
            for i in range(batch_size)
        ]
    except Exception as e:
        raise RuntimeError(f"Simulation failed: {e}") from e

    # Stack all summary statistics into a batch and return
    batch_summaries_array = np.array(batch_summaries)
    return torch.tensor(batch_summaries_array, dtype=torch.float32)


def get_summary_labels(use_autocorrelation: bool = False) -> list:
    """
    Get labels for summary statistics based on configuration.

    Args:
        use_autocorrelation: Whether autocorrelation features are included

    Returns:
        List of labels for all summary statistics
    """
    moment_labels = ["Mean", "Std", "Max", "Skew", "Kurtosis"]
    if use_autocorrelation:
        lags_taken = [10, 50, 100, 200, 400]
        acf_labels = [f"ACF Lag {lag}" for lag in lags_taken]
        stat_labels_per_pop = moment_labels + acf_labels
    else:
        stat_labels_per_pop = moment_labels

    all_labels = [f"Deer {label}" for label in stat_labels_per_pop] + [
        f"Wolf {label}" for label in stat_labels_per_pop
    ]

    return all_labels


def create_lotka_volterra_prior() -> BoxUniform:
    """Create a prior distribution for the Lotka-Volterra parameters."""

    lower_bound = torch.as_tensor([0.05, 0.01, 0.005, 0.005])
    upper_bound = torch.as_tensor([0.15, 0.03, 0.03, 0.15])
    prior = BoxUniform(low=lower_bound, high=upper_bound)

    return prior


def generate_observed_data(
    seed: int = 2025, use_autocorrelation: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the observed data for the wolf-deer scenario.

    This function simulates the "true" observed data that participants analyze,
    representing summary statistics collected by environmental agencies.

    Args:
        seed: Random seed for reproducibility
        use_autocorrelation: Whether to include autocorrelation in the summary statistics

    Returns:
        tuple: (observed_data, true_params) where:
            - observed_data: Summary statistics tensor
            - true_params: The true parameters used to generate the data
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)

    # True parameters for the wolf-deer system
    # [α (deer birth), β (predation), δ (wolf efficiency), γ (wolf death)]
    true_params = torch.tensor([0.1, 0.02, 0.01, 0.1])

    # Generate the observed summary statistics
    observed_data = lotka_volterra_simulator(
        true_params, use_autocorrelation=use_autocorrelation
    )

    return observed_data, true_params


def get_lv_summary_stats_names(use_autocorrelation: bool) -> list[str]:
    """Get the names of the summary statistics for the Lotka-Volterra model."""

    deer_names = ["Deer Mean", "Deer Std", "Deer Max", "Deer Skew", "Deer Kurtosis"]
    wolf_names = ["Wolf Mean", "Wolf Std", "Wolf Max", "Wolf Skew", "Wolf Kurtosis"]

    if use_autocorrelation:
        deer_names += [f"Deer corr lag {i + 1}" for i in range(5)]
        wolf_names += [f"Wolf corr lag {i + 1}" for i in range(5)]

    return deer_names + wolf_names
