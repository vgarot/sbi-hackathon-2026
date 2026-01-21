# Vadim Betrand — PASTAX: SDE/ODE simulator for ocean-surface drifting trajectories (JAX)

## Short description
Simulator based on stochastic differential equations (SDEs) to generate Lagrangian trajectories of drifting objects
at the ocean surface. Starting from initial conditions and forcing fields, the model simulates an ensemble of
trajectories over multiple days, sampled at hourly resolution.

The simulator can also run deterministically (ODE mode) to generate a single trajectory instead of an ensemble.

## Parameters of interest
- Number of parameters to infer: typically **~5 to 10**
  - parameterization is flexible depending on the chosen SDE form
- Notes:
  - some parameters may not be identifiable
  - parameters can have physical constraints (e.g. positive, bounded, circular, ...)

## Simulator output
- Output: **N trajectories** over **D days**, sampled **every hour**
- Output dimension: **N × D × 24**
- Typical values:
  - **N = 50**
  - **D = 5** (or **7**) days

## Simulation runtime
- Implemented in **JAX**, parallelized over ensemble members
- After compilation: **< 1 second** (depends on ensemble size)
- Runs on **CPU or GPU**
  - GPU is beneficial for large **N** or many initial conditions / forcings

## Code availability
- GitHub repository: https://github.com/vadmbertr/pastax
- Documentation (Getting started): https://pastax.readthedocs.io/en/latest/getting_started/

## Notes
- Can be used either as:
  - **stochastic simulator** (SDE, ensemble trajectories)
  - **deterministic simulator** (ODE, single trajectory)
