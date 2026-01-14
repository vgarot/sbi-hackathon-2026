# Gabriel Mouttapa — Tunax: differentiable vertical ocean physics calibration (Python)

## Short description
Differentiable Python model for calibrating parameters of vertical ocean physics against observational or simulated databases.
The simulator produces multivariate spatiotemporal outputs and can be used for SBI on a moderately high-dimensional parameter space.

## Parameters of interest
- Number of parameters to infer: **17**

## Simulator output
- Output shape: **time (~1000) × 1D (64) × 4 variables**
  - i.e. 4 physical variables evolving over time and a 1D vertical dimension

## Simulation runtime
- Approximately **0.1 s** per forward simulation

## Code availability
- GitHub repository: https://github.com/meom-group/tunax