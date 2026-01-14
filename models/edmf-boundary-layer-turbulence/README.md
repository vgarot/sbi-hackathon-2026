# Manolis Perrot — 1D PDE / EDMF model for boundary-layer turbulence

## Short description
Forward model based on a 1D PDE approach (EDMF framework) for boundary-layer turbulence.
The simulator produces a spatiotemporal output (1D space + time), and can also be reduced to a
single scalar metric of interest for inference.

## Parameters of interest
- Number of parameters to infer: **9**

## Simulator output
- Spatiotemporal field: **f(x, t)** (1D space + time)
- Can be reduced to a **single scalar metric** of interest

## Simulation runtime
- Approximately **0.2 s** per simulation

## Code availability
- GitHub repository: https://github.com/ManolisPerrot/edmf_estimation

## References
- Main article: https://doi.org/10.1029/2024MS004616