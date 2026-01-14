# Vincent Garot — Sequence alignments simulator (Birth–Death trees + AliSim)

## Short description
Simulator pipeline to generate sequence alignments by first sampling transmission trees from a Birth–Death model, then generating alignments with AliSim. The model output is high-dimensional and variable-size, so simulations are padded and embedded via a learned summary network for SBI.

## Parameters of interest
- 2 to 4 parameters (exact set depends on the configuration)

## Simulator output
- Sequence alignment data with variable dimension:
  - depends on number of sequences
  - depends on alignment length (number of nucleotides)

## Current preprocessing / representation
- Data padded into a fixed-size tensor:
  - **100 × 250 × 8**
- A learned summary network maps this tensor to a **64-dimensional** embedding vector

## SBI / inference approach
- Large-scale simulation-based inference performed manually:
  - ~**1.5 million** simulated alignments
- A deep neural network is trained to predict **posterior quantiles**

## Code availability
- Main repository (GitLab):
  - https://gitlab.in2p3.fr/vincent.garot/teddy_official
- Simulator available as:
  - a compiled version taking a JSON configuration file (not parameter-based yet)
  - alternative implementation in `generation_cpp/`

## Notes / limitations
- Simulator interface currently relies on JSON input (may be adapted to accept direct parameters if needed)
- Second implementation exists but may be less convenient to use

## References
- Preprint: https://www.biorxiv.org/content/10.64898/2026.01.05.697728v1