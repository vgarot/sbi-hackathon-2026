# Sylvain Doute — Analytic multi-stage simulator for planetary remote sensing (photometric models)

## Short description
Collection of forward models used in planetary remote sensing to map physical surface parameters (e.g. roughness)
to photometric reflectivity measurements under different illumination/observation geometries.

Most models are analytic; one includes numerical integrations. The simulator is designed for high-dimensional outputs
(tens to hundreds of measurement components) and fast simulation times (from milliseconds to ~1 second).

The goal is to solve an inverse problem: infer physical parameters from photometric observations.

## Parameters of interest
- Typical number of parameters: **L = 3 to 6**
- Current applications:
  - **L = 4**, **D = 71**
  - **L = 4**, **D = 11**

## Simulator output
- Photometric measurement vector **Y ∈ R^D**
- Dimension **D** is typically in the **tens to hundreds**

## Simulation runtime
- From **milliseconds** to **~1 second** depending on the model

## Inference / learning approach used so far
- Bayesian inverse problem handled with an adapted version of **Gaussian Locally Linear Mapping (GLLiM)**
- Interested in investigating **neural SBI methods** on the same problem

## Code availability
- Main repository:
  - Planet-GLLiM: https://gitlab.inria.fr/xllim/planet-gllim
- Base package:
  - xllim: https://gitlab.inria.fr/xllim/xllim
- Forward models can be provided as:
  - a simple **Python class** (single `.py` file), e.g. `HapkeModelPython.py`
  - or as a more complex module

## Documentation
- Context / fundamentals:
  - https://xllim.gitlabpages.inria.fr/planet-gllim/rst/scientific_doc/fundamentals/index.html
- Photometric forward models:
  - https://xllim.gitlabpages.inria.fr/planet-gllim/rst/scientific_doc/photometric_models/index.html