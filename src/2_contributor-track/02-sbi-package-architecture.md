# sbi Package Architecture

**Version:** 0.25.0
**Repository:** https://github.com/sbi-dev/sbi

---

## Module Structure

```
sbi/
├── __init__.py              # Main public API exports
├── sbi_types.py             # Type definitions
│
├── inference/               # Core inference algorithms
│   ├── trainers/            # Neural network trainers
│   │   ├── npe/             # Neural Posterior Estimation
│   │   ├── nle/             # Neural Likelihood Estimation
│   │   ├── nre/             # Neural Ratio Estimation
│   │   ├── vfpe/            # Vector Field methods (FMPE, NPSE)
│   │   └── marginal/        # Marginal inference
│   ├── posteriors/          # Posterior wrapper classes
│   ├── potentials/          # Potential function definitions
│   ├── abc/                 # Approximate Bayesian Computation
│   ├── snpe/                # Sequential NPE (aliases)
│   ├── snle/                # Sequential NLE (aliases)
│   └── snre/                # Sequential NRE (aliases)
│
├── neural_nets/             # Neural network components
│   ├── estimators/          # Density/ratio estimators
│   ├── embedding_nets/      # Input embedding networks
│   └── net_builders/        # Network factory functions
│
├── samplers/                # Sampling algorithms
│   ├── mcmc/                # Markov Chain Monte Carlo
│   ├── vi/                  # Variational Inference
│   ├── rejection/           # Rejection sampling
│   ├── importance/          # Importance sampling & SIR
│   ├── score/               # Score-based samplers
│   └── ode_solvers/         # ODE integration
│
├── diagnostics/             # Inference validation
│   ├── sbc.py               # Simulation-Based Calibration
│   ├── tarp.py              # TARP coverage tests
│   ├── lc2st.py             # Local C2ST
│   └── misspecification.py  # Misspecification detection
│
├── analysis/                # Post-inference analysis
│   ├── plot.py              # Visualization (pairplot, etc.)
│   ├── conditional_density.py
│   └── sensitivity_analysis.py
│
├── utils/                   # Utility functions
│   ├── sbiutils.py          # Core SBI utilities
│   ├── torchutils.py        # PyTorch helpers
│   ├── simulation_utils.py  # Simulator wrappers
│   └── ...
│
├── simulators/              # Example simulators
└── examples/                # Usage examples
```

---

## Class Hierarchy

### Inference Trainers

The trainer classes implement the training loop for different SBI methods.

```
NeuralInference (ABC)
    │
    ├── PosteriorEstimatorTrainer
    │       ├── NPE_A (Mixture Density Network)
    │       ├── NPE_B (deprecated)
    │       └── NPE_C (default, normalizing flows)  ← alias: NPE, SNPE
    │
    ├── LikelihoodEstimatorTrainer
    │       ├── NLE_A (normalizing flows)  ← alias: NLE, SNLE
    │       └── MNLE (Mixed data)
    │
    ├── RatioEstimatorTrainer
    │       ├── NRE_A (classifier)
    │       ├── NRE_B (default, classifier)  ← alias: NRE, SNRE
    │       └── NRE_C (contrastive)
    │
    └── VectorFieldTrainer
            ├── FMPE (Flow Matching)
            └── NPSE (Score Estimation)
```

**Key methods:**
- `append_simulations(theta, x)` - Add training data
- `train()` - Train the neural network
- `build_posterior()` - Create a posterior object for sampling

### Posterior Classes

Posterior objects wrap trained networks and provide sampling interfaces.

```
NeuralPosterior (ABC)
    │
    ├── DirectPosterior         # Direct sampling (NPE with flows)
    ├── MCMCPosterior           # MCMC sampling (NLE, NRE)
    ├── VIPosterior             # Variational Inference
    ├── RejectionPosterior      # Rejection sampling
    ├── ImportanceSamplingPosterior  # Importance sampling
    ├── VectorFieldPosterior    # ODE-based sampling (FMPE, NPSE)
    └── EnsemblePosterior       # Ensemble of posteriors
```

**Key methods:**
- `sample(shape, x)` - Draw posterior samples
- `log_prob(theta, x)` - Evaluate log probability
- `map(x)` - Find maximum a posteriori estimate
- `set_default_x(x_o)` - Set default observation

### Density Estimators

Neural networks that estimate conditional densities.

```
ConditionalEstimator (ABC, nn.Module)
    │
    └── ConditionalDensityEstimator (ABC)
            │
            ├── NFlowsFlow       # nflows-based normalizing flow
            ├── ZukoFlow         # zuko-based normalizing flow
            ├── MixedDensityEstimator  # For mixed discrete/continuous
            ├── FlowMatchingEstimator  # Flow matching networks
            └── ScoreEstimator   # Score-based diffusion
```

**Key methods:**
- `sample(num_samples, condition)` - Generate samples
- `log_prob(inputs, condition)` - Evaluate log density
- `loss(inputs, condition)` - Training loss

### Embedding Networks

Networks that transform high-dimensional inputs to lower-dimensional embeddings.

```
nn.Module
    │
    ├── FCEmbedding              # Fully connected (MLP)
    ├── CNNEmbedding             # 1D/2D CNN
    ├── CausalCNNEmbedding       # Causal CNN for time series
    ├── ResNetEmbedding1D        # 1D ResNet
    ├── ResNetEmbedding2D        # 2D ResNet
    ├── TransformerEmbedding     # Transformer encoder
    ├── LRUEmbedding             # Linear Recurrent Unit
    ├── SpectralConvEmbedding    # Spectral convolution
    └── PermutationInvariantEmbedding  # For i.i.d. data
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Prior p(θ)  ──►  θ samples  ──►  Simulator  ──►  x samples        │
│                         │                              │             │
│                         └──────────┬───────────────────┘             │
│                                    ▼                                 │
│                         Training data {(θᵢ, xᵢ)}                    │
│                                    │                                 │
│                                    ▼                                 │
│                    ┌───────────────────────────────┐                │
│                    │     Inference Trainer         │                │
│                    │  (NPE / NLE / NRE / FMPE)     │                │
│                    │                               │                │
│                    │  ┌─────────────────────────┐  │                │
│                    │  │   Embedding Network     │  │                │
│                    │  │   (optional)            │  │                │
│                    │  └──────────┬──────────────┘  │                │
│                    │             ▼                 │                │
│                    │  ┌─────────────────────────┐  │                │
│                    │  │   Density Estimator     │  │                │
│                    │  │   (Flow/MDN/Classifier) │  │                │
│                    │  └─────────────────────────┘  │                │
│                    └───────────────────────────────┘                │
│                                    │                                 │
│                                    ▼                                 │
│                           Trained Network                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PHASE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Observation x_o  ──►  Posterior Object  ──►  θ samples ~ p(θ|x_o) │
│                              │                                       │
│                              ▼                                       │
│                    ┌────────────────────┐                           │
│                    │  Sampling Method   │                           │
│                    │  - Direct (NPE)    │                           │
│                    │  - MCMC (NLE/NRE)  │                           │
│                    │  - VI              │                           │
│                    │  - ODE (FMPE/NPSE) │                           │
│                    └────────────────────┘                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Patterns

### 1. Trainer-Posterior Separation

- **Trainers** handle network training and simulation management
- **Posteriors** handle sampling and probability evaluation
- This separation allows swapping sampling methods without retraining

### 2. Modular Density Estimators

- Estimators are independent of inference method
- Same flow architecture can be used for NPE or NLE
- Factory functions (`posterior_nn`, `likelihood_nn`) handle construction

### 3. Potential-Based Sampling

- NLE/NRE posteriors use "potential functions" for MCMC
- Potential = log_likelihood + log_prior (unnormalized log posterior)
- Enables flexible MCMC sampler choices

### 4. Embedding Network Composition

- Embedding networks preprocess high-dimensional data
- Trained end-to-end with density estimator
- Architecture chosen based on data structure (images → CNN, etc.)

---

## Current Technical Debt / Discussion Points

### API Simplification (#1659)

Current factory functions (`posterior_nn`, `likelihood_nn`, `classifier_nn`) are confusing:

- Proposal: Move to direct estimator instantiation
- Goal: Clearer, more Pythonic API for sbi 1.0

### State Management (#1455)

`LikelihoodEstimator` has inconsistent internal state during sequential training.

### Test Organization (#1429, #1428)

- Tests are slow and could use better fixtures
- Need clearer naming scheme for test files

### Embedding Net Improvements (#1414)

- Some embedding nets lack proper tests
- Need consistent interface across all types

---

## Further Reading

- **API Reference:** https://sbi.readthedocs.io/en/latest/api_reference.html
- **Tutorials:** https://sbi.readthedocs.io/en/latest/tutorials/
- **Paper:** Boelts, Deistler et al. (2024) "sbi reloaded: A toolkit for simulation-based inference workflows"
