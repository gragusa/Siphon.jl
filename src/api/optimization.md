# Estimation & Bayesian

This page documents functions for parameter estimation and Bayesian inference.

## High-Level Estimation API

The recommended API for parameter estimation uses `fit!` with method markers:

```julia
using Siphon

# Create model from specification
spec = local_level()
model = StateSpaceModel(spec, n_obs)

# Maximum likelihood estimation
fit!(MLE(), model, y)

# EM algorithm estimation
fit!(EM(), model, y; maxiter=200, tol=1e-6, verbose=true)

# Access results
loglikelihood(model)        # Log-likelihood at optimum
parameters(model)           # Fitted parameters as NamedTuple
model.theta_values          # Fitted parameter vector (internal)
model.converged             # Whether estimation converged

# Access system matrices at fitted parameters
mats = system_matrices(model)  # All matrices as NamedTuple
mats.Z  # Observation matrix
mats.H  # Observation covariance
mats.T  # Transition matrix
mats.R  # Selection matrix
mats.Q  # State covariance

# Or access individually
obs_matrix(model)        # Z
obs_cov(model)           # H
transition_matrix(model) # T
selection_matrix(model)  # R
state_cov(model)         # Q

# Access filter/smoother results
filtered_states(model)     # E[αₜ|y₁:ₜ]
smoothed_states(model)     # E[αₜ|y₁:ₙ] (computed on demand)
prediction_errors(model)   # yₜ - E[yₜ|y₁:ₜ₋₁]
```

```@docs
Siphon.MLE
Siphon.EM
Siphon.StateSpaceModel
Siphon.fit!
```

## Model Accessors

After fitting a `StateSpaceModel`, use these functions to access results:

### Parameters

```@docs
Siphon.parameters
Siphon.loglikelihood
```

### System Matrices

```@docs
Siphon.system_matrices
Siphon.obs_matrix
Siphon.obs_cov
Siphon.transition_matrix
Siphon.selection_matrix
Siphon.state_cov
```

### Filter and Smoother Results

```@docs
Siphon.filtered_states
Siphon.predicted_states
Siphon.smoothed_states
Siphon.prediction_errors
```

## Direct Optimization

For more control, use `optimize_ssm` directly:

```@docs
Siphon.DSL.optimize_ssm
Siphon.DSL.optimize_ssm_with_stderr
```

## Profile EM for DNS Models

For Dynamic Nelson-Siegel models with nonlinear λ parameter:

```@docs
Siphon.DSL.profile_em_ssm
Siphon.DSL.ProfileEMResult
```

## Initial State Conventions and MARSS Compatibility

Siphon.jl internally uses the `tinitx=1` convention where `(a₁, P₁)` represents the initial state distribution at time t=1. Both `profile_em_ssm` and `DynamicFactorModel` support the `tinitx` parameter to control how the initial state covariance is computed.

For a comprehensive explanation of initial state conventions, see the [Initial State Tutorial](@ref initial_state).

### The `tinitx` Parameter

The `tinitx` parameter controls when the initial state covariance `V0` is defined:

| Setting | V0 Interpretation | P1 Computation |
|---------|------------------|----------------|
| `tinitx=0` (default) | Covariance at t=0 | `P1 = T × V0 × T' + R × Q × R'` |
| `tinitx=1` | Covariance at t=1 | `P1 = V0` (no transformation) |

With `tinitx=0`, the initial covariance incorporates one step of state dynamics, matching MARSS's default behavior.

### The `V0` Parameter

The `V0` parameter specifies the initial state covariance value:

- **Scalar:** `V0=100.0` creates `V0 × I` (identity scaled by V0)
- **Matrix:** Can also pass a full covariance matrix (for `profile_em_ssm`)

**Default:** `V0=100.0` (MARSS default)

### Usage Examples

```julia
# DNS models via profile_em_ssm
result = profile_em_ssm(spec, y; tinitx=0, V0=100.0)  # Default: MARSS-style
result = profile_em_ssm(spec, y; tinitx=1, V0=1e7)    # Diffuse prior at t=1

# Dynamic Factor Models
model = DynamicFactorModel(N, k, n; tinitx=0, V0=100.0)  # Default: MARSS-style
model = DynamicFactorModel(N, k, n; tinitx=1, V0=1e7)    # Diffuse prior at t=1
```

### Choosing `tinitx` and `V0`

| Use Case | Recommended Setting |
|----------|---------------------|
| Match MARSS default | `tinitx=0, V0=100.0` |
| Diffuse prior (large uncertainty) | `tinitx=1, V0=1e7` |
| Informative prior at t=1 | `tinitx=1, V0=<your value>` |
| Short time series | `tinitx=0` (accounts for dynamics) |

**Note:** With `tinitx=0`, very large `V0` values (e.g., 1e7) may cause numerical instability because the transformation `T × V0 × T'` amplifies values. Use `tinitx=1` with large `V0` for diffuse priors.

### Initial State Updating in EM (`update_initial_state`)

By default, `(a₁, P₁)` remains **fixed** throughout EM iterations. Set `update_initial_state=true` to update them at each M-step using smoothed state estimates:

```julia
# For profile EM (DNS models)
result = profile_em_ssm(spec, y; update_initial_state=true)

# For high-level API
fit!(EM(), model, y; update_initial_state=true)
```

**How it works:**

| EM Iteration | With `update_initial_state=false` (default) | With `update_initial_state=true` |
|--------------|---------------------------------------------|----------------------------------|
| 0 (start)    | `a1`, `P1` from `tinitx`/`V0` | `a1`, `P1` from `tinitx`/`V0` |
| 1            | Same `a1`, `P1` | Updated from smoother |
| 2            | Same `a1`, `P1` | Updated from smoother |
| ...          | Same `a1`, `P1` | Updated from smoother |

When `update_initial_state=true`, after each E-step:
```
a₁_new = E[α₁ | y₁:n]      (smoothed state mean at t=1)
P₁_new = Var[α₁ | y₁:n]    (smoothed state covariance at t=1)
```

**When to use `update_initial_state=true`:**
- Short time series where t=1 significantly affects the likelihood
- Comparing results with MARSS R package
- Estimating the unconditional mean/variance of the state process

**When to keep fixed (default):**
- Using a diffuse prior
- Long time series where initial state has negligible effect
- Numerical stability concerns

The final initial state estimates are returned in `EMResult.a1` and `EMResult.P1`.

## Parameter Transformations

```@docs
Siphon.DSL.build_transformation
Siphon.DSL.transform_to_constrained
Siphon.DSL.transform_to_unconstrained
```

## Log-Density Interface

```@docs
Siphon.DSL.SSMLogDensity
Siphon.DSL.logdensity
```

## Prior Distributions

```@docs
Siphon.DSL.FlatPrior
Siphon.DSL.NormalPrior
Siphon.DSL.InverseGammaPrior
Siphon.DSL.CompositePrior
```
