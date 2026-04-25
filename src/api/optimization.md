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

`(a₁, P₁)` is the prior distribution of the state at the *first observation
time*: `α₁ ~ N(a₁, P₁)`, and the filter's first measurement update is
`y₁ - Z·a₁`. This is what MARSS calls the `tinitx = 1` convention.

For a longer treatment with worked examples, see the
[Initial State Conventions tutorial](@ref initial_state).

### How `(a₁, P₁)` are set

You never pass `a1`/`P1` to `fit!`. They live in the spec, and the spec
carries them into the model:

| Spec source | `a₁` | `P₁` |
|---|---|---|
| `local_level()`, `ar1()` (default `diffuse=true`) | `[0]` | `[1e7]` |
| `local_level(diffuse=false)` etc. | `[0]` | `[1e4]` |
| `local_level(diffuse=:exact)` etc. | `[0]` | `P₁_star = 0`, `P₁_inf = 1` |
| `local_linear_trend()`, `arma(p,q)` (default) | zeros | `1e7 · I` |
| `custom_ssm(...; a1, P1)` | required argument | required argument |

If a `FreeParam` appears inside `a1` or `P1` of a `custom_ssm`, that
element becomes a free parameter of the model. By default the templates
use `FixedValue`s, so the initial state is structurally fixed.

### Behaviour under `fit!`

| Path | Initial state behaviour |
|---|---|
| `fit!(EM(), model, y)` | `(a₁, P₁)` from the spec are copied into the workspace once and held **fixed** for every EM iteration. `FreeParam`s in `a1`/`P1` are *not* updated by EM. |
| `fit!(MLE(), model, y)` | `(a₁, P₁)` are rebuilt from the parameter vector at every objective call. **`FreeParam`s inside `a1` / `P1` are estimated** alongside `Z, H, T, Q` parameters. |

### Migrating from MARSS

MARSS specifies `(x₀, V₀)` at time `t = 0` by default (`tinitx = 0`). To
get the same prior at `t = 1` in Siphon, propagate one step manually:

```julia
# MARSS tinitx=0 with x0, V0:
a1 = T_init * x0
P1 = T_init * V0 * T_init' + R * Q_init * R'

spec = custom_ssm(Z=..., H=..., T=T_init, R=R, Q=Q_init, a1=a1, P1=P1)
```

For `tinitx = 1` (MARSS's other mode), `a1 = x0` and `P1 = V0` directly.

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
