# Estimation & Bayesian

This page documents functions for parameter estimation and Bayesian inference.

## High-Level Unified API

Siphon.jl provides a unified API centered around `StateSpaceModel`. There are two ways to create a model:

### Option 1: Create Model with Known Parameters

If you already have parameter values (from prior knowledge, simulation, or external estimation):

```julia
using Siphon

spec = local_level()
θ = (var_obs=15099.0, var_level=1469.0)

# Create model with known parameters
model = StateSpaceModel(spec, θ, n_obs)

# Run filter and smoother
ll = kalman_filter!(model, y)
kalman_smoother!(model)

# Access results
loglikelihood(model)        # Log-likelihood
filtered_states(model)      # E[αₜ|y₁:ₜ]
smoothed_states(model)      # E[αₜ|y₁:ₙ]
parameters(model)           # Parameter NamedTuple
```

### Option 2: Estimate Parameters

If you want to estimate parameters from data:

```julia
using Siphon

spec = local_level()
model = StateSpaceModel(spec, n_obs)

# Maximum likelihood estimation
fit!(MLE(), model, y)

# Or EM algorithm estimation
fit!(EM(), model, y; maxiter=200, tol=1e-6, verbose=true)

# Access results
loglikelihood(model)        # Log-likelihood at optimum
parameters(model)           # Fitted parameters as NamedTuple
isconverged(model)          # Whether estimation converged
niterations(model)          # Number of iterations (EM only)

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

## StateSpaceModel Constructors

```@docs
Siphon.StateSpaceModel
```

## Estimation Methods

```@docs
Siphon.MLE
Siphon.EM
Siphon.fit!
```

## Unified Filter/Smoother Methods

These methods work directly on `StateSpaceModel` objects:

```julia
# Log-likelihood (doesn't modify model state)
ll = kalman_loglik(model, y)

# Filter (stores results in model)
ll = kalman_filter!(model, y)

# Smoother (uses stored filter results)
kalman_smoother!(model)
```

```@docs
Siphon.kalman_loglik(::Siphon.StateSpaceModel, ::AbstractMatrix)
Siphon.kalman_filter!(::Siphon.StateSpaceModel, ::AbstractMatrix)
Siphon.kalman_smoother!(::Siphon.StateSpaceModel)
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

Siphon.jl uses what MARSS calls the **`tinitx = 1`** convention: the pair
`(a₁, P₁)` is the prior distribution of the state at the *first observation
time*, so the first measurement update is `y₁ - Z·a₁`.

For a longer treatment with worked examples, see the
[Initial State Tutorial](../tutorials/initial_state.md).

### How `(a₁, P₁)` are set

You never pass `a1`/`P1` to `fit!`. They are part of the spec, and the
spec carries them into the model:

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

Both estimation paths consume the spec's `(a₁, P₁)`. They do *not* expose
any `tinitx`, `V0`, `x0`, or `update_initial_state` keyword.

| Path | Initial state behaviour |
|---|---|
| `fit!(EM(), model, y)` | `(a₁, P₁)` from the spec are copied into the workspace once and held **fixed** for every EM iteration. Even if `a1` or `P1` contain `FreeParam`s, the EM M-step does not currently update them. |
| `fit!(MLE(), model, y)` | `(a₁, P₁)` are rebuilt from the parameter vector at every objective call. So **`FreeParam`s inside `a1` / `P1` are estimated** alongside `Z, H, T, Q` parameters. |

### Migrating from MARSS

MARSS specifies `(x₀, V₀)` at time `t = 0` by default (`tinitx = 0`). To get
the same prior at `t = 1` in Siphon, propagate one step manually before
constructing the spec:

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
