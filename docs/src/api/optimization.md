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
