# Siphon.jl

*A Julia package for Linear State Space Models with Kalman filtering and smoothing.*

## Installation

Siphon.jl is not yet registered in the General registry. Install it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/gragusa/Siphon.jl")
```

Siphon.jl requires Julia 1.13 or later.

## Overview

Siphon.jl filters, smooths, forecasts and estimates linear Gaussian state space
models, and nonlinear-measurement models through the extended Kalman filter.
Models are written as matrices with free parameters marked, or taken from
templates, and are fitted by maximum likelihood or EM.

### Features

- **Kalman filter and smoother**: scalar and matrix observations, compatible
  with automatic differentiation; in-place workspaces that do not allocate
  across repeated runs.
- **Initialization**: approximate diffuse (large `P1`) or exact diffuse
  (Durbin and Koopman) initialization.
- **Smoothers**: state smoother, disturbance smoother with auxiliary residuals,
  and simulation smoother for drawing state paths.
- **Missing data**: `missing` or `NaN` entries, including periods in which only
  some series are observed.
- **Model specification**: `custom_ssm` with `FreeParam` entries, matrix
  helpers, and templates for the local level, local linear trend, AR(1),
  ARMA(p, q), dynamic factor and dynamic Nelson-Siegel models.
- **Estimation**: maximum likelihood through Optimization.jl
  (`fit!(MLE(), …)`) and the EM algorithm (`fit!(EM(), …)`), with standard
  errors from the observed information; profile EM for the Nelson-Siegel decay
  parameter.
- **Extended Kalman filter**: nonlinear measurement equations with analytic,
  automatic-differentiation or finite-difference Jacobians, with filter,
  smoother and estimation.
- **Dynamic factor models**: `DynamicFactorModel` for many series, with VAR
  factors and AR idiosyncratic errors, estimated by EM.
- **Forecasting**: point forecasts with covariances, intervals and simulated
  paths.
- **Bayesian inference**: priors and a LogDensityProblems.jl log-density for
  MCMC samplers.
- **StatsAPI**: `coef`, `vcov`, `stderror`, `loglikelihood`, `aic`, `bic` and
  related functions.
- **Plotting**: Plots.jl recipes for filter, smoother and forecast results.

## State Space Model Formulation

Siphon.jl implements the standard linear Gaussian state space model:

```math
\begin{aligned}
y_t &= Z \alpha_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, H) \\
\alpha_{t+1} &= T \alpha_t + R \eta_t, \quad \eta_t \sim N(0, Q)
\end{aligned}
```

where:
- ``y_t`` is the ``p \times 1`` observation vector
- ``\alpha_t`` is the ``m \times 1`` state vector
- ``Z`` is the ``p \times m`` observation matrix
- ``H`` is the ``p \times p`` observation error covariance
- ``T`` is the ``m \times m`` transition matrix
- ``R`` is the ``m \times r`` selection matrix
- ``Q`` is the ``r \times r`` state innovation covariance

## Quick Start

```julia
using Siphon

# Create a local level model specification
spec = local_level(var_obs = FreeParam(init = 225.0), var_level = FreeParam(init = 100.0))

# Your data (p × T matrix, where p = number of series, T = time periods)
y = randn(1, 100)

# Create model and estimate parameters
model = StateSpaceModel(spec, size(y, 2))
fit!(MLE(), model, y)           # Maximum likelihood
# or: fit!(EM(), model, y)      # EM algorithm

# Access results
println("Log-likelihood: ", loglikelihood(model))
println("Parameters: ", parameters(model))

# Alternative: use optimize_ssm directly
result = optimize_ssm(spec, y)
println("Estimated parameters: ", result.θ)
```

## Documentation Structure

**Tutorials**

- [Getting Started](tutorials/getting_started.md): filtering, smoothing and estimation of a local level model
- [Custom Models](tutorials/custom_models.md): specifying models with `custom_ssm` and the matrix helpers
- [Dynamic Factor Models](tutorials/dynamic_factor.md): the `dynamic_factor` template and `DynamicFactorModel`
- [ARMA Models](tutorials/arma_models.md): ARMA(p, q) models in state space form
- [Estimation Methods](tutorials/estimation_methods.md): maximum likelihood, EM and profile EM
- [Parameter Transformations](tutorials/transformations.md): bounds and unconstrained parameterization
- [Initial State Conventions](tutorials/initial_state.md): `a1`, `P1`, exact diffuse initialization and MARSS conventions
- [Extended Kalman Filter](tutorials/ekf.md): nonlinear measurement equations
- [Visualization](tutorials/visualization.md): plotting filter, smoother and forecast results

**API Reference**

- [Core Functions](api/core.md): functional filter, smoothers, forecasting and missing-data utilities
- [DSL & Templates](api/dsl.md): model specification types and templates
- [Matrix Helpers](api/matrix_helpers.md): building parameterized matrices
- [Estimation & Bayesian](api/optimization.md): `StateSpaceModel`, `fit!`, accessors, priors and log-densities
- [In-Place Workspaces](api/inplace.md): preallocated filter, smoother and simulation smoother
- [Dynamic Factor Models](api/dfm.md): `DynamicFactorModel` and its accessors
- [Extended Kalman Filter](api/ekf.md): EKF types and specification

[Conversion Guide](conversion_guide.md): changes from Siphon 0.1 and how to update code.

## Related Packages

- [StateSpaceModels.jl](https://github.com/LAMPSPUC/StateSpaceModels.jl): Alternative state space modeling package
- [Optimization.jl](https://github.com/SciML/Optimization.jl): Unified optimization interface used by Siphon.jl
- [TransformVariables.jl](https://github.com/tpapp/TransformVariables.jl): Parameter transformations used internally
- [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl): Interface for Bayesian inference

## License

Siphon.jl is released under the MIT License.
