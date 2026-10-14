# Siphon.jl

[![CI](https://github.com/gragusa/Siphon.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/gragusa/Siphon.jl/actions/workflows/ci.yml) [![codecov.io](http://codecov.io/github/gragusa/Siphon.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/Siphon.jl?branch=master) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) ![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)

Linear State Space Models in Julia: Kalman filtering, smoothing, and estimation.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/gragusa/Siphon.jl")
```

Requires Julia 1.13 or later.

## Features

- Kalman filter and smoother for scalar and matrix observations, compatible
  with automatic differentiation, plus in-place workspaces for large models
- Approximate and exact diffuse initialization
- Disturbance smoother, auxiliary residuals and simulation smoother
- Missing observations (`missing` or `NaN`), including partially observed periods
- Model specification with `custom_ssm` and `FreeParam`, and templates for the
  local level, local linear trend, AR(1), ARMA(p, q), dynamic factor and
  dynamic Nelson-Siegel models
- Estimation by maximum likelihood (`fit!(MLE(), …)`) or EM (`fit!(EM(), …)`),
  with standard errors through the StatsAPI interface
- Extended Kalman filter and smoother for nonlinear measurement equations
- `DynamicFactorModel` for large panels, estimated by EM
- Forecasts with covariances, intervals and simulated paths
- Priors and a LogDensityProblems.jl log-density for Bayesian estimation
- Plots.jl recipes for filter, smoother and forecast results

## Example

```julia
using Siphon

y = randn(1, 100) .+ cumsum(randn(1, 100); dims = 2)   # 1 series × 100 periods
model = StateSpaceModel(local_level(), size(y, 2))
fit!(MLE(), model, y)

parameters(model)          # (var_obs = …, var_level = …)
smoothed_states(model)     # 1 × 100
forecast(model, 12)        # forecasts of periods 101–112
```

## Documentation

See `docs/` for the manual (build with `julia --project=docs docs/make.jl`) and
`examples/` for worked examples. Users of Siphon 0.1 should read
`docs/src/conversion_guide.md`.

## License

MIT
