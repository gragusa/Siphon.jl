# Siphon.jl

A Julia package for Linear State Space Models with Kalman filtering, smoothing, and estimation.

## Features

- **High-performance Kalman filter/smoother**: AD-compatible implementations for maximum likelihood estimation
- **Missing data handling**: Seamless handling of missing observations via NaN
- **DSL for model specification**: Ergonomic domain-specific language for defining state-space models
- **Pre-built templates**: Local level, local linear trend, AR(1), ARMA, Dynamic Nelson-Siegel, and dynamic factor models
- **Multiple estimation methods**:
  - Maximum Likelihood via numerical optimization (Optimization.jl)
  - EM algorithm with automatic backend selection
- **Bayesian support**: Prior specification and LogDensityProblems.jl interface for MCMC
- **StaticArrays support**: Automatic conversion for small models (dimensions ≤ 13)
- **In-place implementations**: Zero-allocation filter/smoother for large-scale applications

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/gragusa/Siphon.jl")
```

## Quick Start

### Local Level Model (Random Walk + Noise)

```julia
using Siphon

# Specify model
spec = local_level()

# Load data (p × n matrix, observations × time)
y = reshape(nile_data, 1, :)

# Estimate via MLE
result = optimize_ssm(spec, y)
println("Estimated parameters: ", result.θ)
println("Log-likelihood: ", result.loglik)

# Or use EM algorithm
em_result = em_ssm(spec, y; maxiter=200, verbose=true)
```

### Custom State-Space Model

```julia
using Siphon

# Local linear trend model
# State: [level, slope]
# y_t = level_t + ε_t
# level_{t+1} = level_t + slope_t + η_level,t
# slope_{t+1} = slope_t + η_slope,t

spec = custom_ssm(
    Z = [1.0 0.0],                           # Observation matrix
    H = [FreeParam(:var_obs, init=100.0, lower=0.0)],  # Obs variance
    T = [1.0 1.0; 0.0 1.0],                  # Transition matrix
    R = [1.0 0.0; 0.0 1.0],                  # Selection matrix
    Q = diag_free(2, :var_state, init=10.0), # State variance (diagonal)
    a1 = [0.0, 0.0],                         # Initial state mean
    P1 = 1e7 * identity_mat(2)               # Diffuse initialization
)

result = optimize_ssm(spec, y)
```

### Dynamic Factor Model

```julia
using Siphon
include("src/inplace.jl")

# Create DFM with 6 factors, VAR(3) dynamics, and AR(1) errors
model = DynamicFactorModel(
    n_obs,              # Number of observable series
    6,                  # Number of latent factors
    n_times;            # Number of time periods
    factor_lags = 3,    # VAR(3) factor dynamics
    error_lags = 1      # AR(1) idiosyncratic errors
)

# Estimate via EM
fit!(EM(), model, y; maxiter=500, verbose=true)

# Access estimated factors and parameters
f = factors(model)           # k × n smoothed factors
Λ = loadings(model)          # Factor loadings
Φ = var_coefficients(model)  # VAR coefficients
δ = ar_coefficients(model)   # AR error coefficients
```

### Filtering and Smoothing

```julia
using Siphon

# Build state-space representation
ss = build_linear_state_space(spec, result.θ, y)

# Run filter
filt = kalman_filter(ss.p, y, ss.a1, ss.P1)
println("Log-likelihood: ", filt.loglik)
println("Filtered states: ", filt.att)

# Run smoother
smooth = kalman_smoother(ss.p.Z, ss.p.T, filt.at, filt.Pt, filt.vt, filt.Ft)
println("Smoothed states: ", smooth.alpha)
```

## State-Space Model Formulation

The package uses the standard linear Gaussian state-space form:

```
Observation equation:  y_t = Z α_t + ε_t,    ε_t ~ N(0, H)
State equation:        α_{t+1} = T α_t + R η_t,  η_t ~ N(0, Q)
Initial conditions:    α_1 ~ N(a_1, P_1)
```

Where:
- `y_t` is the p×1 observation vector
- `α_t` is the m×1 state vector
- `Z` is the p×m observation matrix
- `H` is the p×p observation covariance
- `T` is the m×m transition matrix
- `R` is the m×r selection matrix
- `Q` is the r×r state covariance

## Pre-built Model Templates

| Function | Model | States | Parameters |
|----------|-------|--------|------------|
| `local_level()` | Random walk + noise | 1 | var_obs, var_level |
| `local_linear_trend()` | Random walk with drift | 2 | var_obs, var_level, var_slope |
| `ar1()` | AR(1) + noise | 1 | ρ, var_obs, var_state |
| `arma(p, q)` | ARMA(p,q) | max(p,q+1) | AR and MA coefficients, variances |
| `dns_model()` | Dynamic Nelson-Siegel | 3 | λ, AR coefficients, variances |
| `dynamic_factor()` | Dynamic factor model | k×lags | Loadings, VAR coefficients, variances |

## Examples

See `EXAMPLES.md` for detailed documentation of all examples, including mathematical formulations and data descriptions.

| Example | Model | Data |
|---------|-------|------|
| `nile_example.jl` | Local Level | Nile River annual flow |
| `dns_estimation.jl` | Dynamic Nelson-Siegel | Simulated yield curves |
| `dns_estimation_dsl.jl` | Dynamic Nelson-Siegel (DSL) | Simulated yield curves |
| `dfm_qt_estimation.jl` | Dynamic Factor Model | FRED-QD macro data |
| `dfm_full_estimation.jl` | Dynamic Factor Model | FRED-QD macro data |

```bash
# Run examples
julia --project examples/nile_example.jl
julia --project examples/dns_estimation.jl
julia --project examples/dfm_full_estimation.jl
```

## Documentation

Full documentation is available in the `docs/` folder. Build locally:

```bash
cd docs && julia --project -e 'include("make.jl")'
```

## Testing

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run specific test tags
TI_TAGS=filter,smoother julia --project -e 'using Pkg; Pkg.test()'
```

## Architecture

```
src/
├── Siphon.jl          # Main module and exports
├── types.jl           # KFParms, StaticArrays utilities
├── filter_ad.jl       # AD-compatible Kalman filter
├── smoother_ad.jl     # AD-compatible RTS smoother
├── predict.jl         # Forecasting and missing data utilities
├── inplace.jl         # In-place implementations for large models
│                      # (KalmanWorkspace, EMWorkspace, StateSpaceModel,
│                      #  DynamicFactorModel)
└── dsl/
    ├── dsl.jl         # DSL submodule entry point
    ├── types.jl       # SSMSpec, SSMParameter, etc.
    ├── codegen.jl     # Build KFParms from SSMSpec
    ├── templates.jl   # Pre-built models
    ├── builder.jl     # custom_ssm function
    ├── matrix_helpers.jl  # diag_free, cov_free, etc.
    ├── expressions.jl # Parameter-dependent matrix expressions
    ├── bayesian.jl    # Priors and LogDensityProblems interface
    ├── optimization.jl # optimize_ssm via Optimization.jl
    └── em.jl          # EM algorithm
```

## Dependencies

- `StaticArrays` - Fixed-size arrays for small models
- `ForwardDiff` - Automatic differentiation for MLE
- `TransformVariables` - Parameter constraint handling
- `LogDensityProblems` - Bayesian inference interface
- `Optimization` / `OptimizationOptimJL` - Numerical optimization

## References

- Durbin, J. & Koopman, S.J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.
- Shumway, R.H. & Stoffer, D.S. (2017). *Time Series Analysis and Its Applications*. Springer.
- Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.

## License

MIT License
