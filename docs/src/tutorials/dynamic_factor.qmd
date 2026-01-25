# Dynamic Factor Models

This tutorial covers dynamic factor models (DFMs) in Siphon.jl. DFMs are widely used in macroeconomics, finance, and other fields to extract common latent factors from a large panel of observed time series.

Siphon.jl provides two approaches:
1. **DSL approach** (`dynamic_factor` template): Flexible specification with MLE estimation
2. **High-level API** (`DynamicFactorModel`): Specialized type with EM estimation for large panels

## Model Overview

### Mathematical Formulation

The general dynamic factor model is:

```math
\begin{aligned}
X_t &= \Lambda(L) f_t + e_t \\
f_t &= \Psi(L) f_{t-1} + \eta_t, \quad \eta_t \sim N(0, \Sigma_\eta) \\
e_{it} &= \delta(L) e_{i,t-1} + v_{it}, \quad v_{it} \sim N(0, \sigma^2_i)
\end{aligned}
```

Where:
- ``X_t`` is the ``N \times 1`` vector of observables
- ``f_t`` is the ``k \times 1`` vector of latent factors
- ``\Lambda(L) = \Lambda_0 + \Lambda_1 L + \cdots`` are dynamic factor loadings
- ``\Psi(L)`` is the factor VAR polynomial
- ``\delta(L)`` captures AR dynamics in idiosyncratic errors

## High-Level API: DynamicFactorModel

For large-scale applications, use the `DynamicFactorModel` type with EM estimation:

```julia
using Siphon
include("src/inplace.jl")

# Create model: 100 observables, 6 factors, 200 time periods
model = DynamicFactorModel(
    100,              # N: number of observables
    6,                # k: number of factors
    200;              # n: number of time periods
    factor_lags = 3,  # VAR(3) factor dynamics
    error_lags = 1    # AR(1) idiosyncratic errors
)

# Fit with EM algorithm
fit!(EM(), model, y; maxiter=500, tol=1e-6, verbose=true)

# Check convergence
println("Converged: ", isconverged(model))
println("Log-likelihood: ", loglikelihood(model))

# Access results
f = factors(model)              # k × n smoothed factors
Λ = loadings(model)             # Factor loadings [Λ₀, Λ₁, ...]
Φ = var_coefficients(model)     # VAR coefficients [Φ₁, ..., Φ_q]
δ = ar_coefficients(model)      # AR error coefficients
Σ_η = innovation_cov(model)     # Factor innovation covariance
σ²_v = idiosyncratic_variances(model)  # Idiosyncratic variances
```

### Model Configurations

| Configuration | loading_lags | factor_lags | error_lags | Description |
|--------------|--------------|-------------|------------|-------------|
| Simple DFM | 0 | q | 0 | Static loadings, VAR(q) factors |
| AR errors | 0 | q | r | Static loadings, AR(r) errors |
| Dynamic loadings | p | q | 0 | Dynamic loadings with p lags |
| Full DFM | p | q | r | Dynamic loadings + AR errors |

### State Dimension

The state vector stacks current and lagged factors (and errors if r > 0):

```math
\alpha_t = [f_t', f_{t-1}', \ldots, f_{t-s+1}', e_t', e_{t-1}', \ldots, e_{t-r+1}']'
```

where ``s = \max(q, p+1)``. State dimension: ``m = k \times s + N \times r``.

## DSL Approach: dynamic_factor Template

For smaller models or when you need MLE with AD:

```julia
using Siphon

# 8 observables, 2 factors, AR(1) dynamics
spec = dynamic_factor(8, 2)

println("Parameters: ", param_names(spec))
println("Number of states: ", spec.n_states)

# Estimate via MLE
result = optimize_ssm(spec, y)
```

### Identification

For identification, the first ``k`` rows of ``\Lambda_0`` form a lower triangular structure:
- ``\lambda_{i,i} = 1`` for ``i \leq k``
- ``\lambda_{i,j} = 0`` for ``i < j \leq k``

```math
\Lambda_0 = \begin{bmatrix}
1 & 0 \\
\lambda_{2,1} & 1 \\
\lambda_{3,1} & \lambda_{3,2} \\
\vdots & \vdots
\end{bmatrix}
```

### Multiple Lags

```julia
# 5 observables, 2 factors, VAR(3) factor dynamics, 1 lag in loadings
spec = dynamic_factor(5, 2; factor_lags=3, obs_lags=1)

println("States: ", spec.n_states)  # 6 = 2 factors × max(3, 1+1)
```

## Complete Example: Macroeconomic Factor Model

### Using DynamicFactorModel (Recommended for Large Panels)

```julia
using Siphon
using LinearAlgebra
using Statistics
include("src/inplace.jl")

# Load FRED-QD or similar macro data
# y should be N × n (observables × time)

N, n = 100, 200  # 100 series, 200 quarters
k = 6            # 6 factors

# Create and fit model
model = DynamicFactorModel(N, k, n;
    factor_lags = 3,   # VAR(3)
    error_lags = 1     # AR(1) errors
)

fit!(EM(), model, y; maxiter=200, verbose=true)

# Variance decomposition
f = factors(model)
Λ = loadings(model)[1]  # Contemporaneous loadings
factor_cov = (f * f') / n

communalities = zeros(N)
for i in 1:N
    λ_i = Λ[i, :]
    communalities[i] = λ_i' * factor_cov * λ_i
end

idio_var = idiosyncratic_variances(model)
var_explained = communalities ./ (communalities .+ idio_var)

println("Average variance explained: ", mean(var_explained) * 100, "%")
```

### Using DSL (For Smaller Models with MLE)

```julia
using Siphon
using Random
using LinearAlgebra

Random.seed!(42)

# Simulate data from a 2-factor model
n_obs, T = 8, 200

true_loadings = [1.0 0.0; 0.5 1.0; 0.8 0.3; 0.3 0.7;
                 0.6 0.4; 0.4 0.6; 0.7 0.2; 0.2 0.8]
true_φ = [0.9, 0.7]
true_σ_factor = [0.5, 0.4]
true_σ_obs = fill(0.3, n_obs)

# Simulate factors
f = zeros(2, T)
for t in 2:T
    f[:, t] = Diagonal(true_φ) * f[:, t-1] + Diagonal(true_σ_factor) * randn(2)
end

# Generate observations
y = zeros(n_obs, T)
for t in 1:T
    y[:, t] = true_loadings * f[:, t] + Diagonal(true_σ_obs) * randn(n_obs)
end

# Specify and estimate model
spec = dynamic_factor(n_obs, 2;
    loadings_init = 0.5,
    ar_init = 0.7,
    σ_obs_init = 0.5,
    σ_factor_init = 0.5
)

result = optimize_ssm(spec, y)

println("Estimated AR coefficients:")
println("  φ_1_1 = ", round(result.θ.φ_1_1, digits=3), " (true: 0.9)")
println("  φ_2_1 = ", round(result.θ.φ_2_1, digits=3), " (true: 0.7)")

# Extract smoothed factors
ss = build_linear_state_space(spec, result.θ, y)
filt = kalman_filter(ss.p, y, ss.a1, ss.P1)
smooth = kalman_smoother(ss.p.Z, ss.p.T, filt.at, filt.Pt, filt.vt, filt.Ft)

println("\nCorrelation with true factors:")
println("  Factor 1: ", round(cor(smooth.alpha[1, :], f[1, :]), digits=3))
println("  Factor 2: ", round(cor(smooth.alpha[2, :], f[2, :]), digits=3))
```

## Forecasting

```julia
# Forecast h steps ahead
fc = forecast(model, 12)  # 12-step forecast

# Access forecast components
fc.obs_mean      # N × h forecasted observations
fc.obs_cov       # N × N × h forecast covariances
fc.factor_mean   # k × h forecasted factors
fc.factor_cov    # k × k × h factor forecast covariances
```

## Tips and Best Practices

1. **Number of factors**: Start with fewer factors. Use information criteria or scree plots to select k.

2. **Initialization**: The EM algorithm uses PCA for initialization. For MLE, good initial values help.

3. **Large panels**: Use `DynamicFactorModel` with EM for N > 20. The in-place implementation is memory-efficient.

4. **Missing data**: Both approaches handle missing values (NaN) automatically.

5. **AR errors**: Include AR errors (`error_lags > 0`) when idiosyncratic components are persistent.

6. **Dynamic loadings**: Use `loading_lags > 0` when factors have delayed effects on observables.

7. **Convergence**: Check `isconverged(model)` and examine the log-likelihood history.

## Function Reference

### DynamicFactorModel Constructor

```julia
DynamicFactorModel(n_obs, n_factors, n_times;
    loading_lags = 0,   # Lags in λ(L), 0 = static loadings
    factor_lags = 1,    # Lags in factor VAR
    error_lags = 0,     # Lags in AR errors, 0 = white noise
    T = Float64         # Element type
)
```

### dynamic_factor Template

```julia
dynamic_factor(n_obs, n_factors;
    factor_lags = 1,           # AR lags in factor dynamics
    obs_lags = 0,              # Lagged factor loadings
    correlated_errors = false, # Full H matrix if true
    loadings_init = 0.5,       # Initial loading values
    ar_init = 0.5,             # Initial AR coefficient
    σ_obs_init = 1.0,          # Initial obs error std dev
    σ_factor_init = 1.0,       # Initial factor std dev
    diffuse = true             # Diffuse initialization
)
```

## Next Steps

- Learn about **[Custom Models](custom_models.md)** for more flexible specifications
- Check the **[Core Functions](../api/core.md)** API reference
