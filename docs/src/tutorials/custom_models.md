# Custom Models

This tutorial covers how to specify custom state space models using Siphon.jl's domain-specific language (DSL). You'll learn:

1. The `custom_ssm` function for specifying arbitrary models
2. Using `FreeParam` to mark estimated parameters
3. Matrix helper functions for common patterns
4. Parameter-dependent matrices with `MatrixExpr`
5. Full positive-definite covariance matrices with `cov_free`

## The `custom_ssm` Function

The `custom_ssm` function lets you specify any linear state space model by providing the system matrices directly:

```julia
using Siphon
using LinearAlgebra

# A simple local level model specified manually
spec = custom_ssm(
    Z = [1.0],                                    # Observation matrix
    H = [FreeParam(:var_obs, init=100.0, lower=0.0)],  # Obs variance
    T = [1.0],                                    # Transition matrix
    R = [1.0],                                    # Selection matrix
    Q = [FreeParam(:var_level, init=25.0, lower=0.0)], # State variance
    a1 = [0.0],                                   # Initial state mean
    P1 = [1e7],                                   # Initial state variance (diffuse)
    name = :MyLocalLevel
)

println("Parameters: ", param_names(spec))
# [:var_obs, :var_level]
```

### Key Points

- **Fixed values**: Use regular numbers for fixed matrix elements
- **Free parameters**: Use `FreeParam(...)` for parameters to be estimated
- **Bounds**: Use `lower=0.0` for variance parameters. TransformVariables.jl automatically applies appropriate transformations for constrained parameters.
- **Dimensions**: Inferred automatically from the provided matrices

## The `FreeParam` Type

`FreeParam` marks a matrix element as an estimated parameter:

```julia
FreeParam(name::Symbol;
    init = 0.0,           # Initial value for optimization
    lower = -Inf,         # Lower bound
    upper = Inf           # Upper bound
)
```

### Examples

```julia
# Variance parameter (use lower=0.0 for positivity constraint)
FreeParam(:var_obs, init=100.0, lower=0.0)

# Bounded coefficient (e.g., AR coefficient)
FreeParam(:ρ, init=0.8, lower=-0.99, upper=0.99)

# Unbounded coefficient
FreeParam(:β, init=0.0)
```

### The `@P` Macro

For quick parameter specification:

```julia
@P(:σ, 1.0)  # Equivalent to FreeParam(:σ, init=1.0)
```

## Example: Local Linear Trend Model

The local linear trend model has two states: level and slope.

```math
\begin{aligned}
y_t &= \mu_t + \varepsilon_t \\
\mu_{t+1} &= \mu_t + \nu_t + \eta^\mu_t \\
\nu_{t+1} &= \nu_t + \eta^\nu_t
\end{aligned}
```

```julia
spec = custom_ssm(
    Z = [1.0 0.0],  # Only level is observed
    H = [FreeParam(:var_obs, init=1.0, lower=0.0)],
    T = [1.0 1.0;   # Level depends on previous level + slope
         0.0 1.0],  # Slope is a random walk
    R = Matrix(1.0I, 2, 2),  # Both states receive shocks
    Q = [FreeParam(:var_level, init=0.01, lower=0.0)  0.0;
         0.0  FreeParam(:var_slope, init=0.0001, lower=0.0)],
    a1 = [0.0, 0.0],
    P1 = 1e7 * Matrix(1.0I, 2, 2),
    name = :LocalLinearTrend
)

println("States: ", spec.n_states)  # 2
println("Parameters: ", param_names(spec))  # [:var_obs, :var_level, :var_slope]
```

## Matrix Helper Functions

Siphon.jl provides helper functions for common matrix patterns:

### Diagonal Matrices with Free Parameters

```julia
# Diagonal matrix with n free variance parameters (lower=0.0 by default)
H = diag_free(3, :var_obs)
# Creates parameters: var_obs_1, var_obs_2, var_obs_3

# With custom initial value
H = diag_free(3, :var_obs; init=2.0)
```

### Scalar Matrices

```julia
# Scalar times identity
H = scalar_free(3, :var_obs; init=1.0)
# One parameter var_obs, applied to all diagonal elements
```

### Fixed Diagonal Matrices

```julia
# Fixed diagonal values
H = diag_fixed(3, [1.0, 2.0, 3.0])
```

### Identity and Zero Matrices

```julia
Z = identity_mat(3)     # 3×3 identity
R = zeros_mat(3, 2)     # 3×2 zeros
O = ones_mat(2, 2)      # 2×2 ones
```

### Selection Matrix

```julia
# Select specific states for observation
# If n_states=4 and we observe states 1 and 3:
Z = selection_mat([1, 3], 4)  # 2×4 matrix
```

### Companion Matrix

For VAR/ARMA models in companion form:

```julia
# AR(2) companion matrix with free coefficients
T = companion_mat(2, :φ; init=[0.5, 0.3])
# Creates parameters φ_1, φ_2
# Matrix: [φ_1  φ_2]
#         [1    0  ]
```

### Lower Triangular Free

```julia
# Lower triangular matrix with free parameters
L = lower_triangular_free(3, :L)
# Creates parameters for lower triangle: L_1_1, L_2_1, L_2_2, L_3_1, L_3_2, L_3_3
```

### Symmetric Free

```julia
# Symmetric matrix with free parameters
S = symmetric_free(2, :S)
# Creates parameters: S_1_1, S_2_1, S_2_2
# The matrix is symmetric by construction
```

## Example: Bivariate VAR(1)

A bivariate VAR(1) model:

```math
y_t = \Phi y_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim N(0, \Sigma)
```

```julia
spec = custom_ssm(
    Z = identity_mat(2),          # Observe both states
    H = zeros_mat(2, 2),          # No measurement error (VAR is exact)
    T = [FreeParam(:φ_11, init=0.5)  FreeParam(:φ_12, init=0.1);
         FreeParam(:φ_21, init=0.1)  FreeParam(:φ_22, init=0.5)],
    R = identity_mat(2),
    Q = diag_free(2, :var_innov),  # Diagonal innovation covariance
    a1 = [0.0, 0.0],
    P1 = 1e4 * Matrix(1.0I, 2, 2),
    name = :BivariateVAR1
)
```

## Full Covariance Matrices with `cov_free`

For models with correlated errors, use `cov_free` to specify a full positive-definite covariance matrix:

```julia
# 3×3 positive definite covariance matrix
Q = cov_free(3, :Q)
```

This uses the decomposition ``\Sigma = D \cdot \text{Corr} \cdot D`` where:
- ``D = \text{diag}(\sigma_1, \ldots, \sigma_n)`` contains standard deviations
- ``\text{Corr}`` is a correlation matrix (constructed via Cholesky factor)

### Parameters Created

For `cov_free(n, :prefix)`:
- `n` standard deviation parameters: `prefix_σ_1`, ..., `prefix_σ_n`
- `n(n-1)/2` correlation parameters: `prefix_corr_1`, ...

### Example: VAR with Correlated Innovations

```julia
spec = custom_ssm(
    Z = identity_mat(2),
    H = zeros_mat(2, 2),
    T = [FreeParam(:φ_11, init=0.5)  FreeParam(:φ_12, init=0.0);
         FreeParam(:φ_21, init=0.0)  FreeParam(:φ_22, init=0.5)],
    R = identity_mat(2),
    Q = cov_free(2, :Q),  # Full 2×2 covariance
    a1 = [0.0, 0.0],
    P1 = 1e4 * Matrix(1.0I, 2, 2),
    name = :VAR1_Correlated
)

println("Parameters: ", param_names(spec))
# [:φ_11, :φ_12, :φ_21, :φ_22, :Q_σ_1, :Q_σ_2, :Q_corr_1]
```

## Parameter-Dependent Matrices with `MatrixExpr`

For advanced models where matrix elements depend on parameters in complex ways (e.g., yield curve models), use `MatrixExpr`:

```julia
struct MatrixExpr
    params::Vector{SSMParameter}  # Parameters used
    data::NamedTuple              # Static data (e.g., maturities)
    builder::Function             # Function to build the matrix
    dims::Tuple{Int,Int}          # Matrix dimensions
end
```

### Example: Nelson-Siegel Yield Curve Model

The Dynamic Nelson-Siegel model has loadings that depend on a decay parameter λ:

```math
Z_{i,j} = \begin{cases}
1 & j=1 \\
\frac{1-e^{-\lambda \tau_i}}{\lambda \tau_i} & j=2 \\
\frac{1-e^{-\lambda \tau_i}}{\lambda \tau_i} - e^{-\lambda \tau_i} & j=3
\end{cases}
```

```julia
using Siphon

# Builder function for DNS loadings
function dns_builder(θ::Dict, data)
    λ = θ[:λ]
    τ = data.maturities
    n = length(τ)
    T = eltype(λ)

    Z = zeros(T, n, 3)
    for i in 1:n
        Z[i, 1] = one(T)
        Z[i, 2] = dns_loading1(λ, τ[i])
        Z[i, 3] = dns_loading2(λ, τ[i])
    end
    return Z
end

# Create the MatrixExpr
maturities = [3, 6, 12, 24, 60, 120]  # in months
Z_expr = MatrixExpr(
    [SSMParameter(:λ, 0.001, 0.5, 0.0609)],  # name, lower, upper, init
    (maturities = maturities,),
    dns_builder,
    (length(maturities), 3)
)

# Use in custom_ssm
spec = custom_ssm(
    Z = Z_expr,
    H = diag_free(6, :var_obs),
    T = diag_free(3, :φ; lower=-0.999, upper=0.999, init=0.9),
    R = identity_mat(3),
    Q = diag_free(3, :var_factor),
    a1 = [0.0, 0.0, 0.0],
    P1 = 1e4 * Matrix(1.0I, 3, 3),
    name = :DynamicNelsonSiegel
)
```

### Built-in DNS Helpers

Siphon.jl provides convenience functions for DNS models:

```julia
# Build DNS Z matrix directly
Z = build_dns_loadings(maturities, :λ; λ_init=0.0609)

# Or Svensson (4-factor) loadings
Z = build_svensson_loadings(maturities, :λ1, :λ2; λ1_init=0.05, λ2_init=0.10)
```

## Putting It All Together: A Complete Example

Here's a complete workflow for a custom bivariate model:

```julia
using Siphon
using LinearAlgebra
using Random

Random.seed!(123)

# Specify the model
spec = custom_ssm(
    Z = [1.0 0.0;    # First obs loads on first state
         0.0 1.0],   # Second obs loads on second state
    H = diag_free(2, :var_obs; init=0.5),
    T = [FreeParam(:φ_1, init=0.8, lower=-0.99, upper=0.99)  0.0;
         0.0  FreeParam(:φ_2, init=0.6, lower=-0.99, upper=0.99)],
    R = identity_mat(2),
    Q = cov_free(2, :Q; init_σ=1.0),
    a1 = [0.0, 0.0],
    P1 = 10.0 * Matrix(1.0I, 2, 2),
    name = :BivariateAR1
)

# Check the specification
println("Model: ", spec.name)
println("States: ", spec.n_states)
println("Parameters: ", param_names(spec))
println("Number of parameters: ", n_params(spec))

# Simulate some data
T = 200
y = randn(2, T)

# Estimate parameters
result = optimize_ssm(spec, y)
println("\nEstimated parameters:")
for (name, val) in pairs(result.θ)
    println("  $name = $val")
end

# Get smoothed states
ss = build_linear_state_space(spec, result.θ, y)
filt = kalman_filter(ss.p, y, ss.a1, ss.P1)
smooth = kalman_smoother(ss.p.Z, ss.p.T, filt.at, filt.Pt, filt.vt, filt.Ft)
println("\nSmoothed state at t=100: ", smooth.alpha[:, 100])
```

## Tips and Best Practices

1. **Parameter naming**: Use descriptive names with prefixes (e.g., `var_obs`, `var_level`) for clarity.

2. **Initial values**: Good initial values help optimization converge. Use domain knowledge when possible.

3. **Variance parameters**: Use `lower=0.0` for variance parameters. TransformVariables.jl automatically applies `asℝ₊` transformation to enforce positivity.

4. **Bounds**: Use bounds for constrained parameters (e.g., AR coefficients in (-1, 1) for stationarity).

5. **Diffuse initialization**: Use large values in P1 (e.g., 1e7) for diffuse priors on initial states.

6. **Dimension checks**: `custom_ssm` validates dimensions automatically--let it catch errors early.

## Next Steps

- Learn about [Parameter Transformations](transformations.md) for understanding how bounds are handled
- Learn about [Dynamic Factor Models](dynamic_factor.md) for multivariate factor analysis
- See the [Core Functions](../api/core.md) and [DSL & Templates](../api/dsl.md) for complete API documentation
