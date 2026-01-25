# Getting Started

This tutorial introduces the basic workflow for working with state space models in Siphon.jl. We'll cover:

1. Creating a model specification using templates
2. Computing the log-likelihood
3. Running the Kalman filter
4. Running the Kalman smoother
5. Estimating parameters via maximum likelihood

## The Local Level Model

We'll use the classic **local level model** (also known as the random walk plus noise model) as our running example. This model decomposes a time series into a slowly-evolving level plus observation noise:

```math
\begin{aligned}
y_t &= \mu_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, \sigma^2_\varepsilon) \\
\mu_{t+1} &= \mu_t + \eta_t, \quad \eta_t \sim N(0, \sigma^2_\eta)
\end{aligned}
```

In state space form:
- ``Z = [1]`` (the observation loads directly on the state)
- ``H = [\sigma^2_\varepsilon]`` (observation variance)
- ``T = [1]`` (random walk dynamics)
- ``R = [1]`` (state receives the shock directly)
- ``Q = [\sigma^2_\eta]`` (state innovation variance)

## Creating a Model Specification

```julia
using Siphon

# Create a local level model with initial parameter values
spec = local_level()

# Inspect the specification
println("Parameters: ", param_names(spec))
println("Initial values: ", initial_values(spec))
```

Output:
```
Parameters: (:var_obs, :var_level)
Initial values: [225.0, 100.0]
```

## Loading Data

Let's work with the classic Nile River dataset:

```julia
using DelimitedFiles

# Load the Nile data
nile = readdlm("test/Nile.csv", ',', Float64)
y = nile'  # Convert to 1×n matrix (p × T format)

println("Number of observations: ", size(y, 2))
```

## The Unified StateSpaceModel API

Siphon.jl provides a unified API centered around `StateSpaceModel`. There are two main ways to create a model:

### Option 1: Create Model with Known Parameters

If you already have parameter values (from prior knowledge, simulation, or external estimation):

```julia
# Define known parameters
θ = (var_obs=15099.0, var_level=1469.0)

# Create model with these parameters
model = StateSpaceModel(spec, θ, size(y, 2))

# Run filter and smoother
ll = kalman_filter!(model, y)
kalman_smoother!(model)

# Access results
println("Log-likelihood: ", ll)
println("Filtered states: ", filtered_states(model))
println("Smoothed states: ", smoothed_states(model))
```

### Option 2: Estimate Parameters

If you want to estimate parameters from data:

```julia
# Create an unfitted model
model = StateSpaceModel(spec, size(y, 2))

# Estimate via MLE
fit!(MLE(), model, y)

# Or estimate via EM
# fit!(EM(), model, y; maxiter=200)

# Access results
println("Log-likelihood: ", loglikelihood(model))
println("Parameters: ", parameters(model))
println("Converged: ", isconverged(model))
```

## Computing the Log-Likelihood

With the unified API, computing the log-likelihood is straightforward:

```julia
# With known parameters
θ = (var_obs=15099.0, var_level=1469.0)
model = StateSpaceModel(spec, θ, size(y, 2))

# Method 1: kalman_loglik (doesn't store filter results)
ll = kalman_loglik(model, y)
println("Log-likelihood: ", ll)

# Method 2: kalman_filter! (stores filter results for later use)
ll = kalman_filter!(model, y)
println("Log-likelihood: ", ll)
```

## Kalman Filter

The Kalman filter computes the sequence of filtered state estimates:

```julia
θ = (var_obs=15099.0, var_level=1469.0)
model = StateSpaceModel(spec, θ, size(y, 2))

# Run the filter
kalman_filter!(model, y)

# Access filter results
att = filtered_states(model)           # E[α_t|y_{1:t}] (m × n)
Ptt = filtered_states_cov(model)       # Var[α_t|y_{1:t}] (m × m × n)
at = predicted_states(model)           # E[α_t|y_{1:t-1}] (m × n)
Pt = predicted_states_cov(model)       # Var[α_t|y_{1:t-1}] (m × m × n)
vt = prediction_errors(model)          # y_t - E[y_t|y_{1:t-1}] (p × n)
Ft = prediction_errors_cov(model)      # Var[y_t|y_{1:t-1}] (p × p × n)

println("Final filtered state: ", att[:, end])
println("Log-likelihood: ", loglikelihood(model))
```

### Filter Output Structure

- `filtered_states(model)`: State estimate at time `t` given observations up to `t` (i.e., ``a_{t|t}``)
- `predicted_states(model)`: State estimate at time `t` given observations up to `t-1` (i.e., ``a_{t|t-1}``)
- `prediction_errors(model)`: Innovation (prediction error) at time `t`
- All covariance accessors provide the corresponding variance/covariance matrices

## Kalman Smoother

The Kalman smoother computes smoothed state estimates using all available observations:

```julia
θ = (var_obs=15099.0, var_level=1469.0)
model = StateSpaceModel(spec, θ, size(y, 2))

# Run filter first
kalman_filter!(model, y)

# Then run smoother
kalman_smoother!(model)

# Access results
alpha = smoothed_states(model)      # E[α_t|y_{1:n}] (m × n)
V = smoothed_states_cov(model)      # Var[α_t|y_{1:n}] (m × m × n)

println("Smoothed state at t=50: ", alpha[:, 50])
```

Note: The smoother results are cached, so `smoothed_states(model)` will compute the smoother on first call and return cached results thereafter.

## Parameter Estimation

Siphon.jl provides a unified `fit!` API for parameter estimation:

```julia
# Create a model container
model = StateSpaceModel(spec, size(y, 2))

# Maximum likelihood estimation
fit!(MLE(), model, y)

# Access results
println("Log-likelihood: ", loglikelihood(model))
println("Parameters: ", parameters(model))
println("Converged: ", isconverged(model))
```

### EM Algorithm

For models with only variance parameters, the EM algorithm can be faster:

```julia
# Create model and estimate via EM
model = StateSpaceModel(spec, size(y, 2))
fit!(EM(), model, y; maxiter=200, verbose=true)

println("Converged: ", isconverged(model))
println("Iterations: ", niterations(model))
println("Parameters: ", parameters(model))
```

### Alternative: Direct Optimization

For more control over the optimization process:

```julia
# Use optimize_ssm directly
result = optimize_ssm(spec, y)

θ_mle = result.θ           # Estimated parameters (NamedTuple)
loglik = result.loglik     # Maximized log-likelihood

println("Estimated var_obs: ", θ_mle.var_obs)
println("Estimated var_level: ", θ_mle.var_level)
```

## Filter and Smooth with Estimated Parameters

After fitting, you can access filter and smoother results directly:

```julia
# Fit the model
model = StateSpaceModel(spec, size(y, 2))
fit!(MLE(), model, y)

# Filter results are automatically computed during fitting
println("Filtered state std: ", std(filtered_states(model)[1, :]))

# Smoothed states computed on demand
println("Smoothed state std: ", std(smoothed_states(model)[1, :]))
```

The smoothed estimates are generally more accurate than filtered estimates because they use information from the entire sample.

## Accessing System Matrices

After fitting (or with known parameters), you can access the system matrices:

```julia
model = StateSpaceModel(spec, size(y, 2))
fit!(MLE(), model, y)

# Get all matrices at once
mats = system_matrices(model)
println("Z = ", mats.Z)
println("H = ", mats.H)
println("T = ", mats.T)
println("Q = ", mats.Q)

# Or individual matrices
Z = obs_matrix(model)
H = obs_cov(model)
T = transition_matrix(model)
Q = state_cov(model)
```

## Other Pre-Built Templates

Siphon.jl provides several other templates:

### Local Linear Trend

```julia
# Level + slope model
spec = local_linear_trend()
# Parameters: (:var_obs, :var_level, :var_slope)
```

### AR(1) Process

```julia
# AR(1) with measurement noise
spec = ar1(ρ_init=0.9)
# Parameters: (:ρ, :var_obs, :var_state)
```

### ARMA(p, q)

```julia
# ARMA(2, 1) model
spec = arma(2, 1)
# Parameters: (:ar_1, :ar_2, :ma_1, :var)
```

### Dynamic Nelson-Siegel

```julia
# DNS yield curve model
maturities = [3, 6, 12, 24, 36, 60, 84, 120]
spec = dns_model(maturities)
```

## Exact Diffuse Initialization

For models with unknown initial states (like the local level model), standard practice uses a large initial covariance (e.g., `P1 = 1e7 * I`) as an "approximate diffuse" prior. Siphon.jl also supports **exact diffuse initialization** following Durbin & Koopman (2012), which provides a theoretically correct treatment.

### The Problem with Approximate Diffuse

With approximate diffuse, the first observation contributes to the likelihood even though the initial state is essentially unknown. This can bias parameter estimates, especially for short series.

### Exact Diffuse Solution

Exact diffuse splits the initial covariance into:
- `P1_star`: Finite (known) part of initial covariance
- `P1_inf`: Diffuse (infinite) part indicating unknown components

```julia
# Local level model with exact diffuse initialization
p = KFParms(
    [1.0;;],      # Z
    [15099.0;;],  # H (observation variance)
    [1.0;;],      # T
    [1.0;;],      # R
    [1469.1;;]    # Q (state variance)
)

a1 = [0.0]           # Initial state mean
P1_star = [0.0;;]    # No finite uncertainty initially
P1_inf = [1.0;;]     # Full diffuse on the level

# Use 5-argument form to trigger exact diffuse
ll_exact = kalman_loglik(p, y, a1, P1_star, P1_inf)

# Compare with approximate diffuse (4-argument form)
P1_approx = [1e7;;]
ll_approx = kalman_loglik(p, y, a1, P1_approx)

println("Exact diffuse log-likelihood: ", ll_exact)      # -632.55
println("Approximate diffuse log-likelihood: ", ll_approx) # -641.59
```

For the Nile data, exact diffuse gives a log-likelihood about 9 units higher than approximate diffuse. This difference reflects the correct treatment of the unknown initial state.

### Validation Against R's KFAS

Siphon.jl's exact diffuse implementation matches R's KFAS package. For the Nile local level model:

| Quantity | Siphon | KFAS |
|----------|--------|------|
| Log-likelihood (exact diffuse) | -632.5456 | -632.5456 |
| Log-likelihood (approx, P1=1e7) | -641.5856 | -641.5856 |
| MLE H (obs variance) | 15098.52 | 15098.53 |
| MLE Q (state variance) | 1469.18 | 1469.18 |

### MLE with Exact Diffuse

```julia
using Optimization, OptimizationOptimJL

# Define negative log-likelihood with exact diffuse
function negloglik_diffuse(θ, y)
    H, Q = exp(θ[1]), exp(θ[2])  # Ensure positive via log transform
    p = KFParms([1.0;;], [H;;], [1.0;;], [1.0;;], [Q;;])
    a1 = [0.0]
    P1_star = [0.0;;]
    P1_inf = [1.0;;]
    return -kalman_loglik(p, y, a1, P1_star, P1_inf)
end

# Optimize with autodiff
θ0 = [log(1000.0), log(1000.0)]  # Initial values (log scale)
optf = OptimizationFunction(negloglik_diffuse, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, θ0, y)
sol = solve(prob, LBFGS())

H_mle, Q_mle = exp(sol.u[1]), exp(sol.u[2])
println("H = ", H_mle, ", Q = ", Q_mle)  # H ≈ 15099, Q ≈ 1469
```

### Full Filter with Exact Diffuse

```julia
# Get full filter output with exact diffuse
result = kalman_filter(p, y, a1, P1_star, P1_inf)

# Check the diffuse period (number of observations needed to initialize)
d = diffuse_period(result)
println("Diffuse period: ", d, " observations")

# Access diffuse-specific quantities
Pinf_store, Pstar_store = diffuse_covariances(result)
flags = diffuse_flags(result)

# After diffuse period, filtered states are fully initialized
println("First post-diffuse filtered state: ", result.att[:, d+1])
```

### In-Place Version

For large-scale applications, use `DiffuseKalmanWorkspace`:

```julia
# Create diffuse workspace
ws = DiffuseKalmanWorkspace(Z, H, T, R, Q, a1, P1_star, P1_inf, n)

# Run filter (workspace type determines diffuse algorithm)
ll = kalman_filter!(ws, y)

# Access results
d = diffuse_period(ws)
att = filtered_states(ws)
```

### When to Use Exact Diffuse

Use exact diffuse initialization when:
- The initial state is truly unknown (random walk, integrated processes)
- You want theoretically correct likelihood values
- You're comparing with software that uses exact diffuse (e.g., R's KFAS)

Approximate diffuse is sufficient when:
- The series is long (first observation has negligible impact)
- Speed is critical (exact diffuse has some overhead)
- Initial state has a proper prior

## Working with Missing Data

Siphon.jl handles missing observations automatically using NaN:

```julia
# Create data with missing values
y_missing = copy(y)
y_missing[1, 20:25] .= NaN  # Mark as missing

# Use the unified API
θ = (var_obs=15099.0, var_level=1469.0)
model = StateSpaceModel(spec, θ, size(y_missing, 2))

kalman_filter!(model, y_missing)
kalman_smoother!(model)

# The smoother will interpolate through missing periods
println("Smoothed states through missing period: ", smoothed_states(model)[1, 18:27])
```

## Next Steps

- Learn how to specify [Custom Models](custom_models.md) using the DSL
- Understand [Parameter Transformations](transformations.md) for constrained optimization
- Explore [Dynamic Factor Models](dynamic_factor.md) for multivariate analysis
- See the [Core Functions](../api/core.md) and [DSL & Templates](../api/dsl.md) for complete API documentation
