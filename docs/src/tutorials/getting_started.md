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

## Computing the Log-Likelihood

Given a model specification and parameters, we can compute the log-likelihood:

```julia
# Get initial parameter values
θ = initial_values(spec)

# Build the state space components
ss = build_linear_state_space(spec, θ, y)

# Compute log-likelihood
ll = kalman_loglik(ss.p, y, ss.a1, ss.P1)
println("Log-likelihood at initial values: ", ll)
```

The `build_linear_state_space` function returns a named tuple with:
- `p`: A `KFParms` struct containing the system matrices (Z, H, T, R, Q)
- `a1`: Initial state mean vector
- `P1`: Initial state covariance matrix

## Kalman Filter

The Kalman filter computes the sequence of filtered state estimates and their covariances:

```julia
# Run the full Kalman filter
filt = kalman_filter(ss.p, y, ss.a1, ss.P1)

# Access results
at = filt.at       # Predicted state means E[α_t|y_{1:t-1}] (m × n)
Pt = filt.Pt       # Predicted state covariances (m × m × n)
att = filt.att     # Filtered state means E[α_t|y_{1:t}] (m × n)
Ptt = filt.Ptt     # Filtered state covariances (m × m × n)
vt = filt.vt       # Prediction errors (p × n)
Ft = filt.Ft       # Prediction error variances (p × p × n)
Kt = filt.Kt       # Kalman gains (m × p × n)
ll = filt.loglik   # Log-likelihood

println("Final filtered state: ", att[:, end])
println("Log-likelihood: ", ll)
```

### Filter Output Structure

The `KalmanFilterResult` contains:
- `at[:, t]`: State estimate at time `t` given observations up to `t-1` (i.e., ``a_{t|t-1}``)
- `att[:, t]`: State estimate at time `t` given observations up to `t` (i.e., ``a_{t|t}``)
- `Pt[:, :, t]`, `Ptt[:, :, t]`: Corresponding covariance matrices
- `vt[:, t]`: Innovation (prediction error) at time `t`
- `Ft[:, :, t]`: Innovation covariance at time `t`
- `Kt[:, :, t]`: Kalman gain at time `t`

## Kalman Smoother

The Kalman smoother computes smoothed state estimates using all available observations:

```julia
# Run the Kalman smoother (uses filter output)
smooth = kalman_smoother(ss.p.Z, ss.p.T, filt.at, filt.Pt, filt.vt, filt.Ft)

# Access results
alpha = smooth.alpha  # Smoothed state means E[α_t|y_{1:n}] (m × n)
V = smooth.V          # Smoothed state covariances (m × m × n)

println("Smoothed state at t=50: ", alpha[:, 50])
```

### With Cross-Lag Covariances

For EM algorithm applications, you may need cross-lag covariances:

```julia
smooth = kalman_smoother(ss.p.Z, ss.p.T, filt.at, filt.Pt, filt.vt, filt.Ft;
                          compute_crosscov=true)

# Additional output
P_crosslag = smooth.P_crosslag  # Cov[α_{t+1}, α_t|y_{1:n}] (m × m × n-1)
```

## Parameter Estimation

Siphon.jl provides a unified `fit!` API for parameter estimation:

```julia
# Create a model container
model = StateSpaceModel(spec, size(y, 2))

# Maximum likelihood estimation
fit!(MLE(), model, y)

# Access results
println("Log-likelihood: ", loglikelihood(model))
println("Parameters: ", model.theta_values)
println("Converged: ", model.converged)
```

### EM Algorithm

For models with only variance parameters, the EM algorithm can be faster:

```julia
# Create model and estimate via EM
model = StateSpaceModel(spec, size(y, 2))
fit!(EM(), model, y; maxiter=200, verbose=true)

println("Converged: ", model.converged)
println("Iterations: ", model.iterations)
println("Parameters: ", model.theta_values)
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

After fitting, you can run the filter/smoother with the fitted model:

```julia
# Using the fit! API, filter results are stored in the model
model = StateSpaceModel(spec, size(y, 2))
fit!(MLE(), model, y)

# Access filtered states directly from model
println("Filtered state std: ", std(model.att[1, :]))

# Or use optimize_ssm and build state space manually
result = optimize_ssm(spec, y)
ss_mle = build_linear_state_space(spec, result.θ, y)
filt = kalman_filter(ss_mle.p, y, ss_mle.a1, ss_mle.P1)
smooth = kalman_smoother(ss_mle.p.Z, ss_mle.p.T, filt.at, filt.Pt, filt.vt, filt.Ft)

println("Smoothed state std: ", std(smooth.alpha[1, :]))
```

The smoothed estimates are generally more accurate than filtered estimates because they use information from the entire sample.

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

# Filter and smooth work normally
filt = kalman_filter(ss.p, y_missing, ss.a1, ss.P1)
smooth = kalman_smoother(ss.p.Z, ss.p.T, filt.at, filt.Pt, filt.vt, filt.Ft;
                          missing_mask=filt.missing_mask)

# The smoother will interpolate through missing periods
```

## Next Steps

- Learn how to specify **[Custom Models](custom_models.md)** using the DSL
- Understand **[Parameter Transformations](transformations.md)** for constrained optimization
- Explore **[Dynamic Factor Models](dynamic_factor.md)** for multivariate analysis
- See the **[Core Functions](../api/core.md)** and **[DSL & Templates](../api/dsl.md)** for complete API documentation
