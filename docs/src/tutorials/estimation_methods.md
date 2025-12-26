# Estimation Methods

This tutorial demonstrates how to specify and estimate the same model using Siphon.jl's different approaches:

1. **Unified fit! API** (recommended): Use `StateSpaceModel` with `fit!(MLE(), ...)` or `fit!(EM(), ...)`
2. **Direct optimization**: Use `SSMSpec` with `optimize_ssm()` for more control
3. **Profile EM**: Use `profile_em_ssm()` for DNS models with nonlinear λ

We'll use the **Dynamic Nelson-Siegel (DNS)** yield curve model as our running example.

## The DNS Model

The Dynamic Nelson-Siegel model decomposes the yield curve into three latent factors:

```math
\begin{aligned}
y_t &= Z(\lambda) f_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, H) \\
f_{t+1} &= T f_t + \eta_t, \quad \eta_t \sim N(0, Q)
\end{aligned}
```

where:
- ``y_t`` is the ``p \times 1`` vector of yields at different maturities
- ``f_t = [L_t, S_t, C_t]'`` are Level, Slope, and Curvature factors
- ``Z(\lambda)`` is the ``p \times 3`` loading matrix with decay parameter ``\lambda``

The factor loadings are:
```math
Z(\lambda)_i = \begin{bmatrix} 1 & \frac{1-e^{-\lambda\tau_i}}{\lambda\tau_i} & \frac{1-e^{-\lambda\tau_i}}{\lambda\tau_i} - e^{-\lambda\tau_i} \end{bmatrix}
```

## Simulating Test Data

First, let's simulate data from a known DNS model:

```julia
using Siphon
using LinearAlgebra
using Random
using Statistics

Random.seed!(42)

# Maturities in months
maturities = [3, 6, 12, 24, 36, 60, 84, 120]
n_maturities = length(maturities)
n_obs = 200

# True parameters
λ_true = 0.0609
T_true = Diagonal([0.99, 0.95, 0.90])
Q_true = Diagonal([0.01, 0.02, 0.03])
H_true = 0.0001 * I(n_maturities)

# Build DNS loadings
function dns_loadings(λ, maturities)
    p = length(maturities)
    Z = ones(p, 3)
    for (i, τ) in enumerate(maturities)
        x = λ * τ
        Z[i, 2] = x < 1e-10 ? 1.0 - x/2 : (1 - exp(-x)) / x
        Z[i, 3] = Z[i, 2] - exp(-x)
    end
    return Z
end

Z_true = dns_loadings(λ_true, maturities)

# Simulate factors and yields
L_Q = cholesky(Symmetric(Matrix(Q_true))).L
L_H = cholesky(Symmetric(Matrix(H_true))).L

factors = zeros(3, n_obs)
yields = zeros(n_maturities, n_obs)

for t in 1:n_obs
    if t > 1
        factors[:, t] = T_true * factors[:, t-1] + L_Q * randn(3)
    end
    yields[:, t] = Z_true * factors[:, t] + L_H * randn(n_maturities)
end

println("Simulated $n_obs observations at $(n_maturities) maturities")
```

## Approach 1: DSL with optimize_ssm (MLE)

The DSL approach uses `dns_model()` to create a specification and `optimize_ssm()` for MLE:

```julia
# Create DNS specification with diagonal dynamics
spec = dns_model(maturities;
    T_structure = :diagonal,  # Diagonal AR(1) for each factor
    H_structure = :diagonal,  # Diagonal observation variances
    Q_structure = :diagonal,  # Diagonal state variances
    λ_init = 0.06,
    T_init = 0.9,
    Q_init = 0.01,
    H_init = 0.001
)

println("Parameters: ", param_names(spec))
# [:λ, :T_L, :T_S, :T_C, :Q_L, :Q_S, :Q_C, :H_1, ..., :H_8]

# Estimate via MLE (gradient-based optimization)
result_mle = optimize_ssm(spec, yields; maxiters=500)

println("Estimated λ: ", round(result_mle.θ.λ, digits=4), " (true: $λ_true)")
println("Estimated T diagonal: ", [result_mle.θ.T_L, result_mle.θ.T_S, result_mle.θ.T_C])
println("Log-likelihood: ", round(result_mle.loglik, digits=2))
```

### Extracting Smoothed Factors

```julia
# Build state-space with estimated parameters
ss = build_linear_state_space(spec, result_mle.θ, yields)

# Run filter and smoother
filt = kalman_filter(ss.p, yields, ss.a1, ss.P1)
smooth = kalman_smoother(ss.p.Z, ss.p.T, filt.at, filt.Pt, filt.vt, filt.Ft)

# Compare with true factors
for (i, name) in enumerate(["Level", "Slope", "Curvature"])
    corr = cor(factors[i, :], smooth.alpha[i, :])
    println("$name correlation: ", round(corr, digits=4))
end
```

## Approach 2: DSL with profile_em_ssm (Profile EM)

For DNS models, the decay parameter ``\lambda`` appears nonlinearly in the loadings, making pure EM difficult. The **profile EM** approach:
1. Grids over ``\lambda`` values
2. For each ``\lambda``, runs EM to estimate T, Q, H
3. Returns the ``\lambda`` with highest log-likelihood

```julia
# Create spec with full covariance structures (for EM)
spec_full = dns_model(maturities;
    T_structure = :full,      # Full 3×3 VAR matrix
    H_structure = :diagonal,
    Q_structure = :full,      # Full 3×3 covariance
    λ_init = 0.06
)

# Profile EM estimation
result_em = profile_em_ssm(spec_full, yields;
    λ_grid = 0.02:0.01:0.12,  # Grid of λ values to search
    maxiter = 200,
    verbose = true
)

println("\nProfile EM Results:")
println("Optimal λ: ", round(result_em.λ_optimal, digits=4), " (true: $λ_true)")
println("Log-likelihood: ", round(result_em.loglik, digits=2))

# Access estimated matrices
T_est = result_em.em_result.T
Q_est = result_em.em_result.Q
H_est = result_em.em_result.H

println("\nEstimated T (factor dynamics):")
display(round.(T_est, digits=3))

println("\nEstimated Q diagonal: ", round.(diag(Q_est), digits=4))
println("True Q diagonal: ", diag(Q_true))
```

## Approach 3: StateSpaceModel with fit! (Recommended)

The `StateSpaceModel` type wraps an `SSMSpec` and provides a unified `fit!` interface:

```julia
# Create StateSpaceModel from spec
spec = dns_model(maturities;
    T_structure = :diagonal,
    H_structure = :diagonal,
    Q_structure = :diagonal,
    λ_init = 0.06
)

model = StateSpaceModel(spec, n_obs)

# Fit with MLE
fit!(MLE(), model, yields)

println("MLE Results:")
println("Converged: ", model.converged)
println("Log-likelihood: ", round(loglikelihood(model), digits=2))
println("Parameters: ", model.theta_values)

# Access filtered states directly
println("Filtered state at t=100: ", model.att[:, 100])
```

### StateSpaceModel with EM

For models where EM is applicable, you can use `fit!(EM(), ...)`:

```julia
# Local level model example (EM works well here)
spec_ll = local_level()
model_ll = StateSpaceModel(spec_ll, n_obs)

# Simulate local level data
y_ll = cumsum(randn(n_obs)) + 0.5 * randn(n_obs)
y_ll = reshape(y_ll, 1, n_obs)

# Fit with EM
fit!(EM(), model_ll, y_ll; maxiter=200, tol=1e-6, verbose=true)

println("\nLocal Level EM Results:")
println("Converged: ", model_ll.converged)
println("Iterations: ", model_ll.iterations)
println("Parameters: ", model_ll.theta_values)
```

## Comparison: When to Use Each Approach

| Approach | Best For | Advantages | Limitations |
|----------|----------|------------|-------------|
| `fit!(MLE(), ...)` | General-purpose | Clean API, auto memory management | May converge slowly for many parameters |
| `fit!(EM(), ...)` | Variance estimation | Fast closed-form updates, zero-allocation | Limited to specific model structures |
| `optimize_ssm()` | Fine-grained control | Direct access to optimizer settings | Lower-level API |
| `profile_em_ssm()` | DNS/Svensson models | Robust for λ estimation, full covariances | Requires grid search over λ |

## Complete Example: DNS with Full Pipeline

Here's a complete workflow combining specification, estimation, and analysis:

```julia
using Siphon
using LinearAlgebra
using Statistics

# 1. SPECIFY MODEL
maturities = [3, 6, 12, 24, 60, 120]
spec = dns_model(maturities;
    T_structure = :full,
    Q_structure = :full,
    H_structure = :diagonal,
    λ_init = 0.06
)

println("Model: ", spec.name)
println("Parameters: ", length(spec.params))
println("States: ", spec.n_states)

# 2. LOAD/SIMULATE DATA
# (Using simulated data from above)

# 3. ESTIMATE
result = profile_em_ssm(spec, yields;
    λ_grid = 0.03:0.005:0.10,
    maxiter = 300,
    verbose = false
)

# 4. EXTRACT RESULTS
λ_opt = result.λ_optimal
T_opt = result.em_result.T
Q_opt = result.em_result.Q
H_opt = result.em_result.H

println("\n=== Estimation Results ===")
println("λ: ", round(λ_opt, digits=4))
println("T eigenvalues: ", round.(eigvals(T_opt), digits=3))
println("Q diagonal: ", round.(diag(Q_opt), digits=5))

# 5. SMOOTH FACTORS
Z_opt = dns_loadings(λ_opt, maturities)
p_final = KFParms(Z_opt, H_opt, T_opt, Matrix{Float64}(I, 3, 3), Q_opt)
a1 = zeros(3)
P1 = 1e4 * Matrix{Float64}(I, 3, 3)

filt = kalman_filter(p_final, yields, a1, P1)
smooth = kalman_smoother(Z_opt, T_opt, filt.at, filt.Pt, filt.vt, filt.Ft)

# 6. ANALYZE FACTORS
println("\n=== Factor Analysis ===")
for (i, name) in enumerate(["Level", "Slope", "Curvature"])
    f = smooth.alpha[i, :]
    println("$name: mean=$(round(mean(f), digits=2)), std=$(round(std(f), digits=2))")
end

# 7. FORECAST (optional)
# Build forecast from final state
h = 12  # 12-period ahead
f_last = smooth.alpha[:, end]
f_forecast = zeros(3, h)
for t in 1:h
    f_forecast[:, t] = T_opt^t * f_last
end

y_forecast = Z_opt * f_forecast
println("\n=== 12-Period Yield Forecast ===")
println("Short rate (3m): ", round.(y_forecast[1, :], digits=2))
```

## Summary

Siphon.jl provides multiple estimation approaches to suit different needs:

1. **`fit!(MLE(), model, y)`**: Recommended general-purpose MLE with unified API
2. **`fit!(EM(), model, y)`**: EM algorithm for models with closed-form M-steps
3. **`profile_em_ssm(spec, y)`**: Profile EM for DNS models with nonlinear λ
4. **`optimize_ssm(spec, y)`**: Direct optimization for fine-grained control

Choose based on your model structure and computational requirements.

## Next Steps

- See **[Custom Models](custom_models.md)** for building your own specifications
- See **[Dynamic Factor Models](dynamic_factor.md)** for large-panel factor analysis
- See **[Parameter Transformations](transformations.md)** for understanding constraints
