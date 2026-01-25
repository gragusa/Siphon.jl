# ARMA Models

This tutorial demonstrates how to specify and estimate ARMA(p,q) models in state-space form using Siphon.jl. We cover:

1. Theoretical background on ARMA in state-space form
2. Building an ARMA(2,2) model from scratch using `custom_ssm`
3. Using the built-in `arma` template
4. Estimation with MLE and EM
5. Accessing estimated parameters and matrices
6. Validation against known parameters

## Background: ARMA in State-Space Form

An ARMA(p,q) process is defined as:

```math
y_t = \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}
```

where $\varepsilon_t \sim N(0, \sigma^2)$ is white noise.

This can be written in **innovations state-space form** with state dimension $r = \max(p, q+1)$:

**State vector:**
```math
\alpha_t = \begin{pmatrix} y_t \\ y_{t+1|t} \\ \vdots \\ y_{t+r-1|t} \end{pmatrix}
```
where $y_{t+j|t} = E[y_{t+j} | y_1, \ldots, y_t]$.

**Observation equation:**
```math
y_t = Z \alpha_t, \quad Z = \begin{pmatrix} 1 & 0 & \cdots & 0 \end{pmatrix}
```

**Transition equation:**
```math
\alpha_{t+1} = T \alpha_t + R \eta_t, \quad \eta_t \sim N(0, Q)
```

where:
```math
T = \begin{pmatrix}
\phi_1 & 1 & 0 & \cdots & 0 \\
\phi_2 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\phi_{r-1} & 0 & 0 & \cdots & 1 \\
\phi_r & 0 & 0 & \cdots & 0
\end{pmatrix}, \quad
R = \begin{pmatrix} 1 \\ \theta_1 \\ \theta_2 \\ \vdots \\ \theta_{r-1} \end{pmatrix}, \quad
Q = \sigma^2
```

Note: $\phi_i = 0$ for $i > p$ and $\theta_j = 0$ for $j > q$.

## Building ARMA(2,2) from Scratch

Let's build an ARMA(2,2) model step by step using `custom_ssm`:

```julia
using Siphon
using LinearAlgebra
using Random

Random.seed!(42)

# ARMA(2,2): r = max(2, 2+1) = 3
p, q = 2, 2
r = max(p, q + 1)  # = 3

# Observation matrix: Z = [1 0 0]
Z = zeros(1, r)
Z[1, 1] = 1.0

# Observation noise: H = 0 (pure ARMA, no measurement error)
H = zeros(1, 1)

# Transition matrix: companion form
# T = [φ₁  1  0]
#     [φ₂  0  1]
#     [0   0  0]  (φ₃ = 0 since p=2)
T = zeros(r, r)
T[1, 1] = FreeParam(:φ1, init=0.5, lower=-0.99, upper=0.99)
T[2, 1] = FreeParam(:φ2, init=0.2, lower=-0.99, upper=0.99)
# Subdiagonal ones
T[1, 2] = 1.0
T[2, 3] = 1.0

# Selection matrix: R = [1; θ₁; θ₂]
R = zeros(r, 1)
R[1, 1] = 1.0
R[2, 1] = FreeParam(:θ1, init=0.3)
R[3, 1] = FreeParam(:θ2, init=0.1)

# Innovation variance: Q = σ²
Q = [FreeParam(:var, init=1.0, lower=0.0)]

# Initial state (diffuse)
a1 = zeros(r)
P1 = 1e7 * Matrix(1.0I, r, r)

# Build the specification
spec_manual = custom_ssm(
    Z = Z,
    H = H,
    T = T,
    R = R,
    Q = Q,
    a1 = a1,
    P1 = P1,
    name = :ARMA22_manual
)

println("Model: ", spec_manual.name)
println("State dimension: ", spec_manual.n_states)
println("Parameters: ", param_names(spec_manual))
```

Output:
```
Model: ARMA22_manual
State dimension: 3
Parameters: [:φ1, :φ2, :θ1, :θ2, :var]
```

## Using Matrix Helpers

We can simplify the construction using matrix helpers:

```julia
using Siphon

p, q = 2, 2
r = max(p, q + 1)

# Using helper functions
spec_helpers = custom_ssm(
    Z = [1.0 0.0 0.0],                    # First state observed
    H = zeros_mat(1, 1),                   # No observation noise
    T = [FreeParam(:φ1, init=0.5, lower=-0.99, upper=0.99)  1.0  0.0;
         FreeParam(:φ2, init=0.2, lower=-0.99, upper=0.99)  0.0  1.0;
         0.0                                                 0.0  0.0],
    R = [1.0;
         FreeParam(:θ1, init=0.3);
         FreeParam(:θ2, init=0.1)],
    Q = scalar_free(:var; init=1.0),
    a1 = [0.0, 0.0, 0.0],
    P1 = 1e7 * identity_mat(3),
    name = :ARMA22_helpers
)
```

## Using the Built-in ARMA Template

Siphon.jl provides the `arma` template that handles all this automatically:

```julia
using Siphon

# Create ARMA(2,2) specification
spec_template = arma(2, 2; ar_init=[0.5, 0.2], ma_init=[0.3, 0.1], var_init=1.0)

println("Model: ", spec_template.name)
println("State dimension: ", spec_template.n_states)
println("Parameters: ", param_names(spec_template))
```

The template creates parameters named `φ1`, `φ2` (AR coefficients), `θ1`, `θ2` (MA coefficients), and `var` (innovation variance).

## Simulating and Estimating

Let's simulate data from a known ARMA(2,2) process and estimate the parameters:

```julia
using Siphon
using Random
using Statistics

Random.seed!(123)

# True parameters
φ1_true, φ2_true = 0.7, -0.2
θ1_true, θ2_true = 0.4, 0.1
var_true = 1.5

# Simulate ARMA(2,2) data
function simulate_arma22(n, φ1, φ2, θ1, θ2, σ²)
    y = zeros(n)
    ε = sqrt(σ²) * randn(n + 100)  # Pre-sample for burn-in

    for t in 3:(n + 100)
        y_idx = t - 100
        if y_idx >= 1
            y[y_idx] = (t >= 3 ? φ1 * y[max(1, y_idx-1)] : 0.0) +
                       (t >= 4 ? φ2 * y[max(1, y_idx-2)] : 0.0) +
                       ε[t] + θ1 * ε[t-1] + θ2 * ε[t-2]
        end
    end
    return y
end

n = 500
y_sim = simulate_arma22(n, φ1_true, φ2_true, θ1_true, θ2_true, var_true)

# Reshape for Siphon (expects p × n matrix)
y = reshape(y_sim, 1, n)

# Create and fit model with MLE
spec = arma(2, 2; ar_init=[0.5, 0.0], ma_init=[0.0, 0.0], var_init=1.0)
model = StateSpaceModel(spec, n)
fit!(MLE(), model, y)

# Access estimated parameters
params = parameters(model)
println("\nEstimated parameters (MLE):")
println("  φ1: ", round(params.φ1, digits=4), " (true: $φ1_true)")
println("  φ2: ", round(params.φ2, digits=4), " (true: $φ2_true)")
println("  θ1: ", round(params.θ1, digits=4), " (true: $θ1_true)")
println("  θ2: ", round(params.θ2, digits=4), " (true: $θ2_true)")
println("  var: ", round(params.var, digits=4), " (true: $var_true)")
println("\nLog-likelihood: ", round(loglikelihood(model), digits=2))
```

## Accessing Fitted Matrices

After fitting, you can access the system matrices:

```julia
# Get all matrices at once (efficient)
mats = system_matrices(model)

println("\nFitted matrices:")
println("Z (observation):")
display(mats.Z)

println("\nT (transition):")
display(mats.T)

println("\nR (selection):")
display(mats.R)

println("\nQ (state covariance):")
display(mats.Q)

# Or access individually
Z = obs_matrix(model)
H = obs_cov(model)
T_mat = transition_matrix(model)
R_mat = selection_matrix(model)
Q_mat = state_cov(model)
```

## Parameter Vector vs NamedTuple

After estimation, parameters are available in two forms:

```julia
# As NamedTuple (recommended - clear names)
params = parameters(model)
println("φ1 = ", params.φ1)
println("θ1 = ", params.θ1)

# As Vector (internal representation, same order as param_names(spec))
θ_vec = model.theta_values
names = param_names(spec)
for (name, val) in zip(names, θ_vec)
    println("$name = $val")
end
```

## Comparison: MLE vs EM

For ARMA models, both MLE and EM estimation are available:

```julia
# MLE estimation (gradient-based optimization)
model_mle = StateSpaceModel(spec, n)
fit!(MLE(), model_mle, y)

# EM estimation (iterative)
model_em = StateSpaceModel(spec, n)
fit!(EM(), model_em, y; maxiter=200, verbose=false)

println("\nComparison (MLE vs EM):")
params_mle = parameters(model_mle)
params_em = parameters(model_em)

for name in param_names(spec)
    mle_val = getproperty(params_mle, name)
    em_val = getproperty(params_em, name)
    println("  $name: MLE=$(round(mle_val, digits=4)), EM=$(round(em_val, digits=4))")
end

println("\nLog-likelihoods:")
println("  MLE: ", round(loglikelihood(model_mle), digits=2))
println("  EM:  ", round(loglikelihood(model_em), digits=2))
```

## Complete Working Example

Here is a complete, self-contained example:

```julia
using Siphon
using Random
using LinearAlgebra

Random.seed!(42)

# 1. Generate synthetic ARMA(2,2) data
n = 300
ε = randn(n + 10)
y = zeros(n)
φ1, φ2 = 0.6, -0.15
θ1, θ2 = 0.3, 0.1
σ² = 2.0

for t in 3:n
    y[t] = φ1 * y[t-1] + φ2 * y[t-2] + sqrt(σ²) * ε[t+10] +
           θ1 * sqrt(σ²) * ε[t+9] + θ2 * sqrt(σ²) * ε[t+8]
end
y_mat = reshape(y, 1, n)

# 2. Create model specification
spec = arma(2, 2)

# 3. Create and fit model
model = StateSpaceModel(spec, n)
fit!(MLE(), model, y_mat)

# 4. Results
println("Fitted ARMA(2,2) parameters:")
params = parameters(model)
println("  φ1 = ", round(params.φ1, digits=4))
println("  φ2 = ", round(params.φ2, digits=4))
println("  θ1 = ", round(params.θ1, digits=4))
println("  θ2 = ", round(params.θ2, digits=4))
println("  var = ", round(params.var, digits=4))
println("\nLog-likelihood: ", round(loglikelihood(model), digits=2))

# 5. Get smoothed states
α = smoothed_states(model)
println("\nSmoothed state dimension: ", size(α))

# 6. Get system matrices
mats = system_matrices(model)
println("\nTransition matrix T:")
display(round.(mats.T, digits=4))
```

## Tips and Best Practices

1. **Initial values matter**: ARMA models can have multiple local optima. Good initial values help find the global optimum.

2. **Stationarity**: The `arma` template does not enforce stationarity constraints on AR coefficients. For guaranteed stationarity, use bounds like `lower=-0.99, upper=0.99` for AR coefficients.

3. **Invertibility**: Similarly, MA coefficients are not constrained for invertibility. Check your fitted coefficients satisfy the invertibility conditions.

4. **Model order selection**: Use information criteria (AIC, BIC) computed from the log-likelihood to select appropriate (p,q) values.

5. **EM vs MLE**: MLE is generally faster for ARMA models. EM can be useful when there are many missing observations.

## Next Steps

- Learn about [Custom Models](custom_models.md) for more complex specifications
- See [Parameter Transformations](transformations.md) for constrained optimization
- Explore [Dynamic Factor Models](dynamic_factor.md) for multivariate time series
