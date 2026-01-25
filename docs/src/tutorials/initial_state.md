# Initial State Conventions and MARSS Compatibility

This page explains how Siphon.jl handles the initial state distribution and how it relates to the MARSS R package conventions.

## State Space Model Formulation

Siphon.jl uses the standard state space formulation:

```
Observation:  yₜ = Z αₜ + εₜ,    εₜ ~ N(0, H)
State:        αₜ₊₁ = T αₜ + R ηₜ,  ηₜ ~ N(0, Q)
Initial:      α₁ ~ N(a₁, P₁)
```

The Kalman filter processes observations starting at t=1, so it needs the initial state distribution `(a₁, P₁)` at time t=1.

## MARSS `tinitx` Parameter

The MARSS R package uses a `tinitx` parameter to specify when the initial state is defined:

### `tinitx=0` (MARSS Default)

The initial state `(x₀, V₀)` is specified at time **t=0** (before the first observation). MARSS internally propagates it forward one time step:

```
a₁ = T × x₀
P₁ = T × V₀ × T' + R × Q × R'
```

This incorporates one cycle of state dynamics into the initial covariance at t=1.

### `tinitx=1`

The initial state `(x₀, V₀)` is specified directly at time **t=1**:

```
a₁ = x₀
P₁ = V₀
```

No transformation is applied.

## Siphon.jl Convention

Siphon.jl **always uses the `tinitx=1` convention** internally. You specify `(a₁, P₁)` directly as the state distribution at time t=1.

## Mapping MARSS to Siphon

To match MARSS behavior in Siphon.jl:

| MARSS Setting | Siphon Equivalent |
|---------------|-------------------|
| `tinitx=0`, `x0`, `V0` | `a1 = T * x0`, `P1 = T * V0 * T' + R * Q * R'` |
| `tinitx=1`, `x0`, `V0` | `a1 = x0`, `P1 = V0` |

### Example: Matching MARSS tinitx=0

```julia
using Siphon
using LinearAlgebra

# MARSS uses tinitx=0 with these defaults:
x0 = zeros(m)           # State mean at t=0
V0 = 100.0 * I(m)       # State covariance at t=0

# Your model parameters
T = [0.9 0.0; 0.0 0.8]  # Transition matrix
Q = [0.1 0.0; 0.0 0.2]  # State covariance
R = I(m)                # Selection matrix

# Convert to Siphon convention (tinitx=1):
a1 = T * x0                           # = zeros(m) if x0 = 0
P1 = T * V0 * T' + R * Q * R'         # Incorporates dynamics

# Now use a1 and P1 in Siphon
kf = KFParms(Z, H, T, R, Q)
ll = kalman_loglik(kf, y, a1, P1)
```

## Initial State in EM Algorithm

The EM algorithm can either keep the initial state fixed or update it at each iteration.

### Fixed Initial State (Default)

With `update_initial_state=false` (the default), `(a₁, P₁)` remains **unchanged** throughout all EM iterations:

| EM Iteration | Initial State | Parameters |
|--------------|---------------|------------|
| 0 (start)    | `a1`, `P1` from input | `T₀`, `Q₀`, `H₀` from input |
| 1            | Same `a1`, `P1` | Updated `T₁`, `Q₁`, `H₁` |
| 2            | Same `a1`, `P1` | Updated `T₂`, `Q₂`, `H₂` |
| ...          | Same `a1`, `P1` | ... |

**Use this when:**
- Using a diffuse prior (large P₁)
- Long time series where t=1 has minimal effect
- Numerical stability is a concern

### Updated Initial State (MARSS-style)

With `update_initial_state=true`, `(a₁, P₁)` is **updated at each M-step** using the smoothed state estimates:

```
a₁_new = E[α₁ | y₁:n]      (smoothed state mean at t=1)
P₁_new = Var[α₁ | y₁:n]    (smoothed state covariance at t=1)
```

| EM Iteration | Initial State | Parameters |
|--------------|---------------|------------|
| 0 (start)    | `a1`, `P1` from input | `T₀`, `Q₀`, `H₀` from input |
| 1            | Updated from smoother | Updated `T₁`, `Q₁`, `H₁` |
| 2            | Updated from smoother | Updated `T₂`, `Q₂`, `H₂` |
| ...          | ... | ... |

**Use this when:**
- Short time series where t=1 significantly affects the likelihood
- Comparing results with MARSS
- Estimating the unconditional mean/variance of the state process

### Example: EM with Initial State Updating

```julia
using Siphon
using Siphon.DSL: profile_em_ssm, dns_model

# Create DNS model
maturities = [3, 12, 24, 60, 120]
spec = dns_model(maturities)

# Run EM with initial state updating
result = profile_em_ssm(spec, yields;
    update_initial_state=true,
    verbose=true
)

# Access final initial state estimates
println("Final a1: ", result.em_result.a1)
println("Final P1: ", result.em_result.P1)
```

## Using `tinitx` and `V0` Parameters

Both `profile_em_ssm` (for DNS models) and `DynamicFactorModel` support the `tinitx` and `V0` parameters for controlling initial state covariance:

### `tinitx` Parameter

- **`tinitx=0` (default):** V0 is the covariance at t=0. P1 is computed as:
  ```julia
  P1 = T * V0 * T' + R * Q * R'
  ```
  This incorporates one step of state dynamics into P1.

- **`tinitx=1`:** V0 is the covariance at t=1. P1 = V0 directly (no transformation).

### Examples

```julia
# DNS models via profile_em_ssm
result = profile_em_ssm(spec, y; tinitx=0, V0=100.0)  # Default: MARSS-style
result = profile_em_ssm(spec, y; tinitx=1, V0=1e7)    # Diffuse prior at t=1

# Dynamic Factor Models
model = DynamicFactorModel(N, k, n; tinitx=0, V0=100.0)  # Default
model = DynamicFactorModel(N, k, n; tinitx=1, V0=1e7)    # Diffuse prior at t=1

# V0 can also be a matrix (for profile_em_ssm)
V0_mat = Diagonal([100.0, 200.0, 300.0])
result = profile_em_ssm(spec, y; tinitx=1, V0=V0_mat)
```

### Choosing `tinitx`

| Use Case | Recommended Setting |
|----------|---------------------|
| Match MARSS default | `tinitx=0, V0=100.0` |
| Diffuse prior (large uncertainty) | `tinitx=1, V0=1e7` |
| Informative prior at t=1 | `tinitx=1, V0=<your value>` |
| Short time series | `tinitx=0` (accounts for dynamics) |

**Note:** With `tinitx=0`, very large V0 values (e.g., 1e7) may cause numerical instability because the transformation `T * V0 * T'` still produces large values. With `tinitx=1`, large V0 values are used directly and work well for diffuse priors.

## Numerical Comparison

At identical parameter values, Siphon.jl and MARSS produce matching log-likelihoods:

```julia
# At MARSS converged parameters for DNS model:
# MARSS log-likelihood:  430.032178464672
# Siphon log-likelihood: 430.032178461962
# Difference: ~2.7e-9 (numerical precision)
```

## Summary

| Aspect | MARSS tinitx=0 | MARSS tinitx=1 | Siphon tinitx=0 | Siphon tinitx=1 |
|--------|----------------|----------------|-----------------|-----------------|
| Initial state timing | t=0 | t=1 | t=0 | t=1 |
| a₁ formula | T × x₀ | x₀ | zeros | zeros |
| P₁ formula | T × V₀ × T' + R × Q × R' | V₀ | T × V₀ × T' + R × Q × R' | V₀ |
| EM update | Optional | Optional | `update_initial_state=true` | `update_initial_state=true` |

Siphon now supports both conventions via the `tinitx` parameter:
- `tinitx=0` (default): Matches MARSS tinitx=0 behavior
- `tinitx=1`: Matches MARSS tinitx=1 behavior
