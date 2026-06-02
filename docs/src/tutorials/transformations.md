# Parameter Transformations

This tutorial explains how Siphon.jl handles parameter transformations for constrained optimization. Understanding this system is important for:

- Correctly specifying parameter bounds
- Interpreting optimization results
- Working with Bayesian inference
- Implementing custom models

## Overview

State-space model parameters often have natural constraints:

- **Variance parameters** must be positive: ``\sigma^2 > 0``
- **AR coefficients** must satisfy stationarity: ``|\rho| < 1``
- **Correlation parameters** must be in ``[-1, 1]``
- **Probabilities** must be in ``[0, 1]``

Siphon.jl uses [TransformVariables.jl](https://github.com/tpapp/TransformVariables.jl) to handle these constraints automatically. The key idea is to optimize in an **unconstrained space** ``\mathbb{R}^n`` and transform to the constrained parameter space when needed.

## How It Works

### The Two Spaces

1. **Constrained space** (``\Theta``): Where parameters have their natural interpretation
   - Example: ``\sigma^2_{\text{obs}} = 225.0`` (a variance)

2. **Unconstrained space** (``\mathbb{R}^n``): Where optimization actually happens
   - Example: ``\theta_{\text{unconstrained}} = 5.42`` (the log of ``\sigma^2``)

### Transformation Flow

```
                    transform
Unconstrained θ_u ─────────────> Constrained θ_c (NamedTuple)
      ℝⁿ           (+ logjac)          Θ
                                       │
                                       ▼
                              Build state-space matrices
                                       │
                                       ▼
                              Compute log-likelihood
```

The optimizer works in unconstrained space. For each evaluation:
1. Transform ``\theta_u \to \theta_c`` (unconstrained to constrained)
2. Build state-space matrices using ``\theta_c``
3. Compute log-likelihood
4. (For Bayesian: add log Jacobian and log prior)

## Specifying Parameter Bounds

When you create a `FreeParam`, the bounds determine which transformation is applied:

```julia
# Positive parameter (variance) - uses exp transform
FreeParam(:var_obs, init=100.0, lower=0.0)

# Bounded parameter (AR coefficient) - uses logit-like transform
FreeParam(:ρ, init=0.8, lower=-0.99, upper=0.99)

# Unbounded parameter - identity (no transform)
FreeParam(:β, init=0.0)
```

### Transformation Rules

| Bounds | Transform | Mathematical Form |
|--------|-----------|-------------------|
| ``(-\infty, \infty)`` | Identity | ``\theta_c = \theta_u`` |
| ``[0, \infty)`` | Exponential | ``\theta_c = \exp(\theta_u)`` |
| ``(a, b)`` | Scaled logistic | ``\theta_c = a + (b-a) \cdot \text{logistic}(\theta_u)`` |
| ``(-\infty, b]`` | Negative exp | ``\theta_c = b - \exp(-\theta_u)`` |

### For Variance Parameters

Use `lower=0.0` to ensure positivity:

```julia
# Correct: estimate variance directly with positivity constraint
H = [FreeParam(:var_obs, init=100.0, lower=0.0)]
```

The transformation automatically applied is:
```math
\sigma^2 = \exp(\theta_u)
```

So if the optimizer finds ``\theta_u = 4.6``, you get ``\sigma^2 = \exp(4.6) \approx 100``.

## Working with Transformations Directly

### Building Transformations

```julia
using Siphon

spec = local_level()
t = build_transformation(spec)

# Transform from unconstrained to constrained
θ_u = [4.6, 3.9]  # Unconstrained values
θ_c = TransformVariables.transform(t, θ_u)
# θ_c = (var_obs = 99.5, var_level = 49.4)
```

### Transform with Jacobian

For Bayesian inference, you need the log Jacobian determinant:

```julia
using TransformVariables

θ_c, logjac = transform_and_logjac(t, θ_u)
# logjac accounts for the change of variables
```

### Inverse Transform

To go from constrained back to unconstrained:

```julia
θ_u = transform_to_unconstrained(spec, θ_c)
```

## Full Covariance Matrices

For full positive-definite covariance matrices, Siphon.jl uses a special parameterization via `cov_free`:

```julia
Q = cov_free(2, :Q)  # 2×2 PD covariance matrix
```

This creates:
- **Standard deviation parameters**: `Q_σ_1`, `Q_σ_2` (positive, use exp transform)
- **Correlation parameters**: `Q_corr_1` (unconstrained, maps to valid correlation)

The matrix is reconstructed as:
```math
\Sigma = D \cdot \text{Corr} \cdot D
```

where ``D = \text{diag}(\sigma_1, \sigma_2)`` and ``\text{Corr}`` is built from a Cholesky factor parameterization that guarantees positive definiteness.

## Example: Complete Workflow

```julia
using Siphon

# Specify model with constrained parameters
spec = custom_ssm(
    Z = [1.0],
    H = [FreeParam(:var_obs, init=100.0, lower=0.0)],  # Positive
    T = [FreeParam(:ρ, init=0.8, lower=-0.99, upper=0.99)],  # Bounded
    R = [1.0],
    Q = [FreeParam(:var_state, init=50.0, lower=0.0)],  # Positive
    a1 = [0.0],
    P1 = [1e7]
)

# Simulate data
y = randn(1, 200)

# Optimize - all transformation handled automatically
result = optimize_ssm(spec, y)

# Result is in constrained space
println("var_obs = ", result.θ.var_obs)    # Positive value
println("ρ = ", result.θ.ρ)                # In (-0.99, 0.99)
println("var_state = ", result.θ.var_state)  # Positive value

# If you need unconstrained values (e.g., for MCMC initialization)
θ_u = transform_to_unconstrained(spec, result.θ)
println("Unconstrained: ", θ_u)  # Values in ℝ³
```

## Bayesian Inference

For Bayesian inference with MCMC, the log-density is computed in unconstrained space:

```julia
# Create log-density object
ld = SSMLogDensity(spec, y)

# Evaluate at unconstrained point
θ_u = randn(n_params(spec))
ll = logdensity(ld, θ_u)
# ll includes the log Jacobian automatically
```

The `SSMLogDensity` type implements `LogDensityProblems.jl` interface, so you can use it with any compatible sampler.

## Tips and Best Practices

1. **Always use `lower=0.0` for variances**: This ensures the exp transform is applied.

2. **Use tight bounds for AR coefficients**: `lower=-0.99, upper=0.99` works better than `(-1, 1)` numerically.

3. **Initial values matter**: Provide good initial values in the constrained space. They are automatically transformed.

4. **Standard errors are approximate**: When using `optimize_ssm_with_stderr`, the standard errors are computed via the delta method and may be approximate for highly nonlinear transformations.

5. **For debugging**: Use `transform_to_unconstrained` and `transform_to_constrained` to verify parameter values at each stage.

## API Reference

### Key Functions

```julia
# Build transformation from spec
t = build_transformation(spec)

# Transform unconstrained → constrained
θ_c, logjac = transform_to_constrained(spec, θ_u)

# Transform constrained → unconstrained
θ_u = transform_to_unconstrained(spec, θ_c)
```

### Related Types

- `FreeParam`: Specify a free parameter with bounds
- `SSMLogDensity`: Log-density in unconstrained space
- `CovMatrixExpr`: Expression for positive-definite covariance matrices
