"""
    dsl.jl

Domain-specific language for ergonomic state-space model specification.

This module provides:
- Type definitions for model specification (SSMSpec, SSMParameter)
- Code generation to build KFParms and initial state
- TransformVariables integration for constrained optimization
- Pre-built templates for common models (local level, local trend, AR, ARMA)
- Optimization via Optimization.jl

# Example Usage

```julia
using Siphon

# Use a pre-built template
spec = local_level(σ_obs_init=15.0, σ_level_init=10.0)

# Get parameter info
println("Parameters: ", param_names(spec))
println("Initial values: ", initial_values(spec))

# Build transformation (ℝⁿ → NamedTuple with constraints)
t = build_transformation(spec)

# Transform parameters
θ_nt = TransformVariables.transform(t, randn(2))
# θ_nt is (σ_obs = ..., σ_level = ...) with positive values

# Compute log-likelihood with NamedTuple
ll = kalman_loglik(spec, θ_nt, y)

# Or use optimization (handles constraints automatically)
result = optimize_ssm(spec, y)
# result.θ is a NamedTuple
```

# Pre-built Models

- `local_level()`: Random walk plus noise
- `local_linear_trend()`: Random walk with drift
- `ar1()`: AR(1) plus measurement noise
- `arma(p, q)`: ARMA(p,q) in state-space form
"""
module DSL

using LinearAlgebra
using TransformVariables
using ..Siphon: KFParms, KFParms_static, kalman_loglik
using ..Siphon: kalman_filter, kalman_smoother, ismissing_obs
using ..Siphon: to_static_if_small, STATIC_THRESHOLD

include("types.jl")
include("codegen.jl")
include("templates.jl")
include("builder.jl")
include("matrix_helpers.jl")
include("expressions.jl")
include("bayesian.jl")
include("optimization.jl")
include("em.jl")

# Re-export key functions
export SSMParameter, SSMSpec, FixedValue, ParameterRef, SSMMatrixSpec
export param_names, n_params, initial_values, param_bounds, param_index
export uses_exact_diffuse
export build_kfparms, build_initial_state, build_linear_state_space
export objective_function, ssm_loglik
export local_level, local_linear_trend, ar1, arma, dynamic_factor, dns_model
export custom_ssm, FreeParam, @P

# Matrix helpers
export diag_free, scalar_free, diag_fixed
export identity_mat, zeros_mat, ones_mat
export lower_triangular_free, symmetric_free
export selection_mat, companion_mat
export cov_free, CovFree, CovMatrixExpr

# Functional expressions (for DNS, Svensson, etc.)
export ParamExpr, MatrixExpr
export build_dns_loadings, build_svensson_loadings
export dns_loading1, dns_loading2

# Transformation (TransformVariables.jl integration)
export build_transformation
export transform_to_constrained, transform_to_unconstrained

# Log-density for optimization/sampling (LogDensityProblems.jl integration)
export SSMLogDensity, logdensity
export FlatPrior, NormalPrior, NormalPriorVec, InverseGammaPrior, CompositePrior

# Optimization (Optimization.jl integration)
export optimize_ssm, optimize_ssm_with_stderr

# EM algorithm - main API is fit!(EM(), model, y)
export EMResult, profile_em_ssm, ProfileEMResult
export _mstep_Z, _mstep_T, _mstep_H_diag, _mstep_Q_diag, _mstep_H_full, _mstep_Q_full

end # module DSL
