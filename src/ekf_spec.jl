"""
    ekf_spec.jl

DSL types for nonlinear-measurement state-space models (EKF), parallel to
`SSMSpec` for linear models.

`EKFSpec` carries the same kind of `SSMMatrixSpec`/`MatrixExpr`/`CovFree`
payloads for `H`, `T`, `R`, `Q`, and `P1` as `SSMSpec`, plus a measurement
description that is one of:

  - a fixed concrete `AbstractEKFMeasurement` (no estimable measurement
    parameters),
  - a `MeasurementExpr` (parameters + builder + data) so the measurement
    object is reconstructed each evaluation from the constrained `θ`.

Reuses `SSMParameter`, `SSMMatrixSpec`, and the matrix-input processing helpers
from `builder.jl`. `EKFSpec` and `SSMSpec` share the supertype
`AbstractSSMSpec`, which carries parameter introspection and the
transformations (`build_transformation`, `transform_to_constrained`,
`transform_to_unconstrained`). The codegen and optimisation files add `EKFSpec`
methods of `build_initial_state`, `ssm_loglik` and `optimize_ssm`.
"""

# ============================================================================
# MeasurementExpr
# ============================================================================

"""
    MeasurementExpr{F, D}

Measurement-model expression, parallel to `MatrixExpr` for matrices.

A `MeasurementExpr` is reconstructed at each likelihood evaluation:

```julia
expr.builder(θ_dict, expr.data) -> AbstractEKFMeasurement
```

where `θ_dict::Dict{Symbol, T}` carries values of the measurement parameters
declared in `expr.params` (typed by AD eltype `T`) and `expr.data` is any
context the builder needs (loadings matrix, lower bound, maturity grid).

Use a `MeasurementExpr` when the measurement object depends on free parameters
(e.g. a kink location, a lower bound, or a smoothing scale to be estimated).
For a fully fixed measurement, just pass the `AbstractEKFMeasurement` directly
to `custom_ekf` instead.
"""
struct MeasurementExpr{F, D}
    params::Vector{SSMParameter{Float64}}
    builder::F
    data::D
end

# Convenience constructor with kw args
function MeasurementExpr(; params::Vector{<:SSMParameter} = SSMParameter{Float64}[],
        builder, data = nothing)
    MeasurementExpr{typeof(builder), typeof(data)}(
        Vector{SSMParameter{Float64}}(params), builder, data)
end

# ============================================================================
# EKFSpec
# ============================================================================

"""
    EKFSpec

Specification of an Extended-Kalman-Filter state-space model. Mirrors
`SSMSpec` field-for-field except:

  - The constant observation matrix `Z` is replaced by a measurement
    description `measurement::Union{AbstractEKFMeasurement, MeasurementExpr}`.
  - A `jacobian_mode::AbstractEKFJacobianMode` is stored so downstream code
    knows whether to differentiate `measurement(model, a, t)` (`ADJacobian()`)
    or call user-supplied `measurement_jacobian` methods (`AnalyticJacobian()`).

# Fields
- `name::Symbol` — model name.
- `n_states::Int`, `n_obs::Int`, `n_shocks::Int` — dimensions.
- `params::Vector{SSMParameter}` — all estimable parameters (matrix params
  from `H`, `T`, `R`, `Q`, `P1`, plus `MeasurementExpr.params`).
- `measurement` — fixed `AbstractEKFMeasurement` or `MeasurementExpr`.
- `jacobian_mode::AbstractEKFJacobianMode`.
- `H, T, R, Q, P1` — `SSMMatrixSpec`s (placeholder dims for `MatrixExpr`s).
- `a1::Vector{MatrixElement}` — initial-state mean.
- `matrix_exprs::Dict{Symbol,AbstractMatrixExpr}` — expression-based system matrices.
"""
struct EKFSpec{ME, JM <: AbstractEKFJacobianMode} <: AbstractSSMSpec
    name::Symbol
    n_states::Int
    n_obs::Int
    n_shocks::Int
    params::Vector{SSMParameter{Float64}}
    measurement::ME
    jacobian_mode::JM
    H::SSMMatrixSpec
    T::SSMMatrixSpec
    R::SSMMatrixSpec
    Q::SSMMatrixSpec
    a1::Vector{MatrixElement}
    P1::SSMMatrixSpec
    matrix_exprs::Dict{Symbol, AbstractMatrixExpr}
end

# EKF models use approximate diffuse initialization only.
uses_exact_diffuse(::EKFSpec) = false
