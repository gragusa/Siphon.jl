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
from `builder.jl`. The transformation, codegen, and optimisation layers in
later files dispatch on `EKFSpec` so the DSL surface (`build_transformation`,
`transform_to_constrained`, `transform_to_unconstrained`,
`build_initial_state`, `optimize_*`) is symmetric with the linear case.
"""

using ..Siphon: AbstractEKFMeasurement, AbstractEKFJacobianMode, ADJacobian,
                AnalyticJacobian, EKFParms

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
- `matrix_exprs::Dict{Symbol,Any}` — expression-based system matrices.
"""
struct EKFSpec{ME, JM <: AbstractEKFJacobianMode}
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
    matrix_exprs::Dict{Symbol, Any}
end

# ============================================================================
# Introspection methods (mirror SSMSpec)
# ============================================================================

param_names(spec::EKFSpec) = [p.name for p in spec.params]
n_params(spec::EKFSpec) = length(spec.params)
initial_values(spec::EKFSpec) = [p.init for p in spec.params]
function param_bounds(spec::EKFSpec)
    lower = [p.lower for p in spec.params]
    upper = [p.upper for p in spec.params]
    (lower, upper)
end
function param_index(spec::EKFSpec, name::Symbol)
    idx = findfirst(p -> p.name == name, spec.params)
    idx === nothing && throw(ArgumentError("Unknown parameter: $name"))
    idx
end

# Approximate diffuse only for now. Plan §5 question 4: exact diffuse for EKF
# can be added later when the linear DiffuseKalmanWorkspace ideas are ported.
uses_exact_diffuse(::EKFSpec) = false

# ============================================================================
# build_transformation / transform_* for EKFSpec
# ============================================================================

# Reuse `_param_to_transform` from codegen.jl. Its argument is a single
# SSMParameter, so the same implementation works.
function build_transformation(spec::EKFSpec)
    transforms = [_param_to_transform(p) for p in spec.params]
    names = Tuple(p.name for p in spec.params)
    as(NamedTuple{names}(Tuple(transforms)))
end

function transform_to_constrained(spec::EKFSpec, θ_u::AbstractVector)
    t = build_transformation(spec)
    TransformVariables.transform_and_logjac(t, θ_u)
end

function transform_to_unconstrained(spec::EKFSpec, θ_nt::NamedTuple)
    t = build_transformation(spec)
    TransformVariables.inverse(t, θ_nt)
end

function transform_to_unconstrained(spec::EKFSpec, θ_c::AbstractVector)
    names = Tuple(p.name for p in spec.params)
    θ_nt = NamedTuple{names}(Tuple(θ_c))
    transform_to_unconstrained(spec, θ_nt)
end
