"""
    ekf_codegen.jl

Code generation for `EKFSpec`: build `EKFParms`, build the initial state, and
the `custom_ekf` constructor. Reuses the linear-DSL machinery from `codegen.jl`
and `builder.jl` for system matrices and free parameters; only the measurement
description is new.
"""

# ============================================================================
# Build the measurement object from constrained θ
# ============================================================================

"""
    build_measurement(spec::EKFSpec, θ::NamedTuple) -> AbstractEKFMeasurement

If `spec.measurement` is already an `AbstractEKFMeasurement`, return it
unchanged. If it is a `MeasurementExpr`, call its builder with a Dict of the
relevant parameter values.
"""
function build_measurement(spec::EKFSpec, θ::NamedTuple)
    return _build_measurement(spec.measurement, θ)
end

@inline function _build_measurement(m::AbstractEKFMeasurement, θ::NamedTuple)
    return m
end

function _build_measurement(expr::MeasurementExpr, θ::NamedTuple)
    T = _eltype(θ)
    θ_dict = Dict{Symbol, T}()
    for p in expr.params
        if haskey(θ, p.name)
            θ_dict[p.name] = getproperty(θ, p.name)
        end
    end
    return expr.builder(θ_dict, expr.data)
end

# ============================================================================
# build_ekfparms
# ============================================================================

"""
    build_ekfparms(spec::EKFSpec, θ::NamedTuple) -> EKFParms

Build the EKF parameter container from constrained `θ`. Reuses
`build_matrix_or_expr` for `H`, `T`, `R`, `Q` and `build_measurement` for the
measurement model.
"""
function build_ekfparms(spec::EKFSpec, θ::NamedTuple)
    measurement = build_measurement(spec, θ)
    H = build_matrix_or_expr_ekf(:H, spec, θ)
    Tr = build_matrix_or_expr_ekf(:T, spec, θ)
    R = build_matrix_or_expr_ekf(:R, spec, θ)
    Q = build_matrix_or_expr_ekf(:Q, spec, θ)
    return EKFParms(measurement, spec.jacobian_mode, H, Tr, R, Q)
end

# Mirrors build_matrix_or_expr but reads from spec.matrix_exprs / fields of EKFSpec.
function build_matrix_or_expr_ekf(name::Symbol, spec::EKFSpec, θ::NamedTuple)
    if haskey(spec.matrix_exprs, name)
        return build_from_expr(spec.matrix_exprs[name], θ)
    else
        mat_spec = getfield(spec, name)
        return build_matrix(mat_spec, θ)
    end
end

# ============================================================================
# build_initial_state
# ============================================================================

"""
    build_initial_state(spec::EKFSpec, θ::NamedTuple) -> (a1, P1)

Build initial-state mean and covariance from constrained `θ`. Same convention
as the linear DSL: approximate diffuse via a finite `P1`.
"""
function build_initial_state(spec::EKFSpec, θ::NamedTuple)
    T = _eltype(θ)
    a1 = T[evaluate_element(elem, θ) for elem in spec.a1]
    if haskey(spec.matrix_exprs, :P1)
        P1 = build_from_expr(spec.matrix_exprs[:P1], θ)
    else
        P1 = build_matrix(spec.P1, θ)
    end
    return (a1, P1)
end

# ============================================================================
# build_nonlinear_state_space (parallel to build_linear_state_space)
# ============================================================================

"""
    build_nonlinear_state_space(spec::EKFSpec, θ, y)

Build the EKF parameter container, initial state, and observation matrix
needed to evaluate `kalman_loglik`. Returns a NamedTuple `(p, a1, P1)`.

Unlike `build_linear_state_space`, it takes no `static` keyword: the
measurement object owns the parts that StaticArrays would convert (e.g.
loadings), so the EKF path does not convert them.
"""
function build_nonlinear_state_space(spec::EKFSpec, θ::NamedTuple, y::AbstractMatrix)
    p = build_ekfparms(spec, θ)
    a1, P1 = build_initial_state(spec, θ)
    return (; p, a1, P1)
end

"""
    ssm_loglik(spec::EKFSpec, θ::NamedTuple, y::AbstractMatrix)

Evaluate the extended Kalman filter log-likelihood of an `EKFSpec` at constrained
parameters `θ`. Equivalent to `kalman_loglik(p, y, a1, P1)` after building
`(p, a1, P1)` from the spec.
"""
function ssm_loglik(spec::EKFSpec, θ::NamedTuple, y::AbstractMatrix)
    _spec_loglik(spec, θ, _nan_obs(y), :off)
end

# ============================================================================
# custom_ekf
# ============================================================================

"""
    custom_ekf(; measurement, H, T, R, Q, a1, P1,
               jacobian_mode = ADJacobian(),
               name = :CustomEKF) -> EKFSpec

Construct an `EKFSpec` from explicit matrices and a measurement description.
Parallel to `custom_ssm` for the linear case.

# Arguments
- `measurement` — either a fixed `AbstractEKFMeasurement` or a
  `MeasurementExpr` (parameters + builder + data).
- `H, T, R, Q` — system matrices using the same DSL inputs as `custom_ssm`
  (numbers, `FreeParam`, `MatrixExpr`, `cov_free`, etc.). `T`, `R`, `Q` and `P1`
  may be a scaled identity `I`; `H` must be sized explicitly.
- `a1` — initial-state mean (numbers and/or `FreeParam`).
- `P1` — initial-state covariance (`SSMMatrixSpec` input).
- `jacobian_mode` — `ADJacobian()` (default) or `AnalyticJacobian()`.
- `name` — model name.

The returned `EKFSpec` collects all free parameters from the matrix payloads
plus those declared on the `MeasurementExpr`; a name used more than once must
carry the same settings each time. The measurement is evaluated once, at the
initial parameter values and `a1`, to check that its length matches the size
of `H`.

# Example
```julia
using LinearAlgebra

# Yields bounded below by r_LB: h(β) = r_LB + softplus(Λβ - r_LB)
struct MyMeasurement{LT,T} <: Siphon.AbstractEKFMeasurement
    Λ::LT
    r_LB::T
end
softplus(x) = log1p(exp(x))
Siphon.measurement(m::MyMeasurement, β, t) = m.r_LB .+ softplus.(m.Λ * β .- m.r_LB)

p, m = 4, 2
Λ = [ones(p) range(1.0, 0.2; length = p)]

# The Jacobian comes from automatic differentiation (ADJacobian(), the default).
# For AnalyticJacobian(), also define measurement! and measurement_jacobian!.
spec = custom_ekf(
    measurement = MyMeasurement(Λ, 0.0),
    H = diag_free(p, :var_y; init=0.05),
    T = diag_free(m, :ψ; init=0.95, lower=-0.999, upper=0.999),
    R = I,
    Q = diag_free(m, :var_state; init=0.1),
    a1 = zeros(m),
    P1 = 100.0I,
)
```
"""
function custom_ekf(; measurement,
        H, T, R, Q, a1, P1,
        jacobian_mode::AbstractEKFJacobianMode = ADJacobian(),
        name::Symbol = :CustomEKF)
    params = SSMParameter{Float64}[]
    param_set = Set{Symbol}()
    matrix_exprs = Dict{Symbol, AbstractMatrixExpr}()

    # Size any `I` inputs: a1 gives m, R gives r.
    a1_vec = _to_vector(a1)
    T, R, P1 = map(x -> _sized(x, length(a1_vec)), (T, R, P1))
    Q = _sized(Q, _input_dims(R)[2])

    # Process system matrices (reuses linear DSL machinery)
    H_spec, H_dims = _process_matrix_input(:H, H, params, param_set, matrix_exprs)
    T_spec, T_dims = _process_matrix_input(:T, T, params, param_set, matrix_exprs)
    R_spec, R_dims = _process_matrix_input(:R, R, params, param_set, matrix_exprs)
    Q_spec, Q_dims = _process_matrix_input(:Q, Q, params, param_set, matrix_exprs)
    P1_spec, P1_dims = _process_matrix_input(:P1, P1, params, param_set, matrix_exprs)

    a1_elems = _build_vector_spec(a1_vec, params, param_set)

    # Measurement parameters
    if measurement isa MeasurementExpr
        for p in measurement.params
            _add_param!(params, param_set, p)
        end
    end

    # Infer dimensions: m from T, r from Q. n_obs = p comes from H.
    p = H_dims[1]
    m = T_dims[1]
    r = Q_dims[1]

    H_dims == (p, p) || throw(DimensionMismatch("H must be ($p, $p), got $H_dims"))
    T_dims == (m, m) || throw(DimensionMismatch("T must be ($m, $m), got $T_dims"))
    R_dims == (m, r) || throw(DimensionMismatch("R must be ($m, $r), got $R_dims"))
    Q_dims == (r, r) || throw(DimensionMismatch("Q must be ($r, $r), got $Q_dims"))
    length(a1_vec) == m ||
        throw(DimensionMismatch("a1 must have length $m, got $(length(a1_vec))"))
    P1_dims == (m, m) || throw(DimensionMismatch("P1 must be ($m, $m), got $P1_dims"))

    spec = EKFSpec{typeof(measurement), typeof(jacobian_mode)}(
        name, m, p, r,
        params, measurement, jacobian_mode,
        H_spec, T_spec, R_spec, Q_spec,
        a1_elems, P1_spec, matrix_exprs
    )
    _check_measurement_length(spec)
    return spec
end

# The measurement's output length must equal the size of H. It is evaluated at
# the initial parameter values, the initial state mean and t = 1.
function _check_measurement_length(spec::EKFSpec)
    θ = NamedTuple{Tuple(param_names(spec))}(Tuple(initial_values(spec)))
    a1, _ = build_initial_state(spec, θ)
    k = length(measurement(build_measurement(spec, θ), a1, 1))
    k == spec.n_obs || throw(DimensionMismatch(
        "the measurement returns $k values but H is $(spec.n_obs) × $(spec.n_obs)"))
    return nothing
end
