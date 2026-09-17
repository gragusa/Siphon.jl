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
    build_nonlinear_state_space(spec::EKFSpec, θ, y; use_static=true)

Build the EKF parameter container, initial state, and observation matrix
needed to evaluate `ekf_loglik`. Returns a NamedTuple `(p, a1, P1)`.

`use_static` is currently a no-op for the EKF path — the measurement object
typically owns the parts that would benefit from `SMatrix` conversion (e.g.
loadings) and we should not blindly convert user-owned data. This kwarg is
preserved for API symmetry with `build_linear_state_space`.
"""
function build_nonlinear_state_space(spec::EKFSpec, θ::NamedTuple, y::AbstractMatrix;
        use_static::Bool = true)
    p = build_ekfparms(spec, θ)
    a1, P1 = build_initial_state(spec, θ)
    return (; p, a1, P1)
end

"""
    Siphon.ekf_loglik(spec::EKFSpec, θ, y; use_static=true) -> loglik

Evaluate the EKF log-likelihood for an `EKFSpec` at constrained parameters
`θ::NamedTuple`. Equivalent to `ekf_loglik(p, y, a1, P1)` after building
`(p, a1, P1)` from the spec.

Adds a method to the parent-module `Siphon.ekf_loglik` so callers can use the
same name regardless of whether they have an `EKFParms` or an `EKFSpec`.
"""
function ekf_loglik(spec::EKFSpec, θ::NamedTuple, y::AbstractMatrix;
        use_static::Bool = true)
    ss = build_nonlinear_state_space(spec, θ, y; use_static = use_static)
    return ekf_loglik(ss.p, y, ss.a1, ss.P1)
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
  (numbers, `FreeParam`, `MatrixExpr`, `cov_free`, etc.).
- `a1` — initial-state mean (numbers and/or `FreeParam`).
- `P1` — initial-state covariance (`SSMMatrixSpec` input).
- `jacobian_mode` — `ADJacobian()` (default) or `AnalyticJacobian()`.
- `name` — model name.

The returned `EKFSpec` collects all free parameters from the matrix payloads
plus those declared on the `MeasurementExpr` (deduplicated by name).

# Example
```julia
struct MyMeasurement{LT,T} <: Siphon.AbstractEKFMeasurement; Λ::LT; r_LB::T; end
Siphon.measurement(m::MyMeasurement, β, t) = m.r_LB .+ softplus.(m.Λ * β .- m.r_LB)
# (plus measurement!/measurement_jacobian! if AnalyticJacobian() is desired)

spec = custom_ekf(
    measurement = MyMeasurement(Λ, 0.0),
    jacobian_mode = AnalyticJacobian(),
    H = diag_free(:σ_y, p; init=0.05),
    T = diag_free(:ψ, m; init=0.95, lower=-0.999, upper=0.999),
    R = identity_mat(m),
    Q = diag_free(:σ_state, m; init=0.1),
    a1 = zeros(m),
    P1 = 100.0 * I(m),
)
```
"""
function custom_ekf(; measurement,
        H, T, R, Q, a1, P1,
        jacobian_mode::AbstractEKFJacobianMode = ADJacobian(),
        name::Symbol = :CustomEKF)
    params = SSMParameter{Float64}[]
    param_set = Set{Symbol}()
    matrix_exprs = Dict{Symbol, Any}()

    # Process system matrices (reuses linear DSL machinery)
    H_spec, H_dims = _process_matrix_input(:H, H, params, param_set, matrix_exprs)
    T_spec, T_dims = _process_matrix_input(:T, T, params, param_set, matrix_exprs)
    R_spec, R_dims = _process_matrix_input(:R, R, params, param_set, matrix_exprs)
    Q_spec, Q_dims = _process_matrix_input(:Q, Q, params, param_set, matrix_exprs)
    P1_spec, P1_dims = _process_matrix_input(:P1, P1, params, param_set, matrix_exprs)

    # a1
    a1_vec = _to_vector(a1)
    a1_elems = _build_vector_spec(a1_vec, params, param_set)

    # Measurement parameters
    if measurement isa MeasurementExpr
        for p in measurement.params
            if !(p.name in param_set)
                push!(params, p)
                push!(param_set, p.name)
            end
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

    return EKFSpec{typeof(measurement), typeof(jacobian_mode)}(
        name, m, p, r,
        params, measurement, jacobian_mode,
        H_spec, T_spec, R_spec, Q_spec,
        a1_elems, P1_spec, matrix_exprs
    )
end
