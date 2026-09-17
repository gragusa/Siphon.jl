"""
    ekf_types.jl

Core types for the Extended Kalman Filter extension.

The EKF extension supports nonlinear measurement equations of the form

    y_t = h(α_t, t) + ε_t,    ε_t ~ N(0, H)
    α_{t+1} = T α_t + R η_t,  η_t ~ N(0, Q)

The measurement map `h` is provided by an `AbstractEKFMeasurement` subtype that
defines either a pure functional method (`measurement(model, a, t)` ->
AbstractVector) for the AD path or a mutating method
(`measurement!(out, model, a, t)`) plus an analytic Jacobian
(`measurement_jacobian!(Z, model, a, t)`) for the workspace-safe path.

State equation linearity is preserved deliberately: it keeps the EM machinery
usable later and matches OPSC (2024) shadow-rate DNS variants.
"""

# ============================================================================
# Jacobian-mode markers
# ============================================================================

"""
    AbstractEKFJacobianMode

Tag type controlling how the EKF obtains the measurement Jacobian
`Z_t = ∂h/∂α |_{α = a_t}` at each step.
"""
abstract type AbstractEKFJacobianMode end

"""
    ADJacobian()

Use automatic differentiation (ForwardDiff) on a pure functional measurement
map `measurement(model, a, t)` to obtain the state Jacobian. This is the
default MLE-facing mode: it requires only the immutable functional method.
"""
struct ADJacobian <: AbstractEKFJacobianMode end

"""
    AnalyticJacobian()

Use a user-supplied `measurement_jacobian` / `measurement_jacobian!` method.
This is the production mode for nonsmooth measurements (e.g. B-DNS kink) and
the recommended mode for repeated EM filter/smoother passes.
"""
struct AnalyticJacobian <: AbstractEKFJacobianMode end

"""
    FiniteDiffJacobian()

Use central finite differences on the pure measurement map. Diagnostic only;
should not be selected by default in production code.
"""
struct FiniteDiffJacobian <: AbstractEKFJacobianMode end

# ============================================================================
# AbstractEKFMeasurement and stub methods
# ============================================================================

"""
    AbstractEKFMeasurement

Supertype for measurement objects used by the EKF. Concrete subtypes carry
fixed data (e.g. factor loadings, lower bounds, maturity grids) and define one
or both of:

  - `measurement(model, a, t)::AbstractVector` — pure functional map (used by
    AD and the functional `ekf_loglik`).
  - `measurement!(out, model, a, t)` — mutating map (used by `ekf_filter!`).

The Jacobian `∂h/∂a` may be supplied as

  - `measurement_jacobian(model, a, t)::AbstractMatrix` (analytic, pure), or
  - `measurement_jacobian!(Z, model, a, t)` (analytic, mutating), or
  - obtained automatically via `ADJacobian()` from the pure `measurement` map.

Concrete types are preferred over `Function`-typed fields: they give better
inference and let users store fixed matrices, parameters, and grids cleanly.
"""
abstract type AbstractEKFMeasurement end

"""
    measurement(model::AbstractEKFMeasurement, a, t) -> AbstractVector

Pure functional measurement map. Must be defined for the AD-Jacobian path and
for the functional `ekf_loglik`.
"""
function measurement(model::AbstractEKFMeasurement, a, t)
    throw(MethodError(measurement, (model, a, t)))
end

"""
    measurement!(out, model::AbstractEKFMeasurement, a, t)

Mutating measurement map writing `h(a, t)` into `out`. Required for the
in-place `ekf_filter!` path.
"""
function measurement!(out, model::AbstractEKFMeasurement, a, t)
    throw(MethodError(measurement!, (out, model, a, t)))
end

"""
    measurement_jacobian(model::AbstractEKFMeasurement, a, t) -> AbstractMatrix

Pure analytic Jacobian `∂h/∂a` evaluated at `a`. Optional.
"""
function measurement_jacobian(model::AbstractEKFMeasurement, a, t)
    throw(MethodError(measurement_jacobian, (model, a, t)))
end

"""
    measurement_jacobian!(Z, model::AbstractEKFMeasurement, a, t)

Mutating analytic Jacobian writing `∂h/∂a` into `Z`. Required for the in-place
`ekf_filter!` path under `AnalyticJacobian()`.
"""
function measurement_jacobian!(Z, model::AbstractEKFMeasurement, a, t)
    throw(MethodError(measurement_jacobian!, (Z, model, a, t)))
end

# ============================================================================
# EKFParms
# ============================================================================

"""
    EKFParms{Mt, Jt, Ht, Tt, Rt, Qt}

Parameter container for an EKF state space model:

```
y_t = h(α_t, t) + ε_t,     ε_t ~ N(0, H)
α_{t+1} = T α_t + R η_t,    η_t ~ N(0, Q)
```

# Fields
- `measurement::Mt` — an `AbstractEKFMeasurement` (or any object dispatching on
  the measurement protocol).
- `jacobian_mode::Jt` — `ADJacobian()`, `AnalyticJacobian()`, or
  `FiniteDiffJacobian()`.
- `H::Ht` — observation noise covariance (p × p).
- `T::Tt` — transition matrix (m × m).
- `R::Rt` — selection matrix (m × r).
- `Q::Qt` — state-shock covariance (r × r).
"""
struct EKFParms{Mt, Jt <: AbstractEKFJacobianMode, Ht, Tt, Rt, Qt}
    measurement::Mt
    jacobian_mode::Jt
    H::Ht
    T::Tt
    R::Rt
    Q::Qt
end

# Convenience constructor with default ADJacobian
EKFParms(measurement, H, T, R, Q) = EKFParms(measurement, ADJacobian(), H, T, R, Q)

# ============================================================================
# EKFFilterResult
# ============================================================================

"""
    EKFFilterResult{T<:Real, P<:EKFParms}

Result of an EKF forward pass. Field names mirror `KalmanFilterResult`
(`at`, `Pt`, `att`, `Ptt`, `vt`, `Ft`, `Kt`, `missing_mask`) plus two
EKF-specific fields:

- `yt` — predicted measurement `h(a_t, t)` (p × n)
- `Zt` — measurement Jacobian `∂h/∂α` evaluated at `a_t` (p × m × n)
"""
struct EKFFilterResult{T <: Real, P <: EKFParms}
    p::P
    loglik::T
    at::Matrix{T}           # m × n
    Pt::Array{T, 3}         # m × m × n
    att::Matrix{T}          # m × n
    Ptt::Array{T, 3}        # m × m × n
    yt::Matrix{T}           # p × n  predicted measurement
    Zt::Array{T, 3}         # p × m × n  measurement Jacobians
    vt::Matrix{T}           # p × n
    Ft::Array{T, 3}         # p × p × n
    Kt::Array{T, 3}         # m × p × n
    missing_mask::BitVector
end

# Reuse linear accessors where possible
parameters(r::EKFFilterResult) = r.p
obs_cov(r::EKFFilterResult) = r.p.H
transition_matrix(r::EKFFilterResult) = r.p.T
selection_matrix(r::EKFFilterResult) = r.p.R
state_cov(r::EKFFilterResult) = r.p.Q
predicted_states(r::EKFFilterResult) = r.at
variances_predicted_states(r::EKFFilterResult) = r.Pt
filtered_states(r::EKFFilterResult) = r.att
variances_filtered_states(r::EKFFilterResult) = r.Ptt
prediction_errors(r::EKFFilterResult) = r.vt
variances_prediction_errors(r::EKFFilterResult) = r.Ft
kalman_gains(r::EKFFilterResult) = r.Kt
loglikelihood(r::EKFFilterResult) = r.loglik

"""
    predicted_observations(r::EKFFilterResult) -> Matrix

Return predicted measurements `h(a_t, t)` for t = 1:n (p × n).
"""
predicted_observations(r::EKFFilterResult) = r.yt

"""
    measurement_jacobians(r::EKFFilterResult) -> Array{T,3}

Return measurement Jacobians `∂h/∂α |_{α=a_t}` for t = 1:n (p × m × n).
"""
measurement_jacobians(r::EKFFilterResult) = r.Zt

# ============================================================================
# Element-type helper for AD compatibility
# ============================================================================

@inline _ekf_filter_eltype(p::EKFParms,
    a1,
    P1) = promote_type(
    eltype(p.H), eltype(p.T), eltype(p.R), eltype(p.Q),
    eltype(a1), eltype(P1)
)
