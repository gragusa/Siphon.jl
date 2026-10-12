"""
    KFParms{Zt, Ht, Tt, Rt, Qt}

State-space model parameters for the Kalman filter.

# State Space Model
```
y_t = Z * α_t + ε_t,    ε_t ~ N(0, H)
α_{t+1} = T * α_t + R * η_t,    η_t ~ N(0, Q)
```

# Fields
- `Z`: Observation matrix (p × m)
- `H`: Observation noise covariance (p × p)
- `T`: State transition matrix (m × m)
- `R`: State noise selection matrix (m × r)
- `Q`: State noise covariance (r × r)
"""
struct KFParms{Zt, Ht, Tt, Rt, Qt}
    Z::Zt
    H::Ht
    T::Tt
    R::Rt
    Q::Qt
    function KFParms{Zt, Ht, Tt, Rt, Qt}(Z, H, T, R, Q) where {Zt, Ht, Tt, Rt, Qt}
        _check_system_dims("KFParms", Z, H, T, R, Q)
        return new{Zt, Ht, Tt, Rt, Qt}(Z, H, T, R, Q)
    end
end

function KFParms(Z, H, T, R, Q)
    KFParms{typeof(Z), typeof(H), typeof(T), typeof(R), typeof(Q)}(
        Z, H, T, R, Q)
end

# Throw a DimensionMismatch naming `caller` unless `size(A) == dims`.
function _check_size(caller, name, A, dims::Tuple)
    size(A) == dims && return nothing
    throw(DimensionMismatch("$caller: $name must be $(join(dims, " × ")), " *
                            "got $(join(size(A), " × "))"))
end

# Z is p × m, H p × p, T m × m, R m × r and Q r × r. Scalar systems have nothing to check.
function _check_system_dims(caller, Z::AbstractMatrix, H::AbstractMatrix,
        T::AbstractMatrix, R::AbstractMatrix, Q::AbstractMatrix)
    p, m = size(Z)
    r = size(Q, 1)
    _check_size(caller, "H", H, (p, p))
    _check_size(caller, "T", T, (m, m))
    _check_size(caller, "R", R, (m, r))
    _check_size(caller, "Q", Q, (r, r))
    return nothing
end
_check_system_dims(caller, Z, H, T, R, Q) = nothing

# Size utilities
numstates(T::AbstractMatrix) = size(T, 1)
numstates(T::Real) = 1

nummeasur(Z::AbstractMatrix) = size(Z, 1)
nummeasur(Z::Real) = 1

# ============================================
# StaticArrays specialization
# ============================================

"""
    STATIC_THRESHOLD

Largest model dimension for which `static = :auto` runs the StaticArrays
specialization of the Kalman filter. A model qualifies when its numbers of
states, observed series and shocks are all at most `STATIC_THRESHOLD`.
"""
const STATIC_THRESHOLD = 13

"""
    _use_static(static::Symbol, n_states, n_obs, n_shocks) -> Bool

Resolve the `static = :auto | :on | :off` keyword accepted by the spec-level
entry points. `:auto` selects StaticArrays when every dimension is at most
[`STATIC_THRESHOLD`](@ref); `:on` forces them, warning above the threshold.
"""
function _use_static(static::Symbol, n_states::Integer, n_obs::Integer, n_shocks::Integer)
    max_dim = max(n_states, n_obs, n_shocks)
    if static === :auto
        return max_dim <= STATIC_THRESHOLD
    elseif static === :on
        if max_dim > STATIC_THRESHOLD
            @warn "static = :on requested but the model has dimension $(max_dim) > " *
                  "STATIC_THRESHOLD = $(STATIC_THRESHOLD). StaticArrays above this " *
                  "size typically lose to BLAS and compile slowly; consider " *
                  "static = :auto or :off." maxlog = 1
        end
        return true
    elseif static === :off
        return false
    else
        throw(ArgumentError("static must be :auto, :on or :off, got :$(static)"))
    end
end

"""
    _static_system(p::KFParms, a1, P1) -> (p, a1, P1)

Convert every system matrix and the initial state to `SMatrix`/`SVector`, so
the filter runs its StaticArrays specialization. The conversion covers all
seven arrays or none: a partly static system would take the generic path.
"""
function _static_system(p::KFParms, a1::AbstractVector, P1::AbstractMatrix)
    sm(A::AbstractMatrix) = SMatrix{size(A, 1), size(A, 2)}(A)
    p_s = KFParms(sm(p.Z), sm(p.H), sm(p.T), sm(p.R), sm(p.Q))
    return p_s, SVector{length(a1)}(a1), sm(P1)
end

# ============================================
# KalmanFilterResult - Filter output struct
# ============================================

"""
    KalmanFilterResult{T<:Real, P<:KFParms}

Result of Kalman filter computation, storing model parameters and both predicted and filtered quantities.

Following FKF (R package) naming conventions:
- `at`: Predicted states E[αₜ | y₁:ₜ₋₁] (m × n)
- `Pt`: Predicted covariances Var[αₜ | y₁:ₜ₋₁] (m × m × n)
- `att`: Filtered states E[αₜ | y₁:ₜ] (m × n)
- `Ptt`: Filtered covariances Var[αₜ | y₁:ₜ] (m × m × n)
- `vt`: Prediction errors yₜ - E[yₜ | y₁:ₜ₋₁] (p × n)
- `Ft`: Innovation covariances Var[yₜ | y₁:ₜ₋₁] (p × p × n)
- `Kt`: Kalman gains (m × p × n)

# Fields
- `p`: Model parameters (KFParms with Z, H, T, R, Q)
- `loglik`: Log-likelihood of non-missing observations
- `at`: Predicted state means (m × n)
- `Pt`: Predicted state covariances (m × m × n)
- `att`: Filtered state means (m × n)
- `Ptt`: Filtered state covariances (m × m × n)
- `vt`: Innovations/prediction errors (p × n), NaN for missing
- `Ft`: Innovation covariances (p × p × n)
- `Kt`: Kalman gains (m × p × n), zero for missing
- `missing_mask`: BitVector, true where no row of the period was observed
- `observed_mask`: BitMatrix (p × n), true where an individual value is present

Where only some rows of a period are observed, `vt`, `Ft` and `Kt` hold the
reduced-system quantities in their leading rows and columns for that period; the
observed rows are `findall(view(observed_mask, :, t))`. The accessors
`prediction_errors`, `prediction_errors_cov` and `kalman_gains` return them in series
positions instead.

A model with one state and one series may be given as a `KFParms` of numbers. Its result
has the same layout with `m = p = 1`, and `p` keeps the numbers.

# Accessor Methods
The accessors of [`AbstractFilterResult`](@ref):
- `system_matrices(r)` → `p` (KFParms)
- `obs_matrix(r)`, `obs_cov(r)`, `transition_matrix(r)`, `selection_matrix(r)`,
  `state_cov(r)` → `p.Z`, `p.H`, `p.T`, `p.R`, `p.Q`
- `predicted_states(r)` → `at`
- `predicted_states_cov(r)` → `Pt`
- `filtered_states(r)` → `att`
- `filtered_states_cov(r)` → `Ptt`
- `prediction_errors(r)`, `prediction_errors_cov(r)`, `kalman_gains(r)` → `vt`, `Ft`,
  `Kt` spread back to series positions (copies)
- `loglikelihood(r)` → `loglik`
- `initial_state(r)`, `initial_state_cov(r)` → `at[:, 1]`, `Pt[:, :, 1]`
"""
struct KalmanFilterResult{T <: Real, P <: KFParms} <: AbstractFilterResult
    # Model parameters
    p::P
    # Log-likelihood
    loglik::T
    # Predicted quantities: a_{t|t-1}, P_{t|t-1}
    at::Matrix{T}          # m × n
    Pt::Array{T, 3}         # m × m × n
    # Filtered quantities: a_{t|t}, P_{t|t}
    att::Matrix{T}         # m × n
    Ptt::Array{T, 3}        # m × m × n
    # Innovations
    vt::Matrix{T}          # p × n
    Ft::Array{T, 3}         # p × p × n
    Kt::Array{T, 3}         # m × p × n
    # Missing data
    missing_mask::BitVector    # n: true where no row of the period was observed
    observed_mask::BitMatrix   # p × n: true where an individual value is present
    # 0, or the period in which the filter stopped on a non-positive-definite F
    failed_at::Int
end

function KalmanFilterResult(
        p::KFParms, loglik::Real, at::AbstractMatrix, Pt::AbstractArray{<:Any, 3},
        att::AbstractMatrix, Ptt::AbstractArray{<:Any, 3}, vt::AbstractMatrix,
        Ft::AbstractArray{<:Any, 3}, Kt::AbstractArray{<:Any, 3},
        missing_mask::BitVector, observed_mask::BitMatrix
)
    return KalmanFilterResult(
        p, loglik, at, Pt, att, Ptt, vt, Ft, Kt, missing_mask, observed_mask, 0)
end

# A result built without a per-element mask is fully observed wherever its period
# mask says the period was observed.
function KalmanFilterResult(
        p::KFParms, loglik::Real, at::AbstractMatrix, Pt::AbstractArray{<:Any, 3},
        att::AbstractMatrix, Ptt::AbstractArray{<:Any, 3}, vt::AbstractMatrix,
        Ft::AbstractArray{<:Any, 3}, Kt::AbstractArray{<:Any, 3},
        missing_mask::BitVector; failed_at::Int = 0
)
    observed_mask = BitMatrix(undef, size(vt, 1), size(vt, 2))
    for t in axes(observed_mask, 2)
        observed_mask[:, t] .= !missing_mask[t]
    end
    return KalmanFilterResult(
        p, loglik, at, Pt, att, Ptt, vt, Ft, Kt, missing_mask, observed_mask, failed_at)
end

# Accessor methods for KalmanFilterResult (the shared ones are in accessors.jl)
"""
    observed_mask(r::KalmanFilterResult) -> BitMatrix

Return the `p × n` mask that is true where an individual observation was present.
"""
observed_mask(r::KalmanFilterResult) = _require_output(r, :observed_mask).observed_mask

"""
    observed_rows(r::KalmanFilterResult, t::Integer) -> Vector{Int}

Return the indices of the rows observed in period `t`, in increasing order.
"""
observed_rows(r::KalmanFilterResult, t::Integer) = findall(view(observed_mask(r), :, t))

# ============================================
# SmootherWorkspace - Pre-allocated smoother storage
# ============================================

"""
    SmootherWorkspace{T<:Real}

Pre-allocated workspace for Kalman smoother operations.

Allows in-place computation to avoid allocations when smoothing repeatedly.

# Fields
- `alpha`: Smoothed state means E[αₜ | y₁:ₙ] (m × n)
- `V`: Smoothed state covariances Var[αₜ | y₁:ₙ] (m × m × n)
- `r`: Smoothing recursion auxiliary vector (m × n)
- `N`: Smoothing recursion auxiliary matrix (m × m × n)

# Constructor
    SmootherWorkspace(m::Int, n::Int, ::Type{T}=Float64) where T
"""
mutable struct SmootherWorkspace{T <: Real}
    alpha::Matrix{T}       # m × n: smoothed states
    V::Array{T, 3}          # m × m × n: smoothed covariances
    r::Matrix{T}           # m × n: smoothing recursion auxiliary
    N::Array{T, 3}          # m × m × n: smoothing recursion auxiliary
end

function SmootherWorkspace(m::Int, n::Int, ::Type{T} = Float64) where {T}
    SmootherWorkspace{T}(
        Matrix{T}(undef, m, n),
        Array{T}(undef, m, m, n),
        Matrix{T}(undef, m, n),
        Array{T}(undef, m, m, n)
    )
end

"""
    smoothed_states(w::SmootherWorkspace) -> Matrix

Return smoothed state means E[αₜ | y₁:ₙ] from workspace.
"""
smoothed_states(w::SmootherWorkspace) = w.alpha

"""
    smoothed_states_cov(w::SmootherWorkspace) -> Array{T,3}

Return smoothed state covariances Var[αₜ | y₁:ₙ] from workspace.
"""
smoothed_states_cov(w::SmootherWorkspace) = w.V

# ============================================
# DiffuseFilterResult - Exact diffuse initialization
# ============================================

"""
    DiffuseFilterResult{T<:Real, P<:KFParms}

Result of Kalman filter with exact diffuse initialization (Durbin-Koopman method).

This struct stores filter results when using exact diffuse initialization, which
splits the initial covariance into finite (`P1_star`) and infinite (`P1_inf`) parts.
The diffuse period ends when `norm(Pinf) < tol`.

# Fields
- `p`: Model parameters (KFParms with Z, H, T, R, Q)
- `loglik`: Log-likelihood (only non-diffuse observations contribute)
- `d`: Number of observations in the diffuse period
- `at`: Predicted state means (m × n)
- `Pt`: Predicted state covariances (m × m × n)
- `att`: Filtered state means (m × n)
- `Ptt`: Filtered state covariances (m × m × n)
- `Pinf_store`: Diffuse covariances during diffuse period (m × m × d)
- `Pstar_store`: Finite covariances during diffuse period (m × m × d)
- `vt`: Innovations/prediction errors (p × n), NaN for missing
- `Ft`: Innovation covariances (p × p × n)
- `Kt`: Kalman gains (m × p × n)
- `diffuse_flag`: Vector of length d indicating step type (see Notes)
- `missing_mask`: BitVector, true where no row of the period was observed
- `observed_mask`: BitMatrix (p × n), true where an individual value is present

Where only some rows of a period are observed, `vt`, `Ft` and `Kt` hold the
reduced-system quantities in their leading rows and columns for that period; the
observed rows are `findall(view(observed_mask, :, t))`. The accessors
`prediction_errors`, `prediction_errors_cov` and `kalman_gains` return them in series
positions instead.

# Notes
During the diffuse period (t ≤ d):
- If `diffuse_flag[t] == 1`: Finf was invertible, used exact diffuse update
- If `diffuse_flag[t] == 2`: Finf was singular but nonzero; the observed rows were
  processed one at a time (Koopman & Durbin 2000), each taking the exact diffuse
  update or the Fstar-based one
- If `diffuse_flag[t] == 0`: Finf was zero, used Fstar-based update
- If `diffuse_flag[t] == -1`: no row was observed

Rows that take the exact diffuse update add nothing to the log-likelihood; the
others add their Gaussian term.

`Pt` and `Ptt` hold the finite parts of the predicted and filtered covariances
(the diffuse parts of the predicted covariance are in `Pinf_store`). In an exact
diffuse step `Ft` holds `Finf = Z Pinf Z'` and `Kt` holds the gain
`K⁽⁰⁾ = T Pinf Z' Finf⁻¹` that updates the state, following Durbin & Koopman (2012, §5.2).
In a row-by-row step `Ft` also holds `Finf`, and `Kt` holds `T G`, where `G` is the
gain with `a_{t|t} = a_t + G v_t`.

After the diffuse period (t > d), standard Kalman filter recursion is used.

# See Also
- `kalman_filter(p, y, a1, P1_star, P1_inf)`: Compute diffuse filter
- `kalman_loglik(p, y, a1, P1_star, P1_inf)`: Compute log-likelihood with diffuse initialization
- `KalmanFilterResult`: Standard filter result without diffuse initialization
"""
struct DiffuseFilterResult{T <: Real, P <: KFParms} <: AbstractFilterResult
    # Model parameters
    p::P
    # Log-likelihood (non-diffuse observations only)
    loglik::T
    # Diffuse period length
    d::Int
    # Predicted quantities: a_{t|t-1}, P_{t|t-1}
    at::Matrix{T}           # m × n
    Pt::Array{T, 3}          # m × m × n
    # Filtered quantities: a_{t|t}, P_{t|t}
    att::Matrix{T}          # m × n
    Ptt::Array{T, 3}         # m × m × n
    # Diffuse period covariances (only first d time points)
    Pinf_store::Array{T, 3}  # m × m × d
    Pstar_store::Array{T, 3} # m × m × d
    # Innovations
    vt::Matrix{T}           # p × n
    Ft::Array{T, 3}          # p × p × n
    Kt::Array{T, 3}          # m × p × n
    # Diffuse step flags: 1=Finf invertible, 2=singular nonzero, 0=zero, -1=missing
    diffuse_flag::Vector{Int}  # length d
    # Missing data
    missing_mask::BitVector    # n: true where no row of the period was observed
    observed_mask::BitMatrix   # p × n: true where an individual value is present
    # 0, or the period in which the filter stopped on a non-positive-definite F
    failed_at::Int
end

function DiffuseFilterResult(p::KFParms, loglik::Real, d::Int, at, Pt, att, Ptt,
        Pinf_store, Pstar_store, vt, Ft, Kt, diffuse_flag::Vector{Int},
        missing_mask::BitVector, observed_mask::BitMatrix)
    return DiffuseFilterResult(p, loglik, d, at, Pt, att, Ptt, Pinf_store, Pstar_store,
        vt, Ft, Kt, diffuse_flag, missing_mask, observed_mask, 0)
end

# Accessor methods for DiffuseFilterResult (the shared ones are in accessors.jl)
observed_mask(r::DiffuseFilterResult) = _require_output(r, :observed_mask).observed_mask
observed_rows(r::DiffuseFilterResult, t::Integer) = findall(view(observed_mask(r), :, t))

"""
    diffuse_period(r::DiffuseFilterResult) -> Int

Return the number of observations in the diffuse period.
"""
diffuse_period(r::DiffuseFilterResult) = r.d

"""
    diffuse_covariances(r::DiffuseFilterResult) -> (Pinf_store, Pstar_store)

Return the diffuse (Pinf) and finite (Pstar) covariances during the diffuse period.
Both arrays have dimension m × m × d where d is the diffuse period length.
"""
function diffuse_covariances(r::DiffuseFilterResult)
    _require_output(r, :diffuse_covariances)
    return (r.Pinf_store, r.Pstar_store)
end

"""
    diffuse_flags(r::DiffuseFilterResult) -> Vector{Int}

Return flags indicating which type of diffuse step was used at each time point.
- 1: Finf was invertible (exact diffuse update)
- 2: Finf was singular but nonzero (rows processed one at a time)
- 0: Finf was zero (Fstar-based update)
- -1: no row was observed
"""
diffuse_flags(r::DiffuseFilterResult) = _require_output(r, :diffuse_flags).diffuse_flag
