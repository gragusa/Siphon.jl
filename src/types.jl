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
end

# Size utilities
numstates(T::AbstractMatrix) = size(T, 1)
numstates(T::Real) = 1

nummeasur(Z::AbstractMatrix) = size(Z, 1)
nummeasur(Z::Real) = 1

Base.size(p::KFParms{P}) where P<:Real = (1, 1, 1)
Base.size(p::KFParms{P}) where P<:AbstractArray = (size(p.Z, 1), size(p.T, 1), size(p.Q, 1))

# ============================================
# StaticArrays automatic conversion
# ============================================

"""
Maximum dimension for automatic conversion to StaticArrays.
Matrices with max(rows, cols) ≤ STATIC_THRESHOLD are converted.
"""
const STATIC_THRESHOLD = 13

"""
    to_static_if_small(x)

Convert array to StaticArray if dimensions are small enough (≤ STATIC_THRESHOLD).
Returns the input unchanged if dimensions exceed threshold or if already a StaticArray.

This enables automatic performance optimization for small state-space models
by using stack-allocated arrays that the compiler can unroll.
"""
to_static_if_small(x::Real) = x

function to_static_if_small(x::AbstractVector)
    n = length(x)
    n ≤ STATIC_THRESHOLD ? SVector{n}(x) : x
end

function to_static_if_small(x::AbstractMatrix)
    m, n = size(x)
    (m ≤ STATIC_THRESHOLD && n ≤ STATIC_THRESHOLD) ? SMatrix{m,n}(x) : x
end

# Already static - pass through unchanged
to_static_if_small(x::StaticArray) = x

"""
    KFParms_static(Z, H, T, R, Q)

Create KFParms with automatic StaticArrays conversion for small matrices.

Equivalent to `KFParms(to_static_if_small(Z), ...)` for all parameters.
Use this for potential performance gains when state/observation dimensions are small (≤ 13).

# Example
```julia
# These become SMatrix automatically
p = KFParms_static([1.0;;], [1.0;;], [1.0;;], [1.0;;], [1.0;;])
typeof(p.Z)  # SMatrix{1, 1, Float64, 1}

# Large matrices stay as Matrix
p = KFParms_static(rand(20,20), rand(20,20), rand(20,20), rand(20,20), rand(20,20))
typeof(p.Z)  # Matrix{Float64}
```
"""
function KFParms_static(Z, H, T, R, Q)
    KFParms(
        to_static_if_small(Z),
        to_static_if_small(H),
        to_static_if_small(T),
        to_static_if_small(R),
        to_static_if_small(Q)
    )
end

# ============================================
# Backend selection for filter/smoother
# ============================================

"""
    select_backend(n_states::Int, n_obs::Int, backend::Symbol) -> Symbol

Select filter/smoother backend based on problem dimensions.

# Arguments
- `n_states`: Number of state variables
- `n_obs`: Number of observable variables
- `backend`: Requested backend - one of:
  - `:auto` - Automatically select based on dimensions (default)
  - `:static` - Force pure/functional implementation with StaticArrays
  - `:inplace` - Force in-place implementation with KalmanWorkspace

# Returns
- `:static` if `max(n_states, n_obs) ≤ STATIC_THRESHOLD` and backend is `:auto` or `:static`
- `:inplace` if dimensions exceed threshold or backend is `:inplace`

# Details
The pure/functional backend (`:static`) uses StaticArrays for small problems,
enabling stack allocation and compiler optimizations. It is AD-compatible.

The in-place backend (`:inplace`) uses KalmanWorkspace for zero-allocation
filter/smoother operations, optimal for large problems and repeated EM iterations.
It is NOT AD-compatible.

# Example
```julia
select_backend(3, 2, :auto)      # → :static (small problem)
select_backend(20, 50, :auto)    # → :inplace (large problem)
select_backend(3, 2, :inplace)   # → :inplace (forced)
```
"""
function select_backend(n_states::Int, n_obs::Int, backend::Symbol)
    if backend == :auto
        return max(n_states, n_obs) <= STATIC_THRESHOLD ? :static : :inplace
    elseif backend == :static
        return :static
    elseif backend == :inplace
        return :inplace
    else
        throw(ArgumentError("Invalid backend: $backend. Use :auto, :static, or :inplace"))
    end
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
- `missing_mask`: BitVector indicating missing observations

# Accessor Methods
Use accessor methods for clear, consistent API:
- `parameters(r)` → `p` (KFParms)
- `obs_matrix(r)` → `p.Z`
- `obs_cov(r)` → `p.H`
- `transition_matrix(r)` → `p.T`
- `selection_matrix(r)` → `p.R`
- `state_cov(r)` → `p.Q`
- `predicted_states(r)` → `at`
- `variances_predicted_states(r)` → `Pt`
- `filtered_states(r)` → `att`
- `variances_filtered_states(r)` → `Ptt`
- `prediction_errors(r)` → `vt`
- `variances_prediction_errors(r)` → `Ft`
- `kalman_gains(r)` → `Kt`
- `loglikelihood(r)` → `loglik`
"""
struct KalmanFilterResult{T<:Real, P<:KFParms}
    # Model parameters
    p::P
    # Log-likelihood
    loglik::T
    # Predicted quantities: a_{t|t-1}, P_{t|t-1}
    at::Matrix{T}          # m × n
    Pt::Array{T,3}         # m × m × n
    # Filtered quantities: a_{t|t}, P_{t|t}
    att::Matrix{T}         # m × n
    Ptt::Array{T,3}        # m × m × n
    # Innovations
    vt::Matrix{T}          # p × n
    Ft::Array{T,3}         # p × p × n
    Kt::Array{T,3}         # m × p × n
    # Missing data
    missing_mask::BitVector
end

# Accessor methods for KalmanFilterResult
"""
    parameters(r::KalmanFilterResult) -> KFParms

Return the model parameters (Z, H, T, R, Q).
"""
parameters(r::KalmanFilterResult) = r.p

"""
    obs_matrix(r::KalmanFilterResult)

Return observation matrix Z (p × m).
"""
obs_matrix(r::KalmanFilterResult) = r.p.Z

"""
    obs_cov(r::KalmanFilterResult)

Return observation noise covariance H (p × p).
"""
obs_cov(r::KalmanFilterResult) = r.p.H

"""
    transition_matrix(r::KalmanFilterResult)

Return state transition matrix T (m × m).
"""
transition_matrix(r::KalmanFilterResult) = r.p.T

"""
    selection_matrix(r::KalmanFilterResult)

Return state noise selection matrix R (m × r).
"""
selection_matrix(r::KalmanFilterResult) = r.p.R

"""
    state_cov(r::KalmanFilterResult)

Return state noise covariance Q (r × r).
"""
state_cov(r::KalmanFilterResult) = r.p.Q

"""
    predicted_states(r::KalmanFilterResult) -> Matrix

Return predicted state means E[αₜ | y₁:ₜ₋₁] for t = 1:n. (FKF: at)
"""
predicted_states(r::KalmanFilterResult) = r.at

"""
    variances_predicted_states(r::KalmanFilterResult) -> Array{T,3}

Return predicted state covariances Var[αₜ | y₁:ₜ₋₁] for t = 1:n. (FKF: Pt)
"""
variances_predicted_states(r::KalmanFilterResult) = r.Pt

"""
    filtered_states(r::KalmanFilterResult) -> Matrix

Return filtered state means E[αₜ | y₁:ₜ] for t = 1:n. (FKF: att)
"""
filtered_states(r::KalmanFilterResult) = r.att

"""
    variances_filtered_states(r::KalmanFilterResult) -> Array{T,3}

Return filtered state covariances Var[αₜ | y₁:ₜ] for t = 1:n. (FKF: Ptt)
"""
variances_filtered_states(r::KalmanFilterResult) = r.Ptt

"""
    prediction_errors(r::KalmanFilterResult) -> Matrix

Return prediction errors (innovations) vₜ = yₜ - E[yₜ | y₁:ₜ₋₁] for t = 1:n. (FKF: vt)
NaN for missing observations.
"""
prediction_errors(r::KalmanFilterResult) = r.vt

"""
    variances_prediction_errors(r::KalmanFilterResult) -> Array{T,3}

Return innovation covariances Var[yₜ | y₁:ₜ₋₁] for t = 1:n. (FKF: Ft)
"""
variances_prediction_errors(r::KalmanFilterResult) = r.Ft

"""
    kalman_gains(r::KalmanFilterResult) -> Array{T,3}

Return Kalman gains Kₜ for t = 1:n. Zero for missing observations.
"""
kalman_gains(r::KalmanFilterResult) = r.Kt

"""
    loglikelihood(r::KalmanFilterResult) -> Real

Return log-likelihood of non-missing observations.
"""
loglikelihood(r::KalmanFilterResult) = r.loglik

# ============================================
# KalmanFilterResultScalar - Scalar version
# ============================================

"""
    KalmanFilterResultScalar{T<:Real}

Scalar (univariate state) version of KalmanFilterResult.

Same naming conventions as KalmanFilterResult but with vectors instead of matrices
for state quantities.
"""
struct KalmanFilterResultScalar{T<:Real}
    loglik::T
    at::Vector{T}          # n: predicted states
    Pt::Vector{T}          # n: predicted variances
    att::Vector{T}         # n: filtered states
    Ptt::Vector{T}         # n: filtered variances
    vt::Vector{T}          # n: innovations
    Ft::Vector{T}          # n: innovation variances
    missing_mask::BitVector
end

# Accessor methods for scalar version
predicted_states(r::KalmanFilterResultScalar) = r.at
variances_predicted_states(r::KalmanFilterResultScalar) = r.Pt
filtered_states(r::KalmanFilterResultScalar) = r.att
variances_filtered_states(r::KalmanFilterResultScalar) = r.Ptt
prediction_errors(r::KalmanFilterResultScalar) = r.vt
variances_prediction_errors(r::KalmanFilterResultScalar) = r.Ft
loglikelihood(r::KalmanFilterResultScalar) = r.loglik

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
mutable struct SmootherWorkspace{T<:Real}
    alpha::Matrix{T}       # m × n: smoothed states
    V::Array{T,3}          # m × m × n: smoothed covariances
    r::Matrix{T}           # m × n: smoothing recursion auxiliary
    N::Array{T,3}          # m × m × n: smoothing recursion auxiliary
end

function SmootherWorkspace(m::Int, n::Int, ::Type{T}=Float64) where T
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
    variances_smoothed_states(w::SmootherWorkspace) -> Array{T,3}

Return smoothed state covariances Var[αₜ | y₁:ₙ] from workspace.
"""
variances_smoothed_states(w::SmootherWorkspace) = w.V

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
- `diffuse_flag`: Vector of length d indicating step type (1=Finf invertible, 0=singular)
- `missing_mask`: BitVector indicating missing observations

# Notes
During the diffuse period (t ≤ d):
- If `diffuse_flag[t] == 1`: Finf was invertible, used exact diffuse update
- If `diffuse_flag[t] == 0`: Finf was singular, used Fstar-based update

After the diffuse period (t > d), standard Kalman filter recursion is used.

# See Also
- `kalman_filter_diffuse`: Compute diffuse filter
- `kalman_loglik_diffuse`: Compute log-likelihood with diffuse initialization
- `KalmanFilterResult`: Standard filter result without diffuse initialization
"""
struct DiffuseFilterResult{T<:Real, P<:KFParms}
    # Model parameters
    p::P
    # Log-likelihood (non-diffuse observations only)
    loglik::T
    # Diffuse period length
    d::Int
    # Predicted quantities: a_{t|t-1}, P_{t|t-1}
    at::Matrix{T}           # m × n
    Pt::Array{T,3}          # m × m × n
    # Filtered quantities: a_{t|t}, P_{t|t}
    att::Matrix{T}          # m × n
    Ptt::Array{T,3}         # m × m × n
    # Diffuse period covariances (only first d time points)
    Pinf_store::Array{T,3}  # m × m × d
    Pstar_store::Array{T,3} # m × m × d
    # Innovations
    vt::Matrix{T}           # p × n
    Ft::Array{T,3}          # p × p × n
    Kt::Array{T,3}          # m × p × n
    # Diffuse step flags: 1=Finf invertible, 0=singular
    diffuse_flag::Vector{Int}  # length d
    # Missing data
    missing_mask::BitVector
end

# Accessor methods for DiffuseFilterResult
parameters(r::DiffuseFilterResult) = r.p
obs_matrix(r::DiffuseFilterResult) = r.p.Z
obs_cov(r::DiffuseFilterResult) = r.p.H
transition_matrix(r::DiffuseFilterResult) = r.p.T
selection_matrix(r::DiffuseFilterResult) = r.p.R
state_cov(r::DiffuseFilterResult) = r.p.Q
predicted_states(r::DiffuseFilterResult) = r.at
variances_predicted_states(r::DiffuseFilterResult) = r.Pt
filtered_states(r::DiffuseFilterResult) = r.att
variances_filtered_states(r::DiffuseFilterResult) = r.Ptt
prediction_errors(r::DiffuseFilterResult) = r.vt
variances_prediction_errors(r::DiffuseFilterResult) = r.Ft
kalman_gains(r::DiffuseFilterResult) = r.Kt
loglikelihood(r::DiffuseFilterResult) = r.loglik

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
diffuse_covariances(r::DiffuseFilterResult) = (r.Pinf_store, r.Pstar_store)

"""
    diffuse_flags(r::DiffuseFilterResult) -> Vector{Int}

Return flags indicating which type of diffuse step was used at each time point.
- 1: Finf was invertible (exact diffuse update)
- 0: Finf was singular (Fstar-based update)
"""
diffuse_flags(r::DiffuseFilterResult) = r.diffuse_flag
