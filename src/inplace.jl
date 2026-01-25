"""
    inplace.jl

In-place Kalman filter, smoother, and EM algorithm for large state-space models.

This module provides zero-allocation implementations designed for:
- Dynamic factor models with many observables (p >> m)
- EM algorithm requiring repeated filter/smoother iterations
- Large-scale applications where allocation overhead matters

# Contents

## KalmanWorkspace (lines ~25-340)
Pre-allocated workspace for in-place Kalman filter and smoother operations.
- `KalmanWorkspace{T}` struct with all storage
- `kalman_filter!` - in-place filter with missing data handling
- `kalman_smoother!` - in-place RTS smoother with cross-lag covariances
- `filter_and_smooth!` - combined operation

## EMWorkspace (lines ~890-1400)
EM algorithm infrastructure:
- `EMWorkspace{T}` - sufficient statistics storage
- `compute_sufficient_stats!` - E-step computation
- `update_Z!`, `update_H!`, `update_T!`, `update_Q!` - M-step updates
- `em_estimate!` - main EM algorithm

## StateSpaceModel (lines ~1400-2600)
High-level model interface:
- `StateSpaceModel{T}` - mutable model container
- `fit!(MLE(), model, y)` - MLE via numerical optimization
- `fit!(EM(), model, y)` - EM algorithm (auto-selects backend)
- Accessors: `parameters`, `filtered_states`, `smoothed_states`, etc.
- `forecast(model, h)` - h-step ahead forecasting

## DynamicFactorModel (lines ~2600-end)
Specialized dynamic factor model:
- `DynamicFactorModel{T}` - DFM with optional AR errors
- `DynamicFactorModelSpec` - model specification
- EM estimation with AR error dynamics
- Factor extraction and forecasting
"""

# Note: LinearAlgebra is already imported by the main module

# DSL types (SSMSpec, SSMParameter, etc.) are available from the DSL submodule
# which is loaded before this file

# ============================================
# KalmanWorkspace - Pre-allocated storage
# ============================================

"""
    KalmanWorkspace{T<:Real}

Pre-allocated workspace for in-place Kalman filter and smoother operations.

Stores all matrices, vectors, and scratch space needed for filtering and smoothing,
enabling zero-allocation iterations after initial construction.

# Fields

## Dimensions
- `obs_dim`: Number of observables (p)
- `state_dim`: Number of states (m)
- `shock_dim`: Number of state shocks (r)
- `n_obs`: Number of time periods (n)

## Parameters (owned copies)
- `Z`: Observation matrix (p × m)
- `H`: Observation noise covariance (p × p)
- `T`: State transition matrix (m × m)
- `R`: State noise selection matrix (m × r)
- `Q`: State noise covariance (r × r)
- `a1`: Initial state mean (m)
- `P1`: Initial state covariance (m × m)

## Precomputed
- `RQR`: R * Q * R' (m × m), updated when R or Q change

## Filter storage
- `at`: Predicted states E[αₜ|y₁:ₜ₋₁] (m × n)
- `Pt`: Predicted covariances (m × m × n)
- `att`: Filtered states E[αₜ|y₁:ₜ] (m × n)
- `Ptt`: Filtered covariances (m × m × n)
- `vt`: Innovations yₜ - Z*aₜ (p × n)
- `Ft`: Innovation covariances (p × p × n)
- `Ft_L`: Cholesky factors (lower triangular) of Ft (p × p × n)
- `Kt`: Kalman gains (m × p × n)
- `missing_mask`: BitVector indicating missing observations

## Smoother storage
- `αs`: Smoothed states E[αₜ|y₁:ₙ] (m × n)
- `Vs`: Smoothed covariances (m × m × n)
- `Pcross`: Cross-lag covariances Cov[αₜ₊₁,αₜ|y₁:ₙ] (m × m × n-1)

## Scratch space (temporaries for BLAS operations)
- Various tmp_* matrices and vectors

## Scalars
- `loglik`: Log-likelihood value
- `n_obs_valid`: Count of non-missing observations
"""
mutable struct KalmanWorkspace{T<:Real}
    # Dimensions
    obs_dim::Int      # p
    state_dim::Int    # m
    shock_dim::Int    # r
    n_times::Int      # n

    # Parameters (owned copies)
    Z::Matrix{T}      # p × m
    H::Matrix{T}      # p × p
    Tmat::Matrix{T}   # m × m (named Tmat to avoid conflict with type T)
    R::Matrix{T}      # m × r
    Q::Matrix{T}      # r × r
    a1::Vector{T}     # m
    P1::Matrix{T}     # m × m

    # Precomputed
    RQR::Matrix{T}    # m × m: R * Q * R'

    # Filter storage
    at::Matrix{T}         # m × n: predicted states
    Pt::Array{T,3}        # m × m × n: predicted covariances
    att::Matrix{T}        # m × n: filtered states
    Ptt::Array{T,3}       # m × m × n: filtered covariances
    vt::Matrix{T}         # p × n: innovations
    Ft::Array{T,3}        # p × p × n: innovation covariances
    Ft_L::Array{T,3}      # p × p × n: Cholesky factors (lower tri)
    Kt::Array{T,3}        # m × p × n: Kalman gains
    missing_mask::BitVector

    # Smoother storage
    αs::Matrix{T}         # m × n: smoothed states
    Vs::Array{T,3}        # m × m × n: smoothed covariances
    Pcross::Array{T,3}    # m × m × (n-1): cross-lag covariances

    # Scratch space
    tmp_mm1::Matrix{T}    # m × m
    tmp_mm2::Matrix{T}    # m × m
    tmp_mm3::Matrix{T}    # m × m
    tmp_pp1::Matrix{T}    # p × p
    tmp_pp2::Matrix{T}    # p × p
    tmp_mp::Matrix{T}     # m × p
    tmp_pm::Matrix{T}     # p × m
    tmp_mr::Matrix{T}     # m × r
    tmp_m1::Vector{T}     # m
    tmp_m2::Vector{T}     # m
    tmp_p1::Vector{T}     # p
    tmp_p2::Vector{T}     # p

    # Smoother scratch
    r_smooth::Vector{T}   # m: smoother recursion vector
    N_smooth::Matrix{T}   # m × m: smoother recursion matrix
    L_smooth::Matrix{T}   # m × m: T - K*Z
    J_smooth::Matrix{T}   # m × m: for cross-cov computation

    # Scalars
    loglik::T
    n_obs_valid::Int
end

"""
    KalmanWorkspace{T}(p::Int, m::Int, r::Int, n::Int) where T

Create workspace with specified dimensions. Parameters must be set via `set_params!`.

# Arguments
- `p`: Number of observables
- `m`: Number of states
- `r`: Number of state shocks
- `n`: Number of time periods
"""
function KalmanWorkspace{T}(p::Int, m::Int, r::Int, n::Int) where {T<:Real}
    KalmanWorkspace{T}(
        # Dimensions
        p,
        m,
        r,
        n,

        # Parameters (uninitialized, must call set_params!)
        Matrix{T}(undef, p, m),      # Z
        Matrix{T}(undef, p, p),      # H
        Matrix{T}(undef, m, m),      # Tmat
        Matrix{T}(undef, m, r),      # R
        Matrix{T}(undef, r, r),      # Q
        Vector{T}(undef, m),         # a1
        Matrix{T}(undef, m, m),      # P1

        # Precomputed
        Matrix{T}(undef, m, m),      # RQR

        # Filter storage
        Matrix{T}(undef, m, n),      # at
        Array{T,3}(undef, m, m, n),  # Pt
        Matrix{T}(undef, m, n),      # att
        Array{T,3}(undef, m, m, n),  # Ptt
        Matrix{T}(undef, p, n),      # vt
        Array{T,3}(undef, p, p, n),  # Ft
        Array{T,3}(undef, p, p, n),  # Ft_L
        Array{T,3}(undef, m, p, n),  # Kt
        BitVector(undef, n),         # missing_mask

        # Smoother storage
        Matrix{T}(undef, m, n),      # αs
        Array{T,3}(undef, m, m, n),  # Vs
        Array{T,3}(undef, m, m, max(n-1, 1)),  # Pcross

        # Scratch space
        Matrix{T}(undef, m, m),      # tmp_mm1
        Matrix{T}(undef, m, m),      # tmp_mm2
        Matrix{T}(undef, m, m),      # tmp_mm3
        Matrix{T}(undef, p, p),      # tmp_pp1
        Matrix{T}(undef, p, p),      # tmp_pp2
        Matrix{T}(undef, m, p),      # tmp_mp
        Matrix{T}(undef, p, m),      # tmp_pm
        Matrix{T}(undef, m, r),      # tmp_mr
        Vector{T}(undef, m),         # tmp_m1
        Vector{T}(undef, m),         # tmp_m2
        Vector{T}(undef, p),         # tmp_p1
        Vector{T}(undef, p),         # tmp_p2

        # Smoother scratch
        Vector{T}(undef, m),         # r_smooth
        Matrix{T}(undef, m, m),      # N_smooth
        Matrix{T}(undef, m, m),      # L_smooth
        Matrix{T}(undef, m, m),      # J_smooth

        # Scalars
        zero(T),                     # loglik
        0,                            # n_obs_valid
    )
end

# Convenience constructor inferring T from inputs
KalmanWorkspace(p::Int, m::Int, r::Int, n::Int) = KalmanWorkspace{Float64}(p, m, r, n)

"""
    KalmanWorkspace(Z, H, T, R, Q, a1, P1, n::Int)

Create workspace from parameter matrices and set parameters.
"""
function KalmanWorkspace(
    Z::AbstractMatrix{T},
    H::AbstractMatrix,
    Tmat::AbstractMatrix,
    R::AbstractMatrix,
    Q::AbstractMatrix,
    a1::AbstractVector,
    P1::AbstractMatrix,
    n::Int,
) where {T}
    p, m = size(Z)
    r = size(Q, 1)

    ws = KalmanWorkspace{T}(p, m, r, n)
    set_params!(ws, Z, H, Tmat, R, Q)
    set_initial!(ws, a1, P1)

    return ws
end

"""
    KalmanWorkspace(kf::KFParms, n::Int)

Create workspace from KFParms.
"""
function KalmanWorkspace(kf::KFParms, a1::AbstractVector, P1::AbstractMatrix, n::Int)
    KalmanWorkspace(kf.Z, kf.H, kf.T, kf.R, kf.Q, a1, P1, n)
end

# ============================================
# Accessor methods for KalmanWorkspace
# ============================================

"""
    predicted_states(ws::KalmanWorkspace) -> Matrix

Return predicted state means E[αₜ | y₁:ₜ₋₁] (m × n).
"""
predicted_states(ws::KalmanWorkspace) = ws.at

"""
    variances_predicted_states(ws::KalmanWorkspace) -> Array{T,3}

Return predicted state covariances Var[αₜ | y₁:ₜ₋₁] (m × m × n).
"""
variances_predicted_states(ws::KalmanWorkspace) = ws.Pt

"""
    filtered_states(ws::KalmanWorkspace) -> Matrix

Return filtered state means E[αₜ | y₁:ₜ] (m × n).
"""
filtered_states(ws::KalmanWorkspace) = ws.att

"""
    variances_filtered_states(ws::KalmanWorkspace) -> Array{T,3}

Return filtered state covariances Var[αₜ | y₁:ₜ] (m × m × n).
"""
variances_filtered_states(ws::KalmanWorkspace) = ws.Ptt

"""
    smoothed_states(ws::KalmanWorkspace) -> Matrix

Return smoothed state means E[αₜ | y₁:ₙ] (m × n).
Must call `kalman_smoother!(ws)` first.
"""
smoothed_states(ws::KalmanWorkspace) = ws.αs

"""
    variances_smoothed_states(ws::KalmanWorkspace) -> Array{T,3}

Return smoothed state covariances Var[αₜ | y₁:ₙ] (m × m × n).
Must call `kalman_smoother!(ws)` first.
"""
variances_smoothed_states(ws::KalmanWorkspace) = ws.Vs

"""
    prediction_errors(ws::KalmanWorkspace) -> Matrix

Return prediction errors (innovations) vₜ = yₜ - Z*aₜ (p × n).
NaN for missing observations.
"""
prediction_errors(ws::KalmanWorkspace) = ws.vt

"""
    variances_prediction_errors(ws::KalmanWorkspace) -> Array{T,3}

Return innovation covariances Var[yₜ | y₁:ₜ₋₁] (p × p × n).
"""
variances_prediction_errors(ws::KalmanWorkspace) = ws.Ft

"""
    kalman_gains(ws::KalmanWorkspace) -> Array{T,3}

Return Kalman gains Kₜ (m × p × n). Zero for missing observations.
"""
kalman_gains(ws::KalmanWorkspace) = ws.Kt

"""
    loglikelihood(ws::KalmanWorkspace) -> Real

Return log-likelihood of non-missing observations.
"""
loglikelihood(ws::KalmanWorkspace) = ws.loglik

"""
    missing_mask(ws::KalmanWorkspace) -> BitVector

Return BitVector indicating which observations are missing.
"""
missing_mask(ws::KalmanWorkspace) = ws.missing_mask

# ============================================
# Parameter setters
# ============================================

"""
    _update_RQR!(ws::KalmanWorkspace)

Recompute R * Q * R' and store in ws.RQR.
Uses scratch space to avoid allocations.
"""
function _update_RQR!(ws::KalmanWorkspace{T}) where {T}
    # RQR = R * Q * R'
    # Step 1: tmp_mr = R * Q
    mul!(ws.tmp_mr, ws.R, ws.Q)
    # Step 2: RQR = tmp_mr * R'
    mul!(ws.RQR, ws.tmp_mr, ws.R')
    return nothing
end

"""
    set_params!(ws, Z, H, T, R, Q)

Set all system matrices. Copies data into workspace.
Automatically recomputes RQR.
"""
function set_params!(
    ws::KalmanWorkspace,
    Z::AbstractMatrix,
    H::AbstractMatrix,
    Tmat::AbstractMatrix,
    R::AbstractMatrix,
    Q::AbstractMatrix,
)
    copyto!(ws.Z, Z)
    copyto!(ws.H, H)
    copyto!(ws.Tmat, Tmat)
    copyto!(ws.R, R)
    copyto!(ws.Q, Q)
    _update_RQR!(ws)
    return ws
end

"""
    set_initial!(ws, a1, P1)

Set initial state mean and covariance.
"""
function set_initial!(ws::KalmanWorkspace, a1::AbstractVector, P1::AbstractMatrix)
    copyto!(ws.a1, a1)
    copyto!(ws.P1, P1)
    return ws
end

"""
    update_params!(ws; Z=nothing, H=nothing, T=nothing, R=nothing, Q=nothing)

Selectively update parameters. Only non-nothing arguments are updated.
Recomputes RQR if R or Q change.
"""
function update_params!(
    ws::KalmanWorkspace;
    Z::Union{Nothing,AbstractMatrix} = nothing,
    H::Union{Nothing,AbstractMatrix} = nothing,
    Tmat::Union{Nothing,AbstractMatrix} = nothing,
    R::Union{Nothing,AbstractMatrix} = nothing,
    Q::Union{Nothing,AbstractMatrix} = nothing,
)
    if Z !== nothing
        copyto!(ws.Z, Z)
    end
    if H !== nothing
        copyto!(ws.H, H)
    end
    if Tmat !== nothing
        copyto!(ws.Tmat, Tmat)
    end

    update_RQR = false
    if R !== nothing
        copyto!(ws.R, R)
        update_RQR = true
    end
    if Q !== nothing
        copyto!(ws.Q, Q)
        update_RQR = true
    end

    if update_RQR
        _update_RQR!(ws)
    end

    return ws
end

# Individual parameter updaters for fine-grained control
update_Z!(ws::KalmanWorkspace, Z::AbstractMatrix) = (copyto!(ws.Z, Z); ws)
update_H!(ws::KalmanWorkspace, H::AbstractMatrix) = (copyto!(ws.H, H); ws)
update_T!(ws::KalmanWorkspace, Tmat::AbstractMatrix) = (copyto!(ws.Tmat, Tmat); ws)

function update_R!(ws::KalmanWorkspace, R::AbstractMatrix)
    copyto!(ws.R, R)
    _update_RQR!(ws)
    return ws
end

function update_Q!(ws::KalmanWorkspace, Q::AbstractMatrix)
    copyto!(ws.Q, Q)
    _update_RQR!(ws)
    return ws
end

# ============================================
# Missing data detection
# ============================================

@inline function _has_missing_vec(y::AbstractVector)
    @inbounds for i in eachindex(y)
        isnan(y[i]) && return true
    end
    return false
end

# ============================================
# In-place Kalman filter
# ============================================

"""
    kalman_filter!(ws::KalmanWorkspace, y::AbstractMatrix) -> loglik

Run Kalman filter in-place, storing results in workspace.

# Arguments
- `ws`: Pre-allocated workspace
- `y`: Observations (p × n matrix), missing values as NaN

# Returns
- `loglik`: Log-likelihood of non-missing observations

# Stored Results
Access via: `predicted_states(ws)`, `filtered_states(ws)`, etc.

# Notes
Uses BLAS Level 3 operations where possible for performance.
Cholesky factors of Ft are stored for potential reuse (e.g., EM algorithm).
"""
function kalman_filter!(ws::KalmanWorkspace{T}, y::AbstractMatrix) where {T}
    p, m, n = ws.obs_dim, ws.state_dim, ws.n_times

    @assert size(y) == (p, n) "Observation matrix size mismatch"

    # Reset scalars
    ws.loglik = zero(T)
    ws.n_obs_valid = 0

    # Initialize state: a = a1, P = P1
    # We use tmp_m1 for current state, tmp_mm1 for current covariance
    a_curr = ws.tmp_m1
    P_curr = ws.tmp_mm1
    copyto!(a_curr, ws.a1)
    copyto!(P_curr, ws.P1)

    # Precompute constant for log-likelihood
    log2pi = log(T(2π))

    @inbounds for t = 1:n
        # Store predicted state: at[:, t] = a_curr
        for i = 1:m
            ws.at[i, t] = a_curr[i]
        end
        # Store predicted covariance: Pt[:, :, t] = P_curr
        for j = 1:m, i = 1:m
            ws.Pt[i, j, t] = P_curr[i, j]
        end

        # Check for missing observation
        y_t = view(y, :, t)
        if _has_missing_vec(y_t)
            ws.missing_mask[t] = true

            # Store NaN for innovation
            for i = 1:p
                ws.vt[i, t] = T(NaN)
            end

            # Compute and store F = Z * P * Z' + H (for completeness)
            # F = Z * P_curr * Z' + H
            mul!(ws.tmp_pm, ws.Z, P_curr)           # tmp_pm = Z * P
            mul!(ws.tmp_pp1, ws.tmp_pm, ws.Z')      # tmp_pp1 = Z * P * Z'
            for j = 1:p, i = 1:p
                ws.Ft[i, j, t] = ws.tmp_pp1[i, j] + ws.H[i, j]
                ws.Ft_L[i, j, t] = zero(T)          # Invalid Cholesky
            end

            # K = 0 for missing
            for j = 1:p, i = 1:m
                ws.Kt[i, j, t] = zero(T)
            end

            # Filtered = predicted for missing
            for i = 1:m
                ws.att[i, t] = a_curr[i]
            end
            for j = 1:m, i = 1:m
                ws.Ptt[i, j, t] = P_curr[i, j]
            end

            # Propagate state: a = T * a
            mul!(ws.tmp_m2, ws.Tmat, a_curr)
            copyto!(a_curr, ws.tmp_m2)

            # Propagate covariance: P = T * P * T' + RQR
            mul!(ws.tmp_mm2, ws.Tmat, P_curr)       # tmp_mm2 = T * P
            mul!(ws.tmp_mm3, ws.tmp_mm2, ws.Tmat')  # tmp_mm3 = T * P * T'
            for j = 1:m, i = 1:m
                P_curr[i, j] = ws.tmp_mm3[i, j] + ws.RQR[i, j]
            end
        else
            ws.missing_mask[t] = false
            ws.n_obs_valid += 1

            # Innovation: v = y - Z * a
            mul!(ws.tmp_p1, ws.Z, a_curr)           # tmp_p1 = Z * a
            for i = 1:p
                ws.vt[i, t] = y_t[i] - ws.tmp_p1[i]
            end
            v_t = view(ws.vt, :, t)

            # Innovation covariance: F = Z * P * Z' + H
            mul!(ws.tmp_pm, ws.Z, P_curr)           # tmp_pm = Z * P
            mul!(ws.tmp_pp1, ws.tmp_pm, ws.Z')      # tmp_pp1 = Z * P * Z'
            for j = 1:p, i = 1:p
                ws.tmp_pp1[i, j] += ws.H[i, j]
                ws.Ft[i, j, t] = ws.tmp_pp1[i, j]
            end

            # Cholesky factorization of F (in-place in tmp_pp1)
            # Make symmetric for numerical stability
            for j = 1:p, i = 1:(j-1)
                avg = (ws.tmp_pp1[i, j] + ws.tmp_pp1[j, i]) / 2
                ws.tmp_pp1[i, j] = avg
                ws.tmp_pp1[j, i] = avg
            end

            chol = cholesky!(Symmetric(ws.tmp_pp1, :L))

            # Store Cholesky factor L
            for j = 1:p, i = 1:p
                ws.Ft_L[i, j, t] = i >= j ? chol.L[i, j] : zero(T)
            end

            # Log-likelihood contribution: -0.5 * (log|F| + v' * F^{-1} * v)
            # log|F| = 2 * sum(log(diag(L)))
            logdetF = zero(T)
            for i = 1:p
                logdetF += 2 * log(chol.L[i, i])
            end

            # Solve L * tmp_p2 = v => tmp_p2 = L \ v
            copyto!(ws.tmp_p2, v_t)
            ldiv!(LowerTriangular(chol.L), ws.tmp_p2)

            # quad_form = ||L \ v||^2 = v' * F^{-1} * v
            quad_form = zero(T)
            for i = 1:p
                quad_form += ws.tmp_p2[i]^2
            end

            ws.loglik += -T(0.5) * (logdetF + quad_form)

            # Kalman gain: K = T * P * Z' * F^{-1}
            # First compute P * Z' (stored in tmp_mp)
            mul!(ws.tmp_mp, P_curr, ws.Z')          # tmp_mp = P * Z' (m × p)

            # Then T * (P * Z') into tmp_mm1 reused... no, need separate
            # Actually: K = T * P * Z' * F^{-1}
            # Let's compute step by step:
            # tmp_mp = P * Z'  (m × p)
            # tmp_mp2 = T * tmp_mp = T * P * Z'  (m × p), but we don't have tmp_mp2
            # We can reuse tmp_pm (p × m) transposed... tricky

            # Better approach: compute F^{-1} * Z * P * T' and transpose
            # Or: solve F * K' = Z * P * T' for K'

            # Simpler: K = T * P * Z' * inv(F)
            # inv(F) = L'^{-1} * L^{-1}
            # K = T * P * Z' * L'^{-1} * L^{-1}

            # Kalman gain: K = T * P * Z' * F^{-1}
            # Use Kt[:,:,t] directly as target
            K_t = view(ws.Kt,:,:,t)

            # tmp_mp = P * Z' (m × p) already computed above
            # Compute K_t = T * (P * Z')
            mul!(K_t, ws.Tmat, ws.tmp_mp)           # K_t = T * P * Z' (m × p)

            # Compute K_t * F^{-1} in-place
            # F = L * L' (Cholesky), so F^{-1} = L'^{-1} * L^{-1}
            # K_t * F^{-1} = K_t * L'^{-1} * L^{-1}
            # rdiv!(K, L) solves K * L = X, giving K := K * L^{-1}
            L_lower = LowerTriangular(chol.factors)
            rdiv!(K_t, L_lower')  # K_t := K_t * L'^{-1}
            rdiv!(K_t, L_lower)   # K_t := K_t * L^{-1} = T*P*Z'*F^{-1}

            # Filtered state: a_filt = a + P * Z' * F^{-1} * v
            # We have K = T * P * Z' * F^{-1}, but we need P * Z' * F^{-1} * v
            # P * Z' * F^{-1} = (m×p) * (p×p) = m×p ... same as above but without T

            # Actually easier: a_filt = a + P * Z' * (F^{-1} * v)
            # We already have L \ v in tmp_p2
            # F^{-1} * v = L'^{-1} * (L^{-1} * v) = L'^{-1} * tmp_p2
            ldiv!(L_lower', ws.tmp_p2)  # tmp_p2 = F^{-1} * v

            # tmp_m2 = P * Z' * (F^{-1} * v)
            # tmp_mp = P * Z' (already computed)
            mul!(ws.tmp_m2, ws.tmp_mp, ws.tmp_p2)  # tmp_m2 = P * Z' * F^{-1} * v

            # a_filt = a + tmp_m2
            for i = 1:m
                ws.att[i, t] = a_curr[i] + ws.tmp_m2[i]
            end
            a_filt = view(ws.att, :, t)

            # Filtered covariance: P_filt = P - P * Z' * F^{-1} * Z * P
            # = P - (P * Z' * F^{-1}) * (Z * P)
            # tmp_mp = P * Z' (m × p)
            # Need P * Z' * F^{-1} (m × p)

            # Solve F * X' = (P * Z')' = Z * P' = Z * P for X
            # X = P * Z' * F^{-1}
            # Actually let's compute directly using the Cholesky

            # P * Z' * F^{-1}: solve for each row of P * Z'
            # (P * Z' * F^{-1})' = F^{-1} * Z * P
            # Solve F * Y = Z * P for Y, then (P * Z' * F^{-1}) = Y'

            # Compute F^{-1} * (Z * P)
            # Z * P = tmp_pm (p × m)
            mul!(ws.tmp_pm, ws.Z, P_curr)          # tmp_pm = Z * P (p × m)

            # F^{-1} * (Z*P) = L'^{-1} * L^{-1} * (Z*P)
            # ldiv!(L, X) solves L * X = B, giving X := L^{-1} * B
            ldiv!(L_lower, ws.tmp_pm)   # tmp_pm := L^{-1} * Z * P
            ldiv!(L_lower', ws.tmp_pm)  # tmp_pm := L'^{-1} * tmp_pm = F^{-1} * Z * P

            # P * Z' * F^{-1} = (F^{-1} * Z * P)' but need P * Z' * F^{-1} * Z * P
            # = P * Z' * (F^{-1} * Z * P) where F^{-1} * Z * P = tmp_pm
            # Wait, that's (m×p) * (p×m) = m×m. And tmp_mp = P * Z'.

            # Recompute tmp_mp = P * Z'
            mul!(ws.tmp_mp, P_curr, ws.Z')         # tmp_mp = P * Z' (m × p)

            # tmp_mm2 = tmp_mp * tmp_pm = P * Z' * F^{-1} * Z * P
            mul!(ws.tmp_mm2, ws.tmp_mp, ws.tmp_pm) # tmp_mm2 = P * Z' * F^{-1} * Z * P

            # P_filt = P - tmp_mm2
            for j = 1:m, i = 1:m
                ws.Ptt[i, j, t] = P_curr[i, j] - ws.tmp_mm2[i, j]
            end
            P_filt = view(ws.Ptt,:,:,t)

            # Predict next state: a = T * a_filt
            mul!(a_curr, ws.Tmat, a_filt)

            # Predict next covariance: P = T * P_filt * T' + RQR
            mul!(ws.tmp_mm2, ws.Tmat, P_filt)      # tmp_mm2 = T * P_filt
            mul!(P_curr, ws.tmp_mm2, ws.Tmat')     # P_curr = T * P_filt * T'
            for j = 1:m, i = 1:m
                P_curr[i, j] += ws.RQR[i, j]
            end
        end
    end

    # Add constant term
    ws.loglik += -p * ws.n_obs_valid * log2pi / 2

    return ws.loglik
end

# ============================================
# In-place Kalman smoother
# ============================================

"""
    kalman_smoother!(ws::KalmanWorkspace; crosscov::Bool=true)

Run RTS smoother in-place using filter results stored in workspace.

# Arguments
- `ws`: Workspace with filter results (must call `kalman_filter!` first)
- `crosscov`: If true, compute cross-lag covariances for EM (default: true)

# Stored Results
- `αs`: Smoothed states E[αₜ|y₁:ₙ]
- `Vs`: Smoothed covariances Var[αₜ|y₁:ₙ]
- `Pcross`: Cross-lag covariances Cov[αₜ₊₁,αₜ|y₁:ₙ] (if crosscov=true)

# Notes
Implements Durbin & Koopman (2012) backward recursion.
Cross-lag covariances computed during the same backward pass for efficiency.
"""
function kalman_smoother!(ws::KalmanWorkspace{T}; crosscov::Bool = true) where {T}
    m, n = ws.state_dim, ws.n_times
    p = ws.obs_dim

    # Initialize smoother recursion: r = 0, N = 0
    fill!(ws.r_smooth, zero(T))
    fill!(ws.N_smooth, zero(T))

    # Get Cholesky lower triangular for solves
    # We'll reconstruct from stored Ft_L

    @inbounds for t = n:-1:1
        # Views to stored filter results
        a_t = view(ws.at, :, t)
        P_t = view(ws.Pt,:,:,t)

        if ws.missing_mask[t]
            # Missing observation: r_{t-1} = T' * r_t, N_{t-1} = T' * N_t * T

            # tmp_m1 = T' * r
            mul!(ws.tmp_m1, ws.Tmat', ws.r_smooth)
            copyto!(ws.r_smooth, ws.tmp_m1)

            # tmp_mm1 = T' * N
            mul!(ws.tmp_mm1, ws.Tmat', ws.N_smooth)
            # tmp_mm2 = tmp_mm1 * T = T' * N * T
            mul!(ws.tmp_mm2, ws.tmp_mm1, ws.Tmat)
            copyto!(ws.N_smooth, ws.tmp_mm2)

            # Smoothed state: α = a + P * r
            mul!(ws.tmp_m1, P_t, ws.r_smooth)
            for i = 1:m
                ws.αs[i, t] = a_t[i] + ws.tmp_m1[i]
            end

            # Smoothed covariance: V = P - P * N * P
            mul!(ws.tmp_mm1, ws.N_smooth, P_t)     # N * P
            mul!(ws.tmp_mm2, P_t, ws.tmp_mm1)      # P * N * P
            for j = 1:m, i = 1:m
                ws.Vs[i, j, t] = P_t[i, j] - ws.tmp_mm2[i, j]
            end
        else
            # Valid observation
            v_t = view(ws.vt, :, t)
            K_t = view(ws.Kt,:,:,t)

            # Reconstruct L from stored Ft_L
            L_t = view(ws.Ft_L,:,:,t)
            L_lower = LowerTriangular(L_t)

            # L = T - K * Z
            # L_smooth = T - K * Z
            mul!(ws.L_smooth, K_t, ws.Z)           # L_smooth = K * Z
            for j = 1:m, i = 1:m
                ws.L_smooth[i, j] = ws.Tmat[i, j] - ws.L_smooth[i, j]
            end

            # F^{-1} * v: solve L * L' * x = v
            copyto!(ws.tmp_p1, v_t)
            ldiv!(L_lower, ws.tmp_p1)
            ldiv!(L_lower', ws.tmp_p1)             # tmp_p1 = F^{-1} * v

            # F^{-1} * Z: solve F * X = Z for X
            # F = L * L', so F^{-1} = L'^{-1} * L^{-1}
            # F^{-1} * Z = L'^{-1} * L^{-1} * Z
            copyto!(ws.tmp_pm, ws.Z)               # tmp_pm = Z (p × m)
            ldiv!(L_lower, ws.tmp_pm)              # tmp_pm := L^{-1} * Z
            ldiv!(L_lower', ws.tmp_pm)             # tmp_pm := L'^{-1} * tmp_pm = F^{-1} * Z

            # r_{t-1} = Z' * F^{-1} * v + L' * r
            # = Z' * tmp_p1 + L_smooth' * r_smooth
            mul!(ws.tmp_m1, ws.Z', ws.tmp_p1)      # Z' * F^{-1} * v
            mul!(ws.tmp_m2, ws.L_smooth', ws.r_smooth)  # L' * r
            for i = 1:m
                ws.r_smooth[i] = ws.tmp_m1[i] + ws.tmp_m2[i]
            end

            # N_{t-1} = Z' * F^{-1} * Z + L' * N * L
            # = Z' * tmp_pm + L' * N * L
            mul!(ws.tmp_mm1, ws.Z', ws.tmp_pm)     # Z' * F^{-1} * Z (m × m)
            mul!(ws.tmp_mm2, ws.L_smooth', ws.N_smooth)  # L' * N
            mul!(ws.tmp_mm3, ws.tmp_mm2, ws.L_smooth)    # L' * N * L
            for j = 1:m, i = 1:m
                ws.N_smooth[i, j] = ws.tmp_mm1[i, j] + ws.tmp_mm3[i, j]
            end

            # Smoothed state: α = a + P * r
            mul!(ws.tmp_m1, P_t, ws.r_smooth)
            for i = 1:m
                ws.αs[i, t] = a_t[i] + ws.tmp_m1[i]
            end

            # Smoothed covariance: V = P - P * N * P
            mul!(ws.tmp_mm1, ws.N_smooth, P_t)     # N * P
            mul!(ws.tmp_mm2, P_t, ws.tmp_mm1)      # P * N * P
            for j = 1:m, i = 1:m
                ws.Vs[i, j, t] = P_t[i, j] - ws.tmp_mm2[i, j]
            end
        end

        # Cross-lag covariance: Cov[α_{t}, α_{t-1} | y_{1:n}]
        # Computed during backward pass for t > 1
        # P_{t,t-1|n} = (I - P_t * N_{t-1}) * L_{t-1} * P_{t-1}
        # where L_{t-1} = T - K_{t-1} * Z
        #
        # Alternative formula (Shumway & Stoffer):
        # P_{t,t-1|n} = V_t * J_{t-1}'
        # where J_{t-1} = P_{t-1|t-1} * T' * inv(P_{t|t-1})
        #
        # We use: P_{t,t-1|n} = P_t * L_{t-1}' * (I - N_{t-1} * P_{t-1}) for t > 1
        # Actually easier: compute after main loop using stored V, P, T, K

        # For efficiency, compute cross-cov here using the formula:
        # Cov[α_{t+1}, α_t | y] for t = 1..n-1
        # We're at time t going backward, so we compute Cov[α_t, α_{t-1}]
        # which corresponds to Pcross[:,:,t-1]

        if crosscov && t > 1
            # Need: Cov[α_t, α_{t-1} | y_{1:n}] = P_{t,t-1|n}
            # Using formula: P_{t,t-1|n} = V_t * J_{t-1}'
            # where J_{t-1} = P_{t-1|t-1} * T' * inv(P_{t|t-1})
            #             = Ptt[:,:,t-1] * T' * inv(Pt[:,:,t])

            V_t = view(ws.Vs,:,:,t)
            Ptt_tm1 = view(ws.Ptt,:,:,(t-1))      # P_{t-1|t-1}
            Pt_t = view(ws.Pt,:,:,t)            # P_{t|t-1}

            # J_{t-1} = Ptt_{t-1} * T' * inv(Pt_t)
            # tmp_mm1 = Ptt_{t-1} * T'
            mul!(ws.tmp_mm1, Ptt_tm1, ws.Tmat')

            # tmp_mm2 = inv(Pt_t) - use Cholesky with regularization for numerical stability
            copyto!(ws.tmp_mm2, Pt_t)
            # Symmetrize and add small regularization
            eps_reg = T(1e-10) * max(one(T), tr(ws.tmp_mm2) / m)
            for j = 1:m
                for i = 1:(j-1)
                    avg = (ws.tmp_mm2[i, j] + ws.tmp_mm2[j, i]) / 2
                    ws.tmp_mm2[i, j] = avg
                    ws.tmp_mm2[j, i] = avg
                end
                ws.tmp_mm2[j, j] += eps_reg  # regularization on diagonal
            end
            chol_P = cholesky!(Symmetric(ws.tmp_mm2, :L))

            # J = tmp_mm1 * inv(Pt_t)
            # Solve Pt_t * J' = tmp_mm1' for J'
            # i.e., for each column j of tmp_mm1', solve Pt_t * x = tmp_mm1'[:,j]
            # tmp_mm1' is m × m, so we solve m systems

            # Actually: J = tmp_mm1 * inv(Pt_t)
            # J' = inv(Pt_t)' * tmp_mm1' = inv(Pt_t) * tmp_mm1' (symmetric)
            # Solve Pt_t * J' = tmp_mm1'

            # Copy tmp_mm1' into J_smooth (which is m × m)
            for j = 1:m, i = 1:m
                ws.J_smooth[i, j] = ws.tmp_mm1[j, i]  # transpose
            end

            # Solve L * L' * J' = tmp_mm1' column by column
            L_P = LowerTriangular(chol_P.L)
            for j = 1:m
                col = view(ws.J_smooth, :, j)
                ldiv!(L_P, col)
                ldiv!(L_P', col)
            end
            # Now J_smooth = J' = (Ptt_{t-1} * T' * inv(Pt_t))'

            # P_{t,t-1|n} = V_t * J_{t-1}' = V_t * J_smooth
            Pcross_t = view(ws.Pcross,:,:,(t-1))
            mul!(Pcross_t, V_t, ws.J_smooth)
        end
    end

    return nothing
end

# ============================================
# Combined filter and smooth
# ============================================

"""
    filter_and_smooth!(ws::KalmanWorkspace, y::AbstractMatrix; crosscov::Bool=true) -> loglik

Run filter and smoother in one call.

# Arguments
- `ws`: Pre-allocated workspace
- `y`: Observations (p × n)
- `crosscov`: Compute cross-lag covariances (default: true)

# Returns
- `loglik`: Log-likelihood
"""
function filter_and_smooth!(ws::KalmanWorkspace, y::AbstractMatrix; crosscov::Bool = true)
    loglik = kalman_filter!(ws, y)
    kalman_smoother!(ws; crosscov = crosscov)
    return loglik
end

# ============================================
# Additional Accessors (aliases and new ones)
# ============================================

"""Return cross-lag covariances Cov[αₜ₊₁,αₜ|y₁:ₙ] as view."""
crosslag_covs(ws::KalmanWorkspace) = ws.Pcross

"""Return innovations yₜ - Z*aₜ as view (alias for prediction_errors)."""
innovations(ws::KalmanWorkspace) = ws.vt

"""Return innovation covariances as view (alias for variances_prediction_errors)."""
innovation_covs(ws::KalmanWorkspace) = ws.Ft

"""Return stored Cholesky factors of Ft as view."""
cholesky_factors(ws::KalmanWorkspace) = ws.Ft_L

"""Return count of non-missing observations."""
n_valid_obs(ws::KalmanWorkspace) = ws.n_obs_valid

# ============================================
# Dimension accessors
# ============================================

obs_dim(ws::KalmanWorkspace) = ws.obs_dim
state_dim(ws::KalmanWorkspace) = ws.state_dim
shock_dim(ws::KalmanWorkspace) = ws.shock_dim
n_times(ws::KalmanWorkspace) = ws.n_times

# ============================================
# DiffuseKalmanWorkspace - Exact diffuse initialization
# ============================================

"""
    DiffuseKalmanWorkspace{T<:Real}

Pre-allocated workspace for in-place Kalman filter with exact diffuse initialization.

Extends `KalmanWorkspace` with additional storage for the diffuse period:
- `P1_star`: Finite part of initial covariance
- `P1_inf`: Diffuse (infinite) part of initial covariance
- `Pinf`: Current diffuse covariance (working storage)
- `Pstar`: Current finite covariance (working storage)
- `Finf`: Diffuse innovation covariance Z*Pinf*Z'
- `diffuse_flags`: Track which step type was used (1=Finf invertible, 0=singular, -1=missing)
- `d`: Length of diffuse period (updated during filtering)
- `diffuse_ended`: Flag indicating if diffuse period has ended

# Usage
```julia
ws = DiffuseKalmanWorkspace{Float64}(p, m, r, n)
set_params!(ws, Z, H, T, R, Q)
set_initial_diffuse!(ws, a1, P1_star, P1_inf)
kalman_filter_diffuse!(ws, y)
```

# See Also
- `KalmanWorkspace`: Standard filter workspace
- `kalman_filter_diffuse!`: In-place diffuse filter
- `kalman_filter_diffuse`: Pure functional version
"""
mutable struct DiffuseKalmanWorkspace{T<:Real}
    # Embed standard workspace (for shared functionality)
    base::KalmanWorkspace{T}

    # Diffuse initialization
    P1_star::Matrix{T}     # m × m: finite part of initial covariance
    P1_inf::Matrix{T}      # m × m: diffuse part of initial covariance

    # Diffuse period working storage
    Pinf::Matrix{T}        # m × m: current diffuse covariance
    Pstar::Matrix{T}       # m × m: current finite covariance
    Finf::Matrix{T}        # p × p: diffuse innovation covariance
    Finf_inv::Matrix{T}    # p × p: inverse of Finf (when invertible)

    # Diffuse period tracking
    diffuse_flags::Vector{Int}  # max length n: step type flags
    d::Int                      # current diffuse period length
    diffuse_ended::Bool         # flag for phase switching
    tol::T                      # tolerance for diffuse convergence

    # Additional scratch for diffuse computations
    tmp_mm_diff1::Matrix{T}  # m × m: for Linf, etc.
    tmp_mm_diff2::Matrix{T}  # m × m
    tmp_mp_diff::Matrix{T}   # m × p: for Kinf, Kstar
    tmp_pp_diff::Matrix{T}   # p × p: for Fstar
end

"""
    DiffuseKalmanWorkspace{T}(p::Int, m::Int, r::Int, n::Int; tol=1e-8) where T

Create diffuse workspace with specified dimensions.
"""
function DiffuseKalmanWorkspace{T}(
    p::Int,
    m::Int,
    r::Int,
    n::Int;
    tol::Real = 1e-8,
) where {T<:Real}
    base = KalmanWorkspace{T}(p, m, r, n)

    DiffuseKalmanWorkspace{T}(
        base,
        # Diffuse initialization
        Matrix{T}(undef, m, m),  # P1_star
        Matrix{T}(undef, m, m),  # P1_inf
        # Diffuse working storage
        Matrix{T}(undef, m, m),  # Pinf
        Matrix{T}(undef, m, m),  # Pstar
        Matrix{T}(undef, p, p),  # Finf
        Matrix{T}(undef, p, p),  # Finf_inv
        # Diffuse tracking
        Vector{Int}(undef, n),   # diffuse_flags (preallocate max)
        0,                       # d
        false,                   # diffuse_ended
        T(tol),                  # tol
        # Additional scratch
        Matrix{T}(undef, m, m),  # tmp_mm_diff1
        Matrix{T}(undef, m, m),  # tmp_mm_diff2
        Matrix{T}(undef, m, p),  # tmp_mp_diff
        Matrix{T}(undef, p, p),   # tmp_pp_diff
    )
end

DiffuseKalmanWorkspace(p::Int, m::Int, r::Int, n::Int; tol::Real = 1e-8) =
    DiffuseKalmanWorkspace{Float64}(p, m, r, n; tol = tol)

"""
    DiffuseKalmanWorkspace(Z, H, T, R, Q, a1, P1_star, P1_inf, n; tol=1e-8)

Create workspace from parameter matrices and set parameters.
"""
function DiffuseKalmanWorkspace(
    Z::AbstractMatrix{T},
    H::AbstractMatrix,
    Tmat::AbstractMatrix,
    R::AbstractMatrix,
    Q::AbstractMatrix,
    a1::AbstractVector,
    P1_star::AbstractMatrix,
    P1_inf::AbstractMatrix,
    n::Int;
    tol::Real = 1e-8,
) where {T}
    p, m = size(Z)
    r = size(Q, 1)

    ws = DiffuseKalmanWorkspace{T}(p, m, r, n; tol = tol)
    set_params!(ws, Z, H, Tmat, R, Q)
    set_initial_diffuse!(ws, a1, P1_star, P1_inf)

    return ws
end

# ============================================
# DiffuseKalmanWorkspace - Parameter setters
# ============================================

"""
    set_params!(ws::DiffuseKalmanWorkspace, Z, H, T, R, Q)

Set system matrices (delegates to base workspace).
"""
function set_params!(
    ws::DiffuseKalmanWorkspace,
    Z::AbstractMatrix,
    H::AbstractMatrix,
    Tmat::AbstractMatrix,
    R::AbstractMatrix,
    Q::AbstractMatrix,
)
    set_params!(ws.base, Z, H, Tmat, R, Q)
    return ws
end

"""
    set_initial_diffuse!(ws::DiffuseKalmanWorkspace, a1, P1_star, P1_inf)

Set initial state mean and diffuse covariance decomposition.
"""
function set_initial_diffuse!(
    ws::DiffuseKalmanWorkspace,
    a1::AbstractVector,
    P1_star::AbstractMatrix,
    P1_inf::AbstractMatrix,
)
    copyto!(ws.base.a1, a1)
    copyto!(ws.P1_star, P1_star)
    copyto!(ws.P1_inf, P1_inf)
    # Also set P1 in base to P1_star for compatibility
    copyto!(ws.base.P1, P1_star)
    return ws
end

# Delegate accessors to base
obs_dim(ws::DiffuseKalmanWorkspace) = ws.base.obs_dim
state_dim(ws::DiffuseKalmanWorkspace) = ws.base.state_dim
shock_dim(ws::DiffuseKalmanWorkspace) = ws.base.shock_dim
n_times(ws::DiffuseKalmanWorkspace) = ws.base.n_times

predicted_states(ws::DiffuseKalmanWorkspace) = ws.base.at
variances_predicted_states(ws::DiffuseKalmanWorkspace) = ws.base.Pt
filtered_states(ws::DiffuseKalmanWorkspace) = ws.base.att
variances_filtered_states(ws::DiffuseKalmanWorkspace) = ws.base.Ptt
smoothed_states(ws::DiffuseKalmanWorkspace) = ws.base.αs
variances_smoothed_states(ws::DiffuseKalmanWorkspace) = ws.base.Vs
prediction_errors(ws::DiffuseKalmanWorkspace) = ws.base.vt
variances_prediction_errors(ws::DiffuseKalmanWorkspace) = ws.base.Ft
kalman_gains(ws::DiffuseKalmanWorkspace) = ws.base.Kt
loglikelihood(ws::DiffuseKalmanWorkspace) = ws.base.loglik
missing_mask(ws::DiffuseKalmanWorkspace) = ws.base.missing_mask

"""
    diffuse_period(ws::DiffuseKalmanWorkspace) -> Int

Return the number of observations in the diffuse period.
"""
diffuse_period(ws::DiffuseKalmanWorkspace) = ws.d

"""
    diffuse_flags(ws::DiffuseKalmanWorkspace) -> Vector{Int}

Return flags for diffuse period steps (view of first d elements).
"""
diffuse_flags(ws::DiffuseKalmanWorkspace) = view(ws.diffuse_flags, 1:ws.d)

# ============================================
# In-place diffuse Kalman filter
# ============================================

"""
    _safe_inverse_inplace!(Finv, F, tmp, tol) -> flag

Attempt to invert F in-place into Finv. Returns 1 if successful, 0 if singular.
Uses tmp as scratch space.
"""
@inline function _safe_inverse_inplace!(
    Finv::AbstractMatrix{T},
    F::AbstractMatrix{T},
    tmp::AbstractMatrix{T},
    tol::Real,
) where {T}
    d = det(F)
    if abs(d) > tol
        copyto!(tmp, F)
        # Symmetrize
        p = size(F, 1)
        for j = 1:p, i = 1:(j-1)
            avg = (tmp[i, j] + tmp[j, i]) / 2
            tmp[i, j] = avg
            tmp[j, i] = avg
        end
        # Compute inverse via Cholesky
        chol = cholesky!(Symmetric(tmp, :L))
        copyto!(Finv, I)
        ldiv!(chol, Finv)
        return 1
    else
        return 0
    end
end

"""
    kalman_filter_diffuse!(ws::DiffuseKalmanWorkspace, y::AbstractMatrix) -> loglik

Run Kalman filter with exact diffuse initialization in-place.

# Arguments
- `ws`: Pre-allocated diffuse workspace (must call `set_initial_diffuse!` first)
- `y`: Observations (p × n matrix), missing values as NaN

# Returns
- `loglik`: Log-likelihood (non-diffuse observations only)

# Notes
The filter runs in two phases:
1. Diffuse period: While norm(Pinf) > tol, uses exact diffuse recursion
2. Non-diffuse period: Standard Kalman filter recursion

Only observations after the diffuse period contribute to the log-likelihood.
"""
function kalman_filter_diffuse!(ws::DiffuseKalmanWorkspace{T}, y::AbstractMatrix) where {T}
    base = ws.base
    p, m, n = base.obs_dim, base.state_dim, base.n_times

    @assert size(y) == (p, n) "Observation matrix size mismatch"

    # Reset state
    base.loglik = zero(T)
    base.n_obs_valid = 0
    ws.d = 0
    ws.diffuse_ended = false

    # Initialize: a = a1, Pstar = P1_star, Pinf = P1_inf
    a_curr = base.tmp_m1
    copyto!(a_curr, base.a1)
    copyto!(ws.Pstar, ws.P1_star)
    copyto!(ws.Pinf, ws.P1_inf)

    log2pi = log(T(2π))
    tol = ws.tol

    @inbounds for t = 1:n
        # Store predicted state
        for i = 1:m
            base.at[i, t] = a_curr[i]
        end
        # Store predicted covariance (Pstar is the finite part)
        for j = 1:m, i = 1:m
            base.Pt[i, j, t] = ws.Pstar[i, j]
        end

        y_t = view(y, :, t)

        if _has_missing_vec(y_t)
            base.missing_mask[t] = true

            # Store NaN for innovation
            for i = 1:p
                base.vt[i, t] = T(NaN)
            end

            # Compute F = Z * Pstar * Z' + H
            mul!(base.tmp_pm, base.Z, ws.Pstar)
            mul!(base.tmp_pp1, base.tmp_pm, base.Z')
            for j = 1:p, i = 1:p
                base.Ft[i, j, t] = base.tmp_pp1[i, j] + base.H[i, j]
                base.Ft_L[i, j, t] = zero(T)
            end

            for j = 1:p, i = 1:m
                base.Kt[i, j, t] = zero(T)
            end

            for i = 1:m
                base.att[i, t] = a_curr[i]
            end
            for j = 1:m, i = 1:m
                base.Ptt[i, j, t] = ws.Pstar[i, j]
            end

            if ws.diffuse_ended
                # Standard propagation
                mul!(base.tmp_m2, base.Tmat, a_curr)
                copyto!(a_curr, base.tmp_m2)

                mul!(base.tmp_mm2, base.Tmat, ws.Pstar)
                mul!(base.tmp_mm3, base.tmp_mm2, base.Tmat')
                for j = 1:m, i = 1:m
                    ws.Pstar[i, j] = base.tmp_mm3[i, j] + base.RQR[i, j]
                end
            else
                # Track diffuse period
                ws.d += 1
                ws.diffuse_flags[ws.d] = -1  # missing

                # Diffuse propagation
                mul!(base.tmp_m2, base.Tmat, a_curr)
                copyto!(a_curr, base.tmp_m2)

                # Pinf = T * Pinf * T'
                mul!(base.tmp_mm2, base.Tmat, ws.Pinf)
                mul!(ws.tmp_mm_diff1, base.tmp_mm2, base.Tmat')
                copyto!(ws.Pinf, ws.tmp_mm_diff1)

                # Pstar = T * Pstar * T' + RQR
                mul!(base.tmp_mm2, base.Tmat, ws.Pstar)
                mul!(base.tmp_mm3, base.tmp_mm2, base.Tmat')
                for j = 1:m, i = 1:m
                    ws.Pstar[i, j] = base.tmp_mm3[i, j] + base.RQR[i, j]
                end

                # Check convergence
                if norm(ws.Pinf) <= tol
                    ws.diffuse_ended = true
                end
            end
            continue
        end

        base.missing_mask[t] = false

        if ws.diffuse_ended
            # === Standard Kalman filter step ===
            base.n_obs_valid += 1

            # Innovation: v = y - Z * a
            mul!(base.tmp_p1, base.Z, a_curr)
            for i = 1:p
                base.vt[i, t] = y_t[i] - base.tmp_p1[i]
            end

            # F = Z * Pstar * Z' + H
            mul!(base.tmp_pm, base.Z, ws.Pstar)
            mul!(base.tmp_pp1, base.tmp_pm, base.Z')
            for j = 1:p, i = 1:p
                base.tmp_pp1[i, j] += base.H[i, j]
                base.Ft[i, j, t] = base.tmp_pp1[i, j]
            end

            # Symmetrize and Cholesky
            for j = 1:p, i = 1:(j-1)
                avg = (base.tmp_pp1[i, j] + base.tmp_pp1[j, i]) / 2
                base.tmp_pp1[i, j] = avg
                base.tmp_pp1[j, i] = avg
            end
            chol = cholesky!(Symmetric(base.tmp_pp1, :L))

            for j = 1:p, i = 1:p
                base.Ft_L[i, j, t] = i >= j ? chol.L[i, j] : zero(T)
            end

            # Log-likelihood
            logdetF = zero(T)
            for i = 1:p
                logdetF += 2 * log(chol.L[i, i])
            end

            v_t = view(base.vt, :, t)
            copyto!(base.tmp_p2, v_t)
            ldiv!(LowerTriangular(chol.L), base.tmp_p2)
            quad_form = zero(T)
            for i = 1:p
                quad_form += base.tmp_p2[i]^2
            end
            base.loglik += -T(0.5) * (logdetF + quad_form)

            # Kalman gain K = T * Pstar * Z' * F^{-1}
            mul!(base.tmp_mp, ws.Pstar, base.Z')
            K_t = view(base.Kt,:,:,t)
            mul!(K_t, base.Tmat, base.tmp_mp)
            L_lower = LowerTriangular(chol.factors)
            rdiv!(K_t, L_lower')
            rdiv!(K_t, L_lower)

            # Filtered state
            ldiv!(L_lower', base.tmp_p2)  # tmp_p2 = F^{-1} * v
            mul!(base.tmp_m2, base.tmp_mp, base.tmp_p2)
            for i = 1:m
                base.att[i, t] = a_curr[i] + base.tmp_m2[i]
            end

            # Filtered covariance
            mul!(base.tmp_pm, base.Z, ws.Pstar)
            ldiv!(L_lower, base.tmp_pm)
            ldiv!(L_lower', base.tmp_pm)
            mul!(base.tmp_mp, ws.Pstar, base.Z')
            mul!(base.tmp_mm2, base.tmp_mp, base.tmp_pm)
            for j = 1:m, i = 1:m
                base.Ptt[i, j, t] = ws.Pstar[i, j] - base.tmp_mm2[i, j]
            end

            # Predict next
            a_filt = view(base.att, :, t)
            P_filt = view(base.Ptt,:,:,t)
            mul!(a_curr, base.Tmat, a_filt)
            mul!(base.tmp_mm2, base.Tmat, P_filt)
            mul!(ws.Pstar, base.tmp_mm2, base.Tmat')
            for j = 1:m, i = 1:m
                ws.Pstar[i, j] += base.RQR[i, j]
            end

        else
            # === Diffuse filter step ===
            ws.d += 1

            # Innovation: v = y - Z * a
            mul!(base.tmp_p1, base.Z, a_curr)
            for i = 1:p
                base.vt[i, t] = y_t[i] - base.tmp_p1[i]
            end
            v_t = view(base.vt, :, t)

            # Finf = Z * Pinf * Z'
            mul!(base.tmp_pm, base.Z, ws.Pinf)
            mul!(ws.Finf, base.tmp_pm, base.Z')

            # Check if Finf is invertible
            flag = _safe_inverse_inplace!(ws.Finf_inv, ws.Finf, ws.tmp_pp_diff, tol)
            ws.diffuse_flags[ws.d] = flag

            if flag == 1
                # === Finf invertible: exact diffuse update ===

                # Kinf = T * Pinf * Z' * Finf^{-1}
                mul!(base.tmp_mp, ws.Pinf, base.Z')        # tmp_mp = Pinf * Z'
                mul!(ws.tmp_mp_diff, base.Tmat, base.tmp_mp)  # Kinf = T * Pinf * Z'
                # Kinf = Kinf * Finf_inv
                mul!(base.tmp_mp, ws.tmp_mp_diff, ws.Finf_inv)
                # Store as Kinf in base.tmp_mp, copy to ws.tmp_mp_diff
                copyto!(ws.tmp_mp_diff, base.tmp_mp)  # Kinf

                # Linf = T - Kinf * Z
                mul!(ws.tmp_mm_diff1, ws.tmp_mp_diff, base.Z)
                for j = 1:m, i = 1:m
                    ws.tmp_mm_diff1[i, j] = base.Tmat[i, j] - ws.tmp_mm_diff1[i, j]
                end
                # tmp_mm_diff1 = Linf

                # Fstar = Z * Pstar * Z' + H
                mul!(base.tmp_pm, base.Z, ws.Pstar)
                mul!(ws.tmp_pp_diff, base.tmp_pm, base.Z')
                for j = 1:p, i = 1:p
                    ws.tmp_pp_diff[i, j] += base.H[i, j]
                end
                # tmp_pp_diff = Fstar

                # Kstar = (T * Pstar * Z' + Kinf * Fstar) * Finf^{-1}
                # K1 = T * Pstar * Z' (m × p)
                mul!(base.tmp_mp, ws.Pstar, base.Z')            # tmp_mp = Pstar * Z' (m × p)
                K_t = view(base.Kt,:,:,t)
                mul!(K_t, base.Tmat, base.tmp_mp)               # K_t = T * Pstar * Z' (m × p)

                # K2 = Kinf * Fstar (m × p)
                mul!(base.tmp_mp, ws.tmp_mp_diff, ws.tmp_pp_diff)  # tmp_mp = Kinf * Fstar

                # Kstar = (K1 + K2) * Finf_inv
                for j = 1:p, i = 1:m
                    K_t[i, j] += base.tmp_mp[i, j]  # K1 + K2
                end
                mul!(base.tmp_mp, K_t, ws.Finf_inv)
                copyto!(K_t, base.tmp_mp)  # Kstar in Kt storage

                # Store Finf as F
                for j = 1:p, i = 1:p
                    base.Ft[i, j, t] = ws.Finf[i, j]
                    base.Ft_L[i, j, t] = zero(T)  # No valid Cholesky during diffuse
                end

                # Filtered state: a_filt = a + Pinf * Z' * Finf^{-1} * v
                mul!(base.tmp_mp, ws.Pinf, base.Z')  # Pinf * Z'
                mul!(base.tmp_m2, base.tmp_mp, ws.Finf_inv * v_t)
                for i = 1:m
                    base.att[i, t] = a_curr[i] + base.tmp_m2[i]
                end

                # Filtered covariance (approximation during diffuse)
                mul!(base.tmp_pm, base.Z, ws.Pstar)
                mul!(base.tmp_pp1, base.tmp_pm, base.Z')
                for j = 1:p, i = 1:p
                    base.tmp_pp1[i, j] += base.H[i, j]
                end
                # Use Fstar for filtered cov
                mul!(base.tmp_mp, ws.Pstar, base.Z')
                # Simple approximation: P_filt ≈ Pstar
                for j = 1:m, i = 1:m
                    base.Ptt[i, j, t] = ws.Pstar[i, j]
                end

                # State prediction: a = T * a + Kinf * v
                mul!(base.tmp_m2, base.Tmat, a_curr)
                mul!(base.tmp_m1, ws.tmp_mp_diff, v_t)  # Kinf * v
                for i = 1:m
                    a_curr[i] = base.tmp_m2[i] + base.tmp_m1[i]
                end

                # Covariance updates
                # Pinf' = T * Pinf * Linf'
                mul!(base.tmp_mm2, base.Tmat, ws.Pinf)
                mul!(ws.tmp_mm_diff2, base.tmp_mm2, ws.tmp_mm_diff1')
                copyto!(ws.Pinf, ws.tmp_mm_diff2)

                # Pstar' = T * Pstar * Linf' + Kinf * Finf * Kstar' + RQR
                mul!(base.tmp_mm2, base.Tmat, ws.Pstar)
                mul!(base.tmp_mm3, base.tmp_mm2, ws.tmp_mm_diff1')  # T*Pstar*Linf'

                # Kinf * Finf * Kstar'
                mul!(base.tmp_mp, ws.tmp_mp_diff, ws.Finf)  # Kinf * Finf
                mul!(ws.tmp_mm_diff2, base.tmp_mp, K_t')    # * Kstar'

                for j = 1:m, i = 1:m
                    ws.Pstar[i, j] =
                        base.tmp_mm3[i, j] + ws.tmp_mm_diff2[i, j] + base.RQR[i, j]
                end

                # No likelihood contribution when Finf invertible

            else
                # === Finf singular: use Fstar-based update ===

                # Fstar = Z * Pstar * Z' + H
                mul!(base.tmp_pm, base.Z, ws.Pstar)
                mul!(base.tmp_pp1, base.tmp_pm, base.Z')
                for j = 1:p, i = 1:p
                    base.tmp_pp1[i, j] += base.H[i, j]
                    base.Ft[i, j, t] = base.tmp_pp1[i, j]
                end

                # Symmetrize and Cholesky
                for j = 1:p, i = 1:(j-1)
                    avg = (base.tmp_pp1[i, j] + base.tmp_pp1[j, i]) / 2
                    base.tmp_pp1[i, j] = avg
                    base.tmp_pp1[j, i] = avg
                end
                chol = cholesky!(Symmetric(base.tmp_pp1, :L))

                for j = 1:p, i = 1:p
                    base.Ft_L[i, j, t] = i >= j ? chol.L[i, j] : zero(T)
                end

                # This observation contributes to likelihood
                base.n_obs_valid += 1
                logdetF = zero(T)
                for i = 1:p
                    logdetF += 2 * log(chol.L[i, i])
                end
                copyto!(base.tmp_p2, v_t)
                ldiv!(LowerTriangular(chol.L), base.tmp_p2)
                quad_form = zero(T)
                for i = 1:p
                    quad_form += base.tmp_p2[i]^2
                end
                base.loglik += -T(0.5) * (logdetF + quad_form)

                # Kstar = T * Pstar * Z' * Fstar^{-1}
                mul!(base.tmp_mp, ws.Pstar, base.Z')
                K_t = view(base.Kt,:,:,t)
                mul!(K_t, base.Tmat, base.tmp_mp)
                L_lower = LowerTriangular(chol.factors)
                rdiv!(K_t, L_lower')
                rdiv!(K_t, L_lower)

                # Lstar = T - Kstar * Z
                mul!(ws.tmp_mm_diff1, K_t, base.Z)
                for j = 1:m, i = 1:m
                    ws.tmp_mm_diff1[i, j] = base.Tmat[i, j] - ws.tmp_mm_diff1[i, j]
                end

                # Filtered state
                ldiv!(L_lower', base.tmp_p2)
                mul!(base.tmp_m2, base.tmp_mp, base.tmp_p2)
                for i = 1:m
                    base.att[i, t] = a_curr[i] + base.tmp_m2[i]
                end

                # Filtered covariance
                mul!(base.tmp_pm, base.Z, ws.Pstar)
                ldiv!(L_lower, base.tmp_pm)
                ldiv!(L_lower', base.tmp_pm)
                mul!(base.tmp_mp, ws.Pstar, base.Z')
                mul!(base.tmp_mm2, base.tmp_mp, base.tmp_pm)
                for j = 1:m, i = 1:m
                    base.Ptt[i, j, t] = ws.Pstar[i, j] - base.tmp_mm2[i, j]
                end

                # State prediction: a = T * a + Kstar * v
                mul!(base.tmp_m2, base.Tmat, a_curr)
                mul!(base.tmp_m1, K_t, v_t)
                for i = 1:m
                    a_curr[i] = base.tmp_m2[i] + base.tmp_m1[i]
                end

                # Covariance updates
                # Pinf' = T * Pinf * T'
                mul!(base.tmp_mm2, base.Tmat, ws.Pinf)
                mul!(ws.tmp_mm_diff2, base.tmp_mm2, base.Tmat')
                copyto!(ws.Pinf, ws.tmp_mm_diff2)

                # Pstar' = T * Pstar * Lstar' + RQR
                mul!(base.tmp_mm2, base.Tmat, ws.Pstar)
                mul!(base.tmp_mm3, base.tmp_mm2, ws.tmp_mm_diff1')
                for j = 1:m, i = 1:m
                    ws.Pstar[i, j] = base.tmp_mm3[i, j] + base.RQR[i, j]
                end
            end

            # Check if diffuse period ends
            if norm(ws.Pinf) <= tol
                ws.diffuse_ended = true
            end
        end
    end

    # Add constant term
    base.loglik += -p * base.n_obs_valid * log2pi / 2

    return base.loglik
end

# Unified API: kalman_filter! dispatches on workspace type
"""
    kalman_filter!(ws::DiffuseKalmanWorkspace, y) -> loglik

Run in-place Kalman filter with exact diffuse initialization.

This is the unified API: the workspace type determines whether to use
standard or exact diffuse initialization. Use `DiffuseKalmanWorkspace`
for exact diffuse, `KalmanWorkspace` for standard initialization.

See also: [`kalman_filter!`](@ref) (standard version with KalmanWorkspace)
"""
kalman_filter!(ws::DiffuseKalmanWorkspace, y::AbstractMatrix) =
    kalman_filter_diffuse!(ws, y)

# ============================================
# EM Algorithm Workspace
# ============================================

"""
    EMWorkspace{T<:Real}

Pre-allocated workspace for EM algorithm sufficient statistics and updates.

For a state-space model:
    y_t = Z * α_t + ε_t,  ε_t ~ N(0, H)
    α_{t+1} = T * α_t + R * η_t,  η_t ~ N(0, Q)

The EM algorithm computes sufficient statistics from smoothed states:
    S_00 = Σ_{t=1}^{n-1} E[α_t α_t' | Y]
    S_11 = Σ_{t=2}^{n} E[α_t α_t' | Y]
    S_10 = Σ_{t=2}^{n} E[α_t α_{t-1}' | Y]

For observation equation:
    S_yy = Σ_{t=1}^{n} y_t y_t'
    S_yα = Σ_{t=1}^{n} y_t E[α_t' | Y]
    S_αα = Σ_{t=1}^{n} E[α_t α_t' | Y]
"""
mutable struct EMWorkspace{T<:Real}
    # Dimensions
    obs_dim::Int      # p
    state_dim::Int    # m
    shock_dim::Int    # r
    n_times::Int      # n

    # Sufficient statistics for state equation (m × m)
    S_00::Matrix{T}   # Σ E[α_t α_t' | Y] for t = 1:n-1
    S_11::Matrix{T}   # Σ E[α_t α_t' | Y] for t = 2:n
    S_10::Matrix{T}   # Σ E[α_t α_{t-1}' | Y] for t = 2:n

    # Sufficient statistics for observation equation
    S_yy::Matrix{T}   # Σ y_t y_t' (p × p)
    S_yα::Matrix{T}   # Σ y_t α_t' (p × m)
    S_αα::Matrix{T}   # Σ E[α_t α_t' | Y] for t = 1:n (m × m)

    # Scratch space for M-step computations
    tmp_mm1::Matrix{T}  # m × m
    tmp_mm2::Matrix{T}  # m × m
    tmp_pp1::Matrix{T}  # p × p
    tmp_pm::Matrix{T}   # p × m
    tmp_rr::Matrix{T}   # r × r
    tmp_mr::Matrix{T}   # m × r
    tmp_rm::Matrix{T}   # r × m

    # For dynamic factor models: masks for constrained estimation
    # These indicate which parameters are free vs fixed
    Z_free::BitMatrix      # p × m: which Z elements to estimate
    H_diag_only::Bool      # If true, H is diagonal
    T_free::BitMatrix      # m × m: which T elements to estimate
    Q_free::BitMatrix      # r × r: which Q elements to estimate
end

"""
    EMWorkspace(p::Int, m::Int, r::Int, n::Int, ::Type{T}=Float64) where T

Create EM workspace with given dimensions.
"""
function EMWorkspace(p::Int, m::Int, r::Int, n::Int, ::Type{T} = Float64) where {T}
    EMWorkspace{T}(
        p,
        m,
        r,
        n,
        # Sufficient statistics
        zeros(T, m, m),  # S_00
        zeros(T, m, m),  # S_11
        zeros(T, m, m),  # S_10
        zeros(T, p, p),  # S_yy
        zeros(T, p, m),  # S_yα
        zeros(T, m, m),  # S_αα
        # Scratch space
        zeros(T, m, m),
        zeros(T, m, m),
        zeros(T, p, p),
        zeros(T, p, m),
        zeros(T, r, r),
        zeros(T, m, r),
        zeros(T, r, m),
        # Default: all parameters free
        trues(p, m),     # Z_free
        false,           # H_diag_only
        trues(m, m),     # T_free
        trues(r, r),      # Q_free
    )
end

"""
    EMWorkspace(kf_ws::KalmanWorkspace{T}) where T

Create EM workspace matching dimensions of Kalman workspace.
"""
function EMWorkspace(kf_ws::KalmanWorkspace{T}) where {T}
    EMWorkspace(kf_ws.obs_dim, kf_ws.state_dim, kf_ws.shock_dim, kf_ws.n_times, T)
end

# ============================================
# E-step: Compute sufficient statistics
# ============================================

"""
    compute_sufficient_stats!(em_ws::EMWorkspace, kf_ws::KalmanWorkspace, y::AbstractMatrix)

Compute sufficient statistics from smoothed states for M-step.
Assumes kalman_smoother!(kf_ws; crosscov=true) has been called.
"""
function compute_sufficient_stats!(
    em_ws::EMWorkspace{T},
    kf_ws::KalmanWorkspace{T},
    y::AbstractMatrix,
) where {T}
    p, m, n = em_ws.obs_dim, em_ws.state_dim, em_ws.n_times

    # Reset sufficient statistics
    fill!(em_ws.S_00, zero(T))
    fill!(em_ws.S_11, zero(T))
    fill!(em_ws.S_10, zero(T))
    fill!(em_ws.S_yy, zero(T))
    fill!(em_ws.S_yα, zero(T))
    fill!(em_ws.S_αα, zero(T))

    n_valid = 0

    @inbounds for t = 1:n
        α_t = view(kf_ws.αs, :, t)       # Smoothed state E[α_t | Y]
        V_t = view(kf_ws.Vs,:,:,t)    # Smoothed covariance Var[α_t | Y]

        # E[α_t α_t' | Y] = V_t + α_t α_t'
        # Accumulate into S_αα
        for j = 1:m, i = 1:m
            em_ws.S_αα[i, j] += V_t[i, j] + α_t[i] * α_t[j]
        end

        # For state equation: need S_00 (t=1:n-1), S_11 (t=2:n), S_10 (t=2:n)
        if t < n
            # S_00: Σ_{t=1}^{n-1} E[α_t α_t' | Y]
            for j = 1:m, i = 1:m
                em_ws.S_00[i, j] += V_t[i, j] + α_t[i] * α_t[j]
            end
        end

        if t > 1
            α_tm1 = view(kf_ws.αs, :, t-1)
            # S_11: Σ_{t=2}^{n} E[α_t α_t' | Y]
            for j = 1:m, i = 1:m
                em_ws.S_11[i, j] += V_t[i, j] + α_t[i] * α_t[j]
            end

            # S_10: Σ_{t=2}^{n} E[α_t α_{t-1}' | Y]
            # E[α_t α_{t-1}' | Y] = Pcross_{t-1} + α_t α_{t-1}'
            # where Pcross[:,:,t-1] = Cov[α_t, α_{t-1} | Y]
            Pcross_tm1 = view(kf_ws.Pcross,:,:,(t-1))
            for j = 1:m, i = 1:m
                em_ws.S_10[i, j] += Pcross_tm1[i, j] + α_t[i] * α_tm1[j]
            end
        end

        # For observation equation: only valid (non-missing) observations
        if !kf_ws.missing_mask[t]
            n_valid += 1
            y_t = view(y, :, t)

            # S_yy: Σ y_t y_t'
            for j = 1:p, i = 1:p
                em_ws.S_yy[i, j] += y_t[i] * y_t[j]
            end

            # S_yα: Σ y_t E[α_t' | Y]
            for j = 1:m, i = 1:p
                em_ws.S_yα[i, j] += y_t[i] * α_t[j]
            end
        end
    end

    return n_valid
end

# ============================================
# M-step: Update parameters
# ============================================

"""
    update_Z!(kf_ws::KalmanWorkspace, em_ws::EMWorkspace, n_valid::Int)

M-step update for observation matrix Z.
Z_new = S_yα * S_αα^{-1}

Only updates elements where em_ws.Z_free is true.
"""
function update_Z!(kf_ws::KalmanWorkspace{T}, em_ws::EMWorkspace{T}, n_valid::Int) where {T}
    p, m = em_ws.obs_dim, em_ws.state_dim

    # Compute S_αα^{-1} (use only valid observations portion)
    # tmp_mm1 = S_αα
    copyto!(em_ws.tmp_mm1, em_ws.S_αα)

    # Regularize for numerical stability
    for i = 1:m
        em_ws.tmp_mm1[i, i] += T(1e-10)
    end

    # Cholesky factorization
    chol = cholesky!(Symmetric(em_ws.tmp_mm1, :L))

    # tmp_pm = S_yα * S_αα^{-1}
    # Solve S_αα * X' = S_yα' for X', then X = (X')'
    # Or: (S_yα * S_αα^{-1})' = S_αα^{-1} * S_yα'
    # Solve S_αα * Y = S_yα' for Y, then Z_new = Y'

    # tmp_pm = S_yα (will become Z_new after solving)
    copyto!(em_ws.tmp_pm, em_ws.S_yα)

    # Solve: need S_yα * S_αα^{-1}
    # S_yα is p × m, S_αα is m × m
    # Result is p × m
    # Use rdiv!: tmp_pm := tmp_pm * S_αα^{-1}
    L_lower = LowerTriangular(chol.L)
    rdiv!(em_ws.tmp_pm, L_lower')
    rdiv!(em_ws.tmp_pm, L_lower)

    # Update only free elements
    @inbounds for j = 1:m, i = 1:p
        if em_ws.Z_free[i, j]
            kf_ws.Z[i, j] = em_ws.tmp_pm[i, j]
        end
    end

    return nothing
end

"""
    update_H!(kf_ws::KalmanWorkspace, em_ws::EMWorkspace, n_valid::Int)

M-step update for observation covariance H.
H_new = (1/n) * (S_yy - S_yα * S_αα^{-1} * S_yα')
      = (1/n) * (S_yy - Z_new * S_yα')

If H_diag_only is true, only diagonal elements are updated.
"""
function update_H!(kf_ws::KalmanWorkspace{T}, em_ws::EMWorkspace{T}, n_valid::Int) where {T}
    p, m = em_ws.obs_dim, em_ws.state_dim

    # H = (1/n) * (S_yy - Z * S_yα')
    # tmp_pp1 = Z * S_yα'
    mul!(em_ws.tmp_pp1, kf_ws.Z, em_ws.S_yα')

    # H = (S_yy - tmp_pp1) / n_valid
    scale = one(T) / n_valid

    if em_ws.H_diag_only
        @inbounds for i = 1:p
            h_ii = (em_ws.S_yy[i, i] - em_ws.tmp_pp1[i, i]) * scale
            kf_ws.H[i, i] = max(h_ii, T(1e-10))  # Ensure positive
        end
    else
        @inbounds for j = 1:p, i = 1:p
            kf_ws.H[i, j] = (em_ws.S_yy[i, j] - em_ws.tmp_pp1[i, j]) * scale
        end
        # Ensure symmetry and positive definiteness
        for j = 1:p, i = 1:(j-1)
            avg = (kf_ws.H[i, j] + kf_ws.H[j, i]) / 2
            kf_ws.H[i, j] = avg
            kf_ws.H[j, i] = avg
        end
        # Add small regularization to diagonal
        for i = 1:p
            kf_ws.H[i, i] = max(kf_ws.H[i, i], T(1e-10))
        end
    end

    return nothing
end

"""
    update_T!(kf_ws::KalmanWorkspace, em_ws::EMWorkspace)

M-step update for state transition matrix T.
T_new = S_10 * S_00^{-1}

Only updates elements where em_ws.T_free is true.
"""
function update_T!(kf_ws::KalmanWorkspace{T}, em_ws::EMWorkspace{T}) where {T}
    m = em_ws.state_dim

    # tmp_mm1 = S_00
    copyto!(em_ws.tmp_mm1, em_ws.S_00)

    # Regularize
    for i = 1:m
        em_ws.tmp_mm1[i, i] += T(1e-10)
    end

    # Cholesky
    chol = cholesky!(Symmetric(em_ws.tmp_mm1, :L))

    # T_new = S_10 * S_00^{-1}
    # tmp_mm2 = S_10, then solve
    copyto!(em_ws.tmp_mm2, em_ws.S_10)

    L_lower = LowerTriangular(chol.L)
    rdiv!(em_ws.tmp_mm2, L_lower')
    rdiv!(em_ws.tmp_mm2, L_lower)

    # Update only free elements
    @inbounds for j = 1:m, i = 1:m
        if em_ws.T_free[i, j]
            kf_ws.Tmat[i, j] = em_ws.tmp_mm2[i, j]
        end
    end

    return nothing
end

"""
    update_Q!(kf_ws::KalmanWorkspace, em_ws::EMWorkspace)

M-step update for state covariance Q.
Q_new = (1/(n-1)) * R' * (S_11 - T * S_10' - S_10 * T' + T * S_00 * T') * R

For the case R = [I; 0] (shocks affect first r states only):
Q_new = (1/(n-1)) * (S_11[1:r,1:r] - T[1:r,:] * S_10'[:,1:r] - ...)

Simplified when R = I (r = m):
Q_new = (1/(n-1)) * (S_11 - T * S_10' - S_10 * T' + T * S_00 * T')
"""
function update_Q!(kf_ws::KalmanWorkspace{T}, em_ws::EMWorkspace{T}) where {T}
    m, r = em_ws.state_dim, em_ws.shock_dim
    n = em_ws.n_times

    # Compute: S_11 - T * S_10' - S_10 * T' + T * S_00 * T'
    # = S_11 - S_10 * T' - T * S_10' + T * S_00 * T'

    # tmp_mm1 = T * S_00
    mul!(em_ws.tmp_mm1, kf_ws.Tmat, em_ws.S_00)

    # tmp_mm2 = T * S_00 * T'
    mul!(em_ws.tmp_mm2, em_ws.tmp_mm1, kf_ws.Tmat')

    # tmp_mm2 += S_11
    for j = 1:m, i = 1:m
        em_ws.tmp_mm2[i, j] += em_ws.S_11[i, j]
    end

    # tmp_mm1 = S_10 * T'
    mul!(em_ws.tmp_mm1, em_ws.S_10, kf_ws.Tmat')

    # tmp_mm2 -= S_10 * T' + T * S_10' = S_10 * T' + (S_10 * T')'
    for j = 1:m, i = 1:m
        em_ws.tmp_mm2[i, j] -= em_ws.tmp_mm1[i, j] + em_ws.tmp_mm1[j, i]
    end

    # Scale by 1/(n-1)
    scale = one(T) / (n - 1)

    # Now compute R' * tmp_mm2 * R to get Q
    # For general R (m × r): Q = R' * tmp_mm2 * R
    if r == m
        # R = I case: Q = tmp_mm2
        @inbounds for j = 1:r, i = 1:r
            if em_ws.Q_free[i, j]
                kf_ws.Q[i, j] = em_ws.tmp_mm2[i, j] * scale
            end
        end
    else
        # General case: Q = R' * tmp_mm2 * R
        # tmp_mr = tmp_mm2 * R
        mul!(em_ws.tmp_mr, em_ws.tmp_mm2, kf_ws.R)
        # tmp_rr = R' * tmp_mr
        mul!(em_ws.tmp_rr, kf_ws.R', em_ws.tmp_mr)

        @inbounds for j = 1:r, i = 1:r
            if em_ws.Q_free[i, j]
                kf_ws.Q[i, j] = em_ws.tmp_rr[i, j] * scale
            end
        end
    end

    # Ensure symmetry and positive definiteness
    for j = 1:r, i = 1:(j-1)
        avg = (kf_ws.Q[i, j] + kf_ws.Q[j, i]) / 2
        kf_ws.Q[i, j] = avg
        kf_ws.Q[j, i] = avg
    end
    for i = 1:r
        kf_ws.Q[i, i] = max(kf_ws.Q[i, i], T(1e-10))
    end

    # Update RQR
    _update_RQR!(kf_ws)

    return nothing
end

# ============================================
# Main EM Algorithm
# ============================================

"""
    EMResult{T}

Result of EM estimation.
"""
struct EMResult{T<:Real}
    converged::Bool
    iterations::Int
    loglik::T
    loglik_history::Vector{T}
    Z::Matrix{T}
    H::Matrix{T}
    T::Matrix{T}
    Q::Matrix{T}
end

"""
    em_estimate!(kf_ws::KalmanWorkspace, em_ws::EMWorkspace, y::AbstractMatrix;
                 maxiter::Int=500, tol::Real=1e-6, verbose::Bool=false,
                 estimate_Z::Bool=true, estimate_H::Bool=true,
                 estimate_T::Bool=true, estimate_Q::Bool=true) -> EMResult

Run EM algorithm to estimate state-space model parameters.

# Arguments
- `kf_ws`: Kalman filter/smoother workspace (contains initial parameters)
- `em_ws`: EM algorithm workspace
- `y`: Observations (p × n matrix), missing values as NaN

# Keyword Arguments
- `maxiter`: Maximum number of EM iterations (default: 500)
- `tol`: Convergence tolerance for log-likelihood change (default: 1e-6)
- `verbose`: Print progress (default: false)
- `estimate_Z`: Update observation matrix Z (default: true)
- `estimate_H`: Update observation covariance H (default: true)
- `estimate_T`: Update transition matrix T (default: true)
- `estimate_Q`: Update state covariance Q (default: true)

# Returns
- `EMResult` containing convergence info and estimated parameters
"""
function em_estimate!(
    kf_ws::KalmanWorkspace{T},
    em_ws::EMWorkspace{T},
    y::AbstractMatrix;
    maxiter::Int = 500,
    tol::Real = 1e-6,
    verbose::Bool = false,
    estimate_Z::Bool = true,
    estimate_H::Bool = true,
    estimate_T::Bool = true,
    estimate_Q::Bool = true,
) where {T}

    loglik_history = Vector{T}(undef, maxiter)
    ll_prev = T(-Inf)
    converged = false
    iter = 0

    for i = 1:maxiter
        iter = i

        # E-step: Filter and smooth
        filter_and_smooth!(kf_ws, y)
        ll = kf_ws.loglik
        loglik_history[i] = ll

        # Check convergence
        if i > 1
            ll_change = abs(ll - ll_prev)
            rel_change = ll_change / (abs(ll_prev) + T(1e-10))

            if verbose && (i % 10 == 0 || i <= 5)
                println("EM iter $i: loglik = $ll, change = $ll_change, rel = $rel_change")
            end

            if rel_change < tol
                converged = true
                if verbose
                    println("EM converged at iteration $i")
                end
                break
            end
        elseif verbose
            println("EM iter $i: loglik = $ll")
        end

        ll_prev = ll

        # E-step: Compute sufficient statistics
        n_valid = compute_sufficient_stats!(em_ws, kf_ws, y)

        # M-step: Update parameters
        if estimate_Z
            update_Z!(kf_ws, em_ws, n_valid)
        end
        if estimate_H
            update_H!(kf_ws, em_ws, n_valid)
        end
        if estimate_T
            update_T!(kf_ws, em_ws)
        end
        if estimate_Q
            update_Q!(kf_ws, em_ws)
        end
    end

    # Final log-likelihood
    filter_and_smooth!(kf_ws, y)

    return EMResult{T}(
        converged,
        iter,
        kf_ws.loglik,
        loglik_history[1:iter],
        copy(kf_ws.Z),
        copy(kf_ws.H),
        copy(kf_ws.Tmat),
        copy(kf_ws.Q),
    )
end

# ============================================
# Dynamic Factor Model Helper Functions
# ============================================

"""
    setup_dfm_workspaces(p::Int, k::Int, s::Int, n::Int; T::Type=Float64)

Create KalmanWorkspace and EMWorkspace for a dynamic factor model.

# Arguments
- `p`: Number of observable series
- `k`: Number of latent factors
- `s`: Number of factor VAR lags
- `n`: Number of time periods

# Returns
Tuple `(kf_ws, em_ws)` of KalmanWorkspace and EMWorkspace.

# DFM State Space Form
The state αₜ = [fₜ', fₜ₋₁', ..., fₜ₋ₛ₊₁']' has dimension m = k × s.
Observation: yₜ = Λ fₜ + εₜ = [Λ, 0, ..., 0] αₜ + εₜ
Transition: αₜ₊₁ = T αₜ + R ηₜ where T is companion form and R = [Iₖ; 0; ...].
"""
function setup_dfm_workspaces(
    p::Int,
    k::Int,
    s::Int,
    n::Int;
    T::Type{TT} = Float64,
) where {TT<:Real}
    m = k * s    # State dimension
    r = k        # Shock dimension (innovations only on current factors)

    # Create Kalman workspace
    kf_ws = KalmanWorkspace{TT}(p, m, r, n)

    # Zero out all parameter matrices (constructor uses undef)
    fill!(kf_ws.Z, zero(TT))
    fill!(kf_ws.H, zero(TT))
    fill!(kf_ws.Tmat, zero(TT))
    fill!(kf_ws.R, zero(TT))
    fill!(kf_ws.Q, zero(TT))
    fill!(kf_ws.a1, zero(TT))
    fill!(kf_ws.P1, zero(TT))

    # Initialize Z: [Λ, 0, ..., 0] where Λ is p × k
    # Loadings go in first k columns, initialized by caller with PCA

    # Initialize H: diagonal idiosyncratic variances (set by caller)
    # For now, set default small diagonal to avoid singularity
    for i = 1:p
        kf_ws.H[i, i] = one(TT)
    end

    # Initialize T: companion form for VAR(s)
    # Top k rows: [Φ₁, Φ₂, ..., Φₛ] (set by caller)
    # Below: [I, 0, ..., 0; 0, I, ..., 0; ...]
    # Set identity blocks for companion form
    for lag = 1:(s-1)
        row_start = lag * k + 1
        col_start = (lag - 1) * k + 1
        for i = 1:k
            kf_ws.Tmat[row_start+i-1, col_start+i-1] = one(TT)
        end
    end

    # Initialize R: [Iₖ; 0; ...]
    for i = 1:k
        kf_ws.R[i, i] = one(TT)
    end

    # Initialize Q: factor innovation covariance (k × k)
    for i = 1:k
        kf_ws.Q[i, i] = one(TT)
    end
    _update_RQR!(kf_ws)

    # Initialize P1: diffuse
    for i = 1:m
        kf_ws.P1[i, i] = TT(1e7)
    end

    # Create EM workspace
    em_ws = EMWorkspace(kf_ws)

    return (kf_ws, em_ws)
end

"""
    extract_loadings(result::EMResult, k::Int) -> Matrix

Extract factor loadings matrix Λ (p × k) from EM result.
The loadings are the first k columns of the Z matrix.
"""
function extract_loadings(result::EMResult, k::Int)
    return result.Z[:, 1:k]
end

"""
    extract_factors(kf_ws::KalmanWorkspace, k::Int) -> Matrix

Extract smoothed factors (k × n) from Kalman workspace.
The factors are the first k rows of the smoothed state vector αₜ.
"""
function extract_factors(kf_ws::KalmanWorkspace, k::Int)
    return kf_ws.αs[1:k, :]
end

# ============================================
# Estimation Method Types (StatsAPI-style)
# ============================================

"""
    EM

Estimation method type for EM (Expectation-Maximization) algorithm.
Used with `fit!(EM(), model, data; kwargs...)`.
"""
struct EM end

"""
    MLE

Estimation method type for direct Maximum Likelihood Estimation.
Used with `fit!(MLE(), model, data; kwargs...)`.
"""
struct MLE end

# ============================================
# Abstract State-Space Model
# ============================================

"""
    AbstractStateSpaceModel

Abstract supertype for all state-space models in Siphon.jl.
Provides a common interface for fitting, querying, and forecasting.

Subtypes:
- `StateSpaceModel` - General state-space model from SSMSpec
- `DynamicFactorModel` - Dynamic factor model with EM estimation
"""
abstract type AbstractStateSpaceModel end

# ============================================
# StateSpaceModel
# ============================================

"""
    StateSpaceModelWorkspaces{T}

Pre-allocated workspaces for in-place EM computation.
Only allocated for large models (dimensions > STATIC_THRESHOLD).
"""
struct StateSpaceModelWorkspaces{T<:Real}
    kf_ws::KalmanWorkspace{T}
    em_ws::EMWorkspace{T}
end

"""
    StateSpaceModel{T}

Mutable state-space model container wrapping an SSMSpec.

Supports both MLE and EM estimation with automatic backend selection for EM.

# Usage
```julia
# Create from SSMSpec
spec = local_level(var_obs=:free, var_level=:free)
model = StateSpaceModel(spec, 200)

# Fit with MLE (pure Kalman filter, AD-compatible)
fit!(MLE(), model, y)

# Or fit with EM (auto-selects backend based on dimensions)
fit!(EM(), model, y; maxiter=500, verbose=true)

# Access results
params = parameters(model)         # NamedTuple
f = filtered_states(model)         # m × n
s = smoothed_states(model)         # computed on-demand

# Forecast
fc = forecast(model, 10)
```

# Backend Selection for EM
- If max(n_states, n_obs) > 13: uses in-place KalmanWorkspace (zero allocations)
- Otherwise: uses pure StaticArrays implementation

# Fields (internal)
See source for field documentation.
"""
mutable struct StateSpaceModel{T<:Real} <: AbstractStateSpaceModel
    # Specification
    spec::SSMSpec
    n_times::Int

    # Fitted parameters
    theta_values::Vector{T}
    theta_fitted::Bool

    # Fit statistics
    loglik::T
    fitted::Bool
    converged::Bool
    iterations::Int
    backend::Symbol  # :none, :mle, :em_static, :em_inplace

    # Filter results (pre-allocated)
    at::Matrix{T}           # Predicted states (m × n)
    Pt::Array{T,3}          # Predicted covariances (m × m × n)
    att::Matrix{T}          # Filtered states (m × n)
    Ptt::Array{T,3}         # Filtered covariances (m × m × n)
    vt::Matrix{T}           # Innovations (p × n)
    Ft::Array{T,3}          # Innovation covariances (p × p × n)
    Kt::Array{T,3}          # Kalman gains (m × p × n)
    missing_mask::BitVector
    filter_valid::Bool

    # Smoother cache (lazily computed)
    smoother_computed::Bool
    smoothed_alpha::Matrix{T}
    smoothed_V::Array{T,3}

    # In-place workspaces (for large EM)
    kf_workspace_allocated::Bool
    workspaces_ref::Base.RefValue{Any}
end

"""
    StateSpaceModel(spec::SSMSpec, n_times::Int; T::Type{<:Real}=Float64)

Construct a StateSpaceModel from an SSMSpec.

# Arguments
- `spec::SSMSpec`: Model specification from DSL (local_level, ar1, custom_ssm, etc.)
- `n_times::Int`: Number of time periods

# Keyword Arguments
- `T::Type{<:Real}`: Element type (default: Float64)

# Behavior
- Pre-allocates filter storage
- If max(n_states, n_obs) > STATIC_THRESHOLD: also pre-allocates KalmanWorkspace for EM

# Example
```julia
spec = local_level(var_obs=:free, var_level=:free)
model = StateSpaceModel(spec, 200)
```
"""
function StateSpaceModel(spec::SSMSpec, n_times::Int; T::Type{<:Real} = Float64)
    m = spec.n_states
    p = spec.n_obs
    r = spec.n_shocks
    n = n_times

    # Determine if we need large-model workspaces
    needs_inplace = max(m, p, r) > STATIC_THRESHOLD

    # Pre-allocate filter storage
    at = Matrix{T}(undef, m, n)
    Pt = Array{T,3}(undef, m, m, n)
    att = Matrix{T}(undef, m, n)
    Ptt = Array{T,3}(undef, m, m, n)
    vt = Matrix{T}(undef, p, n)
    Ft = Array{T,3}(undef, p, p, n)
    Kt = Array{T,3}(undef, m, p, n)
    missing_mask = BitVector(undef, n)

    # Smoother storage (allocated but not valid until computed)
    smoothed_alpha = Matrix{T}(undef, m, n)
    smoothed_V = Array{T,3}(undef, m, m, n)

    # Parameter storage
    n_params = length(spec.params)
    theta_values = zeros(T, n_params)

    # Create workspace reference (will be populated if needed)
    workspaces_ref = Ref{Any}(nothing)

    # Pre-allocate in-place workspaces if needed for large models
    if needs_inplace
        # Build initial state-space matrices from initial parameter values
        theta_init = [p.init for p in spec.params]
        names = Tuple(prm.name for prm in spec.params)
        theta_nt = NamedTuple{names}(Tuple(theta_init))

        # Use Siphon's build functions
        kfparms = Siphon.build_kfparms(spec, theta_nt)
        a1_init, P1_init = Siphon.build_initial_state(spec, theta_nt)

        kf_ws = KalmanWorkspace(
            kfparms.Z,
            kfparms.H,
            kfparms.T,
            kfparms.R,
            kfparms.Q,
            a1_init,
            P1_init,
            n,
        )
        em_ws = EMWorkspace(kf_ws)

        workspaces_ref[] = StateSpaceModelWorkspaces{T}(kf_ws, em_ws)
    end

    StateSpaceModel{T}(
        spec,
        n,
        theta_values,
        false,  # theta not fitted
        T(-Inf),
        false,
        false,
        0,
        :none,
        at,
        Pt,
        att,
        Ptt,
        vt,
        Ft,
        Kt,
        missing_mask,
        false,
        false,
        smoothed_alpha,
        smoothed_V,
        needs_inplace,
        workspaces_ref,
    )
end

# ============================================
# StateSpaceModel Helper Functions
# ============================================

"""Store fitted parameters from NamedTuple to internal storage."""
function _ssm_store_theta!(model::StateSpaceModel{T}, theta::NamedTuple) where {T}
    for (i, prm) in enumerate(model.spec.params)
        model.theta_values[i] = T(getproperty(theta, prm.name))
    end
    model.theta_fitted = true
    return nothing
end

"""Reconstruct NamedTuple from internal parameter storage."""
function _ssm_get_theta_namedtuple(model::StateSpaceModel)
    model.theta_fitted || throw(ArgumentError("Parameters not fitted"))
    names = Tuple(prm.name for prm in model.spec.params)
    return NamedTuple{names}(Tuple(model.theta_values))
end

"""Copy KalmanFilterResult to model storage."""
function _ssm_store_filter_results!(model::StateSpaceModel, filt)
    copyto!(model.at, filt.at)
    copyto!(model.Pt, filt.Pt)
    copyto!(model.att, filt.att)
    copyto!(model.Ptt, filt.Ptt)
    copyto!(model.vt, filt.vt)
    copyto!(model.Ft, filt.Ft)
    copyto!(model.Kt, filt.Kt)
    copyto!(model.missing_mask, filt.missing_mask)
    model.filter_valid = true
    return nothing
end

"""Copy filter results from KalmanWorkspace to model."""
function _ssm_copy_from_workspace!(model::StateSpaceModel, kf_ws::KalmanWorkspace)
    copyto!(model.at, kf_ws.at)
    copyto!(model.Pt, kf_ws.Pt)
    copyto!(model.att, kf_ws.att)
    copyto!(model.Ptt, kf_ws.Ptt)
    copyto!(model.vt, kf_ws.vt)
    copyto!(model.Ft, kf_ws.Ft)
    copyto!(model.Kt, kf_ws.Kt)
    # Missing mask from workspace
    for t = 1:model.n_times
        model.missing_mask[t] = kf_ws.missing_mask[t]
    end
    model.filter_valid = true
    return nothing
end

# ============================================
# StateSpaceModel Status Accessors
# ============================================

"""Check if model has been fitted."""
isfitted(model::StateSpaceModel) = model.fitted

"""Check if estimation converged. Throws if not fitted."""
function isconverged(model::StateSpaceModel)
    model.fitted || throw(
        ArgumentError(
            "Model not fitted. Call fit!(MLE(), model, y) or fit!(EM(), model, y) first.",
        ),
    )
    return model.converged
end

"""Return log-likelihood at fitted parameters. Throws if not fitted."""
function loglikelihood(model::StateSpaceModel)
    model.fitted || throw(ArgumentError("Model not fitted."))
    return model.loglik
end

"""Return number of iterations. Throws if not fitted."""
function niterations(model::StateSpaceModel)
    model.fitted || throw(ArgumentError("Model not fitted."))
    return model.iterations
end

"""Return fitted parameters as NamedTuple. Throws if not fitted."""
function parameters(model::StateSpaceModel)
    model.fitted || throw(ArgumentError("Model not fitted."))
    return _ssm_get_theta_namedtuple(model)
end

# ============================================
# StateSpaceModel Filter Accessors
# ============================================

"""Return filtered states E[αₜ|y₁:ₜ] (m × n). Throws if not fitted."""
function filtered_states(model::StateSpaceModel)
    model.filter_valid || throw(ArgumentError("Filter not run. Call fit! first."))
    return model.att
end

"""Return filtered state covariances Var[αₜ|y₁:ₜ] (m × m × n). Throws if not fitted."""
function filtered_states_cov(model::StateSpaceModel)
    model.filter_valid || throw(ArgumentError("Filter not run."))
    return model.Ptt
end

"""Return predicted states E[αₜ|y₁:ₜ₋₁] (m × n). Throws if not fitted."""
function predicted_states(model::StateSpaceModel)
    model.filter_valid || throw(ArgumentError("Filter not run."))
    return model.at
end

"""Return predicted state covariances Var[αₜ|y₁:ₜ₋₁] (m × m × n). Throws if not fitted."""
function predicted_states_cov(model::StateSpaceModel)
    model.filter_valid || throw(ArgumentError("Filter not run."))
    return model.Pt
end

"""Return prediction errors / innovations yₜ - E[yₜ|y₁:ₜ₋₁] (p × n). Throws if not fitted."""
function prediction_errors(model::StateSpaceModel)
    model.filter_valid || throw(ArgumentError("Filter not run."))
    return model.vt
end

"""Return innovation covariances Var[yₜ|y₁:ₜ₋₁] (p × p × n). Throws if not fitted."""
function prediction_errors_cov(model::StateSpaceModel)
    model.filter_valid || throw(ArgumentError("Filter not run."))
    return model.Ft
end

# ============================================
# StateSpaceModel Smoother (Lazy Computation)
# ============================================

"""Compute smoother and cache results."""
function _ssm_compute_smoother!(model::StateSpaceModel)
    model.filter_valid || throw(ArgumentError("Filter not run."))

    # Get system matrices at fitted parameters
    theta_nt = _ssm_get_theta_namedtuple(model)
    kfparms = Siphon.build_kfparms(model.spec, theta_nt)

    # Run smoother using stored filter results
    alpha, V =
        Siphon.kalman_smoother(kfparms.Z, kfparms.T, model.at, model.Pt, model.vt, model.Ft)

    # Cache results
    copyto!(model.smoothed_alpha, alpha)
    copyto!(model.smoothed_V, V)
    model.smoother_computed = true
    return nothing
end

"""
    smoothed_states(model::StateSpaceModel)

Return smoothed states E[αₜ|y₁:ₙ] (m × n).

Computes smoother on first call and caches the result.
"""
function smoothed_states(model::StateSpaceModel)
    model.filter_valid || throw(ArgumentError("Filter not run. Call fit! first."))
    if !model.smoother_computed
        _ssm_compute_smoother!(model)
    end
    return model.smoothed_alpha
end

"""
    smoothed_states_cov(model::StateSpaceModel)

Return smoothed state covariances Var[αₜ|y₁:ₙ] (m × m × n).

Computes smoother on first call and caches the result.
"""
function smoothed_states_cov(model::StateSpaceModel)
    model.filter_valid || throw(ArgumentError("Filter not run."))
    if !model.smoother_computed
        _ssm_compute_smoother!(model)
    end
    return model.smoothed_V
end

# ============================================
# StateSpaceModel Matrix Accessors
# ============================================

"""
    obs_matrix(model::StateSpaceModel)

Return observation matrix Z (p × m) at fitted parameters.

Throws if model is not fitted.
"""
function obs_matrix(model::StateSpaceModel)
    model.fitted || throw(ArgumentError("Model not fitted. Call fit! first."))
    theta_nt = _ssm_get_theta_namedtuple(model)
    kfparms = Siphon.build_kfparms(model.spec, theta_nt)
    return kfparms.Z
end

"""
    obs_cov(model::StateSpaceModel)

Return observation noise covariance H (p × p) at fitted parameters.

Throws if model is not fitted.
"""
function obs_cov(model::StateSpaceModel)
    model.fitted || throw(ArgumentError("Model not fitted. Call fit! first."))
    theta_nt = _ssm_get_theta_namedtuple(model)
    kfparms = Siphon.build_kfparms(model.spec, theta_nt)
    return kfparms.H
end

"""
    transition_matrix(model::StateSpaceModel)

Return state transition matrix T (m × m) at fitted parameters.

Throws if model is not fitted.
"""
function transition_matrix(model::StateSpaceModel)
    model.fitted || throw(ArgumentError("Model not fitted. Call fit! first."))
    theta_nt = _ssm_get_theta_namedtuple(model)
    kfparms = Siphon.build_kfparms(model.spec, theta_nt)
    return kfparms.T
end

"""
    selection_matrix(model::StateSpaceModel)

Return state noise selection matrix R (m × r) at fitted parameters.

Throws if model is not fitted.
"""
function selection_matrix(model::StateSpaceModel)
    model.fitted || throw(ArgumentError("Model not fitted. Call fit! first."))
    theta_nt = _ssm_get_theta_namedtuple(model)
    kfparms = Siphon.build_kfparms(model.spec, theta_nt)
    return kfparms.R
end

"""
    state_cov(model::StateSpaceModel)

Return state noise covariance Q (r × r) at fitted parameters.

Throws if model is not fitted.
"""
function state_cov(model::StateSpaceModel)
    model.fitted || throw(ArgumentError("Model not fitted. Call fit! first."))
    theta_nt = _ssm_get_theta_namedtuple(model)
    kfparms = Siphon.build_kfparms(model.spec, theta_nt)
    return kfparms.Q
end

"""
    system_matrices(model::StateSpaceModel)

Return all system matrices as a NamedTuple (Z, H, T, R, Q) at fitted parameters.

This is more efficient than calling individual accessors when multiple matrices are needed.

# Example
```julia
mats = system_matrices(model)
mats.Z  # observation matrix
mats.H  # observation covariance
mats.T  # transition matrix
mats.R  # selection matrix
mats.Q  # state covariance
```

Throws if model is not fitted.
"""
function system_matrices(model::StateSpaceModel)
    model.fitted || throw(ArgumentError("Model not fitted. Call fit! first."))
    theta_nt = _ssm_get_theta_namedtuple(model)
    kfparms = Siphon.build_kfparms(model.spec, theta_nt)
    return (Z = kfparms.Z, H = kfparms.H, T = kfparms.T, R = kfparms.R, Q = kfparms.Q)
end

# ============================================
# StateSpaceModel fit!(MLE(), ...)
# ============================================

"""
    fit!(::MLE, model::StateSpaceModel, y::AbstractMatrix; kwargs...)

Fit state-space model using Maximum Likelihood Estimation.

Uses the pure AD-compatible Kalman filter via Optimization.jl.
Automatically uses StaticArrays for small models (dims ≤ 13).

# Arguments
- `MLE()`: Estimation method selector
- `model`: StateSpaceModel to fit (mutated in-place)
- `y`: Observations (p × n matrix), missing values as NaN

# Keyword Arguments
- `method`: Optimization algorithm (default: LBFGS from Optim.jl)
- `verbose`: Print optimization progress (default: false)
- Additional kwargs passed to optimize_ssm

# Returns
The fitted `model` (same object, mutated)

# Example
```julia
spec = local_level(var_obs=:free, var_level=:free)
model = StateSpaceModel(spec, 100)
fit!(MLE(), model, randn(1, 100))
parameters(model)  # (var_obs=..., var_level=...)
```
"""
function fit!(
    ::MLE,
    model::StateSpaceModel{T},
    y::AbstractMatrix;
    verbose::Bool = false,
    kwargs...,
) where {T}

    p_obs, n = size(y)
    @assert n == model.n_times "Observation length $n != model.n_times $(model.n_times)"
    @assert p_obs == model.spec.n_obs "Observation dim $p_obs != spec.n_obs $(model.spec.n_obs)"

    # Use existing optimize_ssm infrastructure
    use_static = max(model.spec.n_states, model.spec.n_obs) <= STATIC_THRESHOLD

    result = Siphon.optimize_ssm(model.spec, y; use_static = use_static, kwargs...)

    # Store fitted parameters
    _ssm_store_theta!(model, result.θ)

    # Run full filter to get filter results
    theta_nt = _ssm_get_theta_namedtuple(model)
    ss = Siphon.build_linear_state_space(model.spec, theta_nt, y; use_static = use_static)
    filt = Siphon.kalman_filter(ss.p, y, ss.a1, ss.P1)

    # Copy filter results to model storage
    _ssm_store_filter_results!(model, filt)

    # Update model state
    model.loglik = result.loglik
    model.fitted = true
    model.converged = result.converged
    model.iterations = 0  # MLE doesn't have iteration count in same sense
    model.backend = :mle
    model.smoother_computed = false

    if verbose
        println("MLE fitting complete:")
        println("  Log-likelihood: ", round(model.loglik, digits = 4))
        println("  Converged: ", model.converged)
    end

    return model
end

# ============================================
# StateSpaceModel fit!(EM(), ...)
# ============================================

"""
    fit!(::EM, model::StateSpaceModel, y::AbstractMatrix; kwargs...)

Fit state-space model using EM algorithm.

Automatically selects backend based on dimensions:
- If max(n_states, n_obs) > 13: Use in-place KalmanWorkspace (zero allocations)
- Otherwise: Use pure implementation

# Arguments
- `EM()`: Estimation method selector
- `model`: StateSpaceModel to fit (mutated in-place)
- `y`: Observations (p × n matrix), missing values as NaN

# Keyword Arguments
- `maxiter::Int=500`: Maximum EM iterations
- `tol::Real=1e-6`: Convergence tolerance (relative change in log-likelihood)
- `verbose::Bool=false`: Print progress

# Returns
The fitted `model` (same object, mutated)

# Note
EM estimation requires that free parameters are marked in the SSMSpec.
For general state-space models, EM typically updates H and Q (covariances).

# Example
```julia
spec = local_level(var_obs=:free, var_level=:free)
model = StateSpaceModel(spec, 100)
fit!(EM(), model, randn(1, 100); maxiter=200, verbose=true)
```
"""
function fit!(
    ::EM,
    model::StateSpaceModel{T},
    y::AbstractMatrix;
    maxiter::Int = 500,
    tol::Real = 1e-6,
    verbose::Bool = false,
) where {T}

    p_obs, n = size(y)
    @assert n == model.n_times "Observation length $n != model.n_times $(model.n_times)"
    @assert p_obs == model.spec.n_obs "Observation dim $p_obs != spec.n_obs $(model.spec.n_obs)"

    if model.kf_workspace_allocated
        # Large model: use in-place EM
        _ssm_fit_em_inplace!(model, y; maxiter = maxiter, tol = tol, verbose = verbose)
        model.backend = :em_inplace
    else
        # Small model: use pure/static EM (fall back to MLE for now)
        # TODO: Implement proper static EM
        _ssm_fit_em_static!(model, y; maxiter = maxiter, tol = tol, verbose = verbose)
        model.backend = :em_static
    end

    model.fitted = true
    model.filter_valid = true
    model.smoother_computed = false

    return model
end

"""In-place EM backend for large models."""
function _ssm_fit_em_inplace!(
    model::StateSpaceModel{T},
    y::AbstractMatrix;
    maxiter::Int,
    tol::Real,
    verbose::Bool,
) where {T}

    # Get pre-allocated workspaces
    ws = model.workspaces_ref[]::StateSpaceModelWorkspaces{T}
    kf_ws = ws.kf_ws
    em_ws = ws.em_ws

    # Initialize parameters from spec
    theta_init = [prm.init for prm in model.spec.params]
    names = Tuple(prm.name for prm in model.spec.params)
    theta_nt = NamedTuple{names}(Tuple(theta_init))

    # Set initial parameters in workspace
    kfparms = Siphon.build_kfparms(model.spec, theta_nt)
    a1_init, P1_init = Siphon.build_initial_state(model.spec, theta_nt)
    _set_workspace_params!(kf_ws, kfparms, a1_init, P1_init)

    # Run EM using existing em_estimate!
    em_result =
        em_estimate!(kf_ws, em_ws, y; maxiter = maxiter, tol = tol, verbose = verbose)

    # Extract fitted parameters from workspace back to spec format
    # For general SSM, we extract from Z, H, T, R, Q matrices
    _ssm_extract_params_from_workspace!(model, kf_ws)

    # Copy filter results from workspace
    _ssm_copy_from_workspace!(model, kf_ws)

    # Update model state
    model.loglik = kf_ws.loglik
    model.converged = em_result.converged
    model.iterations = em_result.iterations

    return nothing
end

"""Set workspace parameters from KFParms."""
function _set_workspace_params!(kf_ws::KalmanWorkspace{T}, kfparms, a1, P1) where {T}
    copyto!(kf_ws.Z, kfparms.Z)
    copyto!(kf_ws.H, kfparms.H)
    copyto!(kf_ws.Tmat, kfparms.T)
    copyto!(kf_ws.R, kfparms.R)
    copyto!(kf_ws.Q, kfparms.Q)
    copyto!(kf_ws.a1, a1)
    copyto!(kf_ws.P1, P1)
    _update_RQR!(kf_ws)
    return nothing
end

"""Extract fitted parameters from workspace matrices back to model.theta_values."""
function _ssm_extract_params_from_workspace!(
    model::StateSpaceModel{T},
    kf_ws::KalmanWorkspace{T},
) where {T}
    # This requires mapping SSMSpec parameter locations back to matrix elements
    # For each parameter in spec.params, find its location in Z, H, T, R, Q
    # and extract the current value from the workspace

    spec = model.spec

    for (i, prm) in enumerate(spec.params)
        # Search in each matrix spec for this parameter
        val = _find_param_value(prm.name, spec, kf_ws)
        model.theta_values[i] = val
    end
    model.theta_fitted = true
    return nothing
end

"""Find parameter value by searching matrix specs."""
function _find_param_value(name::Symbol, spec::SSMSpec, kf_ws::KalmanWorkspace{T}) where {T}
    # Check Z
    for ((row, col), elem) in spec.Z.elements
        if elem isa ParameterRef && elem.name == name
            return kf_ws.Z[row, col]
        end
    end
    # Check H
    for ((row, col), elem) in spec.H.elements
        if elem isa ParameterRef && elem.name == name
            return kf_ws.H[row, col]
        end
    end
    # Check T
    for ((row, col), elem) in spec.T.elements
        if elem isa ParameterRef && elem.name == name
            return kf_ws.Tmat[row, col]
        end
    end
    # Check R
    for ((row, col), elem) in spec.R.elements
        if elem isa ParameterRef && elem.name == name
            return kf_ws.R[row, col]
        end
    end
    # Check Q
    for ((row, col), elem) in spec.Q.elements
        if elem isa ParameterRef && elem.name == name
            return kf_ws.Q[row, col]
        end
    end

    # If not found in matrices, return current value (shouldn't happen for valid specs)
    @warn "Parameter $name not found in matrix specs, using initial value"
    idx = findfirst(p -> p.name == name, spec.params)
    return idx !== nothing ? spec.params[idx].init : zero(T)
end

# ============================================
# Static EM Helper Functions
# ============================================

"""
    StaticEMSuffStats{T}

Sufficient statistics for static EM algorithm computed from smoothed states.
Allocated fresh each iteration (acceptable for small static models).

# Fields
- `S_yy`: p×p sum of y_t * y_t' (observation outer products)
- `S_ya`: p×m sum of y_t * α̂_t' (observation-state cross products)
- `S_aa`: m×m sum of (α̂_t α̂_t' + V̂_t) for all t (state second moments)
- `S_aa_prev`: m×m sum of (α̂_{t-1} α̂_{t-1}' + V̂_{t-1}) for t=2:n (lagged state moments)
- `S_10`: m×m sum of (α̂_t α̂_{t-1}' + P_{t,t-1|n}) for t=2:n (cross-lag moments)
- `S_11`: m×m sum of (α̂_t α̂_t' + V̂_t) for t=2:n (state moments excluding t=1)
- `n_obs`: number of non-missing observations
"""
struct StaticEMSuffStats{T<:Real}
    S_yy::Matrix{T}
    S_ya::Matrix{T}
    S_aa::Matrix{T}
    S_aa_prev::Matrix{T}
    S_10::Matrix{T}
    S_11::Matrix{T}
    n_obs::Int
end

"""
    _compute_static_sufficient_stats(y, alpha, V, P_crosslag, missing_mask)

Compute sufficient statistics from Kalman smoother output for the M-step.

Uses smoothed states α̂_t = E[α_t | y_{1:n}], smoothed covariances V̂_t = Var[α_t | y_{1:n}],
and cross-lag covariances P_{t,t-1|n} = Cov[α_t, α_{t-1} | y_{1:n}].

# Arguments
- `y`: Observations (p × n)
- `alpha`: Smoothed states (m × n)
- `V`: Smoothed covariances (m × m × n)
- `P_crosslag`: Cross-lag covariances (m × m × (n-1)), where P_crosslag[:,:,t] = Cov[α_{t+1}, α_t | Y]
- `missing_mask`: BitVector indicating missing observations

# Returns
`StaticEMSuffStats` containing all sufficient statistics needed for M-step.

# Notes
The formulas follow Shumway & Stoffer (2017), Chapter 6.
Missing observations are excluded from observation-related statistics (S_yy, S_ya)
but all time points contribute to state statistics (S_aa, S_10, etc.).
"""
function _compute_static_sufficient_stats(
    y::AbstractMatrix,
    alpha::AbstractMatrix,
    V::AbstractArray,
    P_crosslag::AbstractArray,
    missing_mask::BitVector,
)
    T = promote_type(eltype(y), eltype(alpha))
    p, n = size(y)
    m = size(alpha, 1)

    # Initialize sufficient statistics matrices
    S_yy = zeros(T, p, p)
    S_ya = zeros(T, p, m)
    S_aa = zeros(T, m, m)
    S_aa_prev = zeros(T, m, m)
    S_10 = zeros(T, m, m)
    S_11 = zeros(T, m, m)
    n_obs = 0

    @inbounds for t = 1:n
        alpha_t = view(alpha, :, t)
        V_t = view(V,:,:,t)

        # Observation-related statistics: only for non-missing observations
        if !missing_mask[t]
            n_obs += 1
            y_t = view(y, :, t)

            # S_yy += y_t * y_t'
            for j = 1:p, i = 1:p
                S_yy[i, j] += y_t[i] * y_t[j]
            end

            # S_ya += y_t * α̂_t'
            for j = 1:m, i = 1:p
                S_ya[i, j] += y_t[i] * alpha_t[j]
            end
        end

        # State statistics: all time points (for S_aa)
        # S_aa += E[α_t α_t' | Y] = V̂_t + α̂_t α̂_t'
        for j = 1:m, i = 1:m
            S_aa[i, j] += V_t[i, j] + alpha_t[i] * alpha_t[j]
        end

        # Cross-lag and lagged statistics: for t >= 2
        if t >= 2
            alpha_tm1 = view(alpha, :, t-1)
            V_tm1 = view(V,:,:,(t-1))
            # P_crosslag[:,:,t-1] contains Cov[α_t, α_{t-1} | Y]
            P_cross = view(P_crosslag,:,:,(t-1))

            # S_aa_prev += E[α_{t-1} α_{t-1}' | Y] for t=2:n
            for j = 1:m, i = 1:m
                S_aa_prev[i, j] += V_tm1[i, j] + alpha_tm1[i] * alpha_tm1[j]
            end

            # S_10 += E[α_t α_{t-1}' | Y] = P_{t,t-1|n} + α̂_t α̂_{t-1}'
            for j = 1:m, i = 1:m
                S_10[i, j] += P_cross[i, j] + alpha_t[i] * alpha_tm1[j]
            end

            # S_11 += E[α_t α_t' | Y] for t=2:n (needed for Q update)
            for j = 1:m, i = 1:m
                S_11[i, j] += V_t[i, j] + alpha_t[i] * alpha_t[j]
            end
        end
    end

    return StaticEMSuffStats(S_yy, S_ya, S_aa, S_aa_prev, S_10, S_11, n_obs)
end

"""
    _mstep_static_matrices(Z, T_mat, R, stats, n; regularize=1e-10)

Compute unconstrained M-step updates for all system matrices.

# Arguments
- `Z`: Current observation matrix (p × m)
- `T_mat`: Current transition matrix (m × m)
- `R`: Selection matrix (m × r)
- `stats`: `StaticEMSuffStats` from E-step
- `n`: Total number of time periods

# Keyword Arguments
- `regularize`: Small value added to diagonal for numerical stability (default: 1e-10)

# Returns
Tuple `(Z_new, T_new, H_new, Q_new)` with updated matrices.

# M-Step Formulas (Shumway & Stoffer, 2017)

The closed-form M-step updates are:

- Z_new = S_ya * S_aa^{-1}
- T_new = S_10 * S_aa_prev^{-1}
- H_new = (1/n_obs) * (S_yy - Z * S_ya')
- Q_new = (1/(n-1)) * R^† * (S_11 - T*S_10' - S_10*T' + T*S_aa_prev*T') * R^†'

where R^† is the Moore-Penrose pseudoinverse of R.

# Numerical Stability

- S_aa and S_aa_prev are regularized by adding `regularize * I` before inversion.
  This is a standard numerical safeguard for near-singular matrices.
- H_new is computed using current Z (not Z_new) to maintain EM monotonicity.
- Both H_new and Q_new are symmetrized after computation.
"""
function _mstep_static_matrices(
    Z::AbstractMatrix,
    T_mat::AbstractMatrix,
    R::AbstractMatrix,
    stats::StaticEMSuffStats{T},
    n::Int;
    regularize::Real = 1e-10,
) where {T}
    p = size(Z, 1)
    m = size(T_mat, 1)
    r = size(R, 2)
    n_obs = stats.n_obs

    # Regularize S_aa for numerical stability
    # NOTE: Adding small diagonal improves conditioning for near-singular cases.
    # Value of 1e-10 is chosen as typical machine-epsilon-scale safeguard.
    S_aa_reg = Matrix(stats.S_aa)
    for i = 1:m
        S_aa_reg[i, i] += T(regularize)
    end

    # Regularize S_aa_prev similarly
    S_aa_prev_reg = Matrix(stats.S_aa_prev)
    for i = 1:m
        S_aa_prev_reg[i, i] += T(regularize)
    end

    # Z_new = S_ya * S_aa^{-1}
    Z_new = stats.S_ya / S_aa_reg
    # Check for NaN/Inf (can happen with degenerate data)
    if any(!isfinite, Z_new)
        @warn "Z_new contains NaN/Inf, keeping current Z"
        Z_new = Matrix(Z)
    end

    # T_new = S_10 * S_aa_prev^{-1}
    T_new = stats.S_10 / S_aa_prev_reg
    # Check for NaN/Inf (can happen with degenerate data)
    if any(!isfinite, T_new)
        @warn "T_new contains NaN/Inf, keeping current T"
        T_new = Matrix(T_mat)
    end

    # H_new = (1/n_obs) * (S_yy - Z * S_ya')
    # NOTE: We use current Z (not Z_new) to maintain EM monotonicity per Shumway & Stoffer.
    # The formula H = (1/n)(S_yy - Z*S_ya') is guaranteed PSD when using current Z.
    H_new = (stats.S_yy - Z * stats.S_ya') / n_obs
    # Ensure symmetry (numerical safeguard)
    H_new = (H_new + H_new') / 2
    # Check for NaN/Inf (can happen with degenerate data or all-missing variables)
    if any(!isfinite, H_new)
        @warn "H_new contains NaN/Inf, using fallback identity scaling"
        H_new = Matrix{T}(I(p) * T(0.01))
    end
    # Ensure positive diagonal (numerical safeguard for tiny eigenvalues)
    for i = 1:p
        H_new[i, i] = max(H_new[i, i], T(1e-10))
    end

    # Q_new computation:
    # Q_new = (1/(n-1)) * R^† * Σ_η * R^†'
    # where Σ_η = S_11 - T*S_10' - S_10*T' + T*S_aa_prev*T'
    #
    # NOTE: This formula is PSD by construction when derived from conditional expectations.
    # The R^† projection handles cases where R is not square.

    # Compute Σ_η = S_11 - T*S_10' - S_10*T' + T*S_aa_prev*T'
    Sigma_eta = Matrix(stats.S_11)
    TS_10t = T_mat * stats.S_10'
    T_S_aa_prev_Tt = T_mat * stats.S_aa_prev * T_mat'

    for j = 1:m, i = 1:m
        Sigma_eta[i, j] =
            Sigma_eta[i, j] - TS_10t[i, j] - TS_10t[j, i] + T_S_aa_prev_Tt[i, j]
    end

    # Scale by 1/(n-1)
    scale = one(T) / (n - 1)
    Sigma_eta .*= scale

    # Apply R projection: Q = R^† * Σ_η * R^†'
    if r == m && R ≈ I(m)
        # R = I case: Q = Σ_η directly
        Q_new = Sigma_eta
    else
        # General case: Q = R^† * Σ_η * R^†'
        R_pinv = pinv(Matrix(R))
        Q_new = R_pinv * Sigma_eta * R_pinv'
    end

    # Ensure symmetry
    Q_new = (Q_new + Q_new') / 2

    # Check for NaN/Inf before eigendecomposition (can happen with degenerate data)
    if any(!isfinite, Q_new)
        # Fall back to identity-scaled matrix if Q_new is degenerate
        @warn "Q_new contains NaN/Inf, using fallback identity scaling"
        Q_new = Matrix{T}(I(r) * T(0.01))
    else
        # Ensure positive semi-definiteness via eigenvalue projection
        # NOTE: In theory, Q_new should be PSD. If numerical errors cause small negative
        # eigenvalues, we project to nearest PSD matrix by zeroing negative eigenvalues.
        eig = eigen(Symmetric(Q_new))
        if any(eig.values .< 0)
            # Project to PSD: set negative eigenvalues to zero
            # This is the standard projection to the PSD cone.
            eig_vals_proj = max.(eig.values, T(0))
            Q_new = eig.vectors * Diagonal(eig_vals_proj) * eig.vectors'
            Q_new = (Q_new + Q_new') / 2
        end
    end

    # Ensure positive diagonal (numerical safeguard)
    for i = 1:r
        Q_new[i, i] = max(Q_new[i, i], T(1e-10))
    end

    return (Z_new, T_new, H_new, Q_new)
end

"""
    _extract_params_from_matrices_static(spec, Z_new, T_new, H_new, Q_new)

Extract parameter values from M-step updated matrices based on SSMSpec structure.

Maps the updated system matrices back to the parameters defined in `spec.params`
by finding which matrix elements correspond to each parameter.

# Arguments
- `spec`: SSMSpec containing parameter definitions and matrix structures
- `Z_new`, `T_new`, `H_new`, `Q_new`: Updated matrices from M-step

# Returns
Dict{Symbol, Float64} mapping parameter names to their updated values.

# Notes
- For parameters referenced via `ParameterRef` in matrix specs, extracts directly.
- For `CovMatrixExpr` (H or Q with D*Corr*D parameterization), extracts σ from sqrt(diag).
- Parameters not found in any matrix spec retain their current values.
"""
function _extract_params_from_matrices_static(
    spec::SSMSpec,
    Z_new::AbstractMatrix,
    T_new::AbstractMatrix,
    H_new::AbstractMatrix,
    Q_new::AbstractMatrix,
)
    params = Dict{Symbol,Float64}()

    # Extract from Z matrix spec
    _extract_from_matrix_spec_static!(params, spec.Z, Z_new)

    # Extract from T matrix spec
    _extract_from_matrix_spec_static!(params, spec.T, T_new)

    # Extract from H matrix spec (or CovMatrixExpr)
    if haskey(spec.matrix_exprs, :H)
        _extract_from_cov_expr_static!(params, spec.matrix_exprs[:H], H_new)
    else
        _extract_from_matrix_spec_static!(params, spec.H, H_new)
    end

    # Extract from Q matrix spec (or CovMatrixExpr)
    if haskey(spec.matrix_exprs, :Q)
        _extract_from_cov_expr_static!(params, spec.matrix_exprs[:Q], Q_new)
    else
        _extract_from_matrix_spec_static!(params, spec.Q, Q_new)
    end

    # R is typically fixed in state-space models (selection matrix).
    # If R has free parameters, they are not updated by standard EM.
    # Skip R extraction entirely - it remains at its initial values.

    return params
end

"""
    _extract_from_matrix_spec_static!(params, mat_spec, mat_new)

Extract parameters from a simple matrix specification.

For each `(row, col) => ParameterRef(name)` in `mat_spec.elements`,
sets `params[name] = mat_new[row, col]`.
"""
function _extract_from_matrix_spec_static!(
    params::Dict{Symbol,Float64},
    mat_spec::SSMMatrixSpec,
    mat_new::AbstractMatrix,
)
    for ((row, col), elem) in mat_spec.elements
        if elem isa ParameterRef
            params[elem.name] = Float64(mat_new[row, col])
        end
    end
end

"""
    _extract_from_cov_expr_static!(params, expr, Sigma_new)

Extract parameters from a CovMatrixExpr (D*Corr*D decomposition).

Given the updated covariance matrix Σ_new, extracts:
- Standard deviation parameters: σ_i = sqrt(Σ_new[i,i])
- Correlation parameters: set to 0 (unconstrained EM doesn't update correlations cleanly)

# Notes
The inverse transform from Σ to correlation parameters is non-trivial and not
implemented here. Correlation parameters are left unchanged (effectively 0 for
initial uncorrelated case). This is a limitation of the current implementation.
"""
function _extract_from_cov_expr_static!(
    params::Dict{Symbol,Float64},
    expr::CovMatrixExpr,
    Sigma_new::AbstractMatrix,
)
    n = expr.n

    # Extract standard deviations from diagonal
    for (i, name) in enumerate(expr.σ_param_names)
        # σ_i = sqrt(Σ[i,i])
        params[name] = sqrt(max(Float64(Sigma_new[i, i]), 1e-10))
    end

    # Correlation parameters: leave at 0 (identity correlation)
    # NOTE: Proper inverse of corr_cholesky_factor transform is complex.
    # For unconstrained EM, we assume diagonal covariances or leave correlations unchanged.
    for name in expr.corr_param_names
        if !haskey(params, name)
            params[name] = 0.0
        end
    end
end

# ============================================
# Static EM Main Function
# ============================================

"""
    _ssm_fit_em_static!(model, y; maxiter, tol, verbose)

Static/pure EM backend for small state-space models.

This implements the EM algorithm using pure Kalman filter and smoother (without
pre-allocated workspaces), suitable for models where max(n_states, n_obs) ≤ 13.
The pure implementation uses StaticArrays internally for better performance on small models.

# Algorithm

1. **E-step**: Run Kalman filter and smoother with cross-covariances
2. **M-step**: Compute closed-form updates for Z, T, H, Q matrices
3. **Parameter extraction**: Map updated matrices back to SSMSpec parameters
4. **Convergence check**: Based on relative change in log-likelihood

# Limitations

**IMPORTANT**: This implements UNCONSTRAINED EM. Parameters with finite bounds in
SSMSpec are NOT respected during optimization—the algorithm assumes unconstrained
closed-form updates. A warning is issued if any updated parameter violates its bounds.
Proper constrained EM (via reparameterization or barrier methods) is planned for
future implementation.

# Arguments
- `model`: StateSpaceModel to fit
- `y`: Observations (p × n matrix)
- `maxiter`: Maximum EM iterations
- `tol`: Convergence tolerance for relative log-likelihood change
- `verbose`: Print iteration progress
"""
function _ssm_fit_em_static!(
    model::StateSpaceModel{T},
    y::AbstractMatrix;
    maxiter::Int,
    tol::Real,
    verbose::Bool,
) where {T}
    spec = model.spec
    n = size(y, 2)

    # Initialize from spec
    theta_curr = T[prm.init for prm in spec.params]
    names = Tuple(prm.name for prm in spec.params)

    loglik_prev = T(-Inf)
    converged = false
    iter = 0

    for iter_i = 1:maxiter
        iter = iter_i
        theta_nt = NamedTuple{names}(Tuple(theta_curr))

        # Build state-space model from current parameters
        ss = Siphon.build_linear_state_space(spec, theta_nt, y; use_static = true)

        # E-step: Run Kalman filter
        filt = Siphon.kalman_filter(ss.p, y, ss.a1, ss.P1)
        loglik_curr = filt.loglik

        # Check convergence based on relative log-likelihood change
        rel_change = abs(loglik_curr - loglik_prev) / (abs(loglik_prev) + T(1e-10))
        if rel_change < tol && iter_i > 1
            converged = true
            verbose && println(
                "EM converged at iteration $iter_i, loglik=$(round(loglik_curr, digits=4))",
            )
            break
        end
        loglik_prev = loglik_curr

        # E-step: Run smoother with cross-covariances for M-step
        smooth = Siphon.kalman_smoother(
            ss.p.Z,
            ss.p.T,
            filt.at,
            filt.Pt,
            filt.vt,
            filt.Ft;
            compute_crosscov = true,
        )

        # Compute sufficient statistics from smoother output
        stats = _compute_static_sufficient_stats(
            y,
            smooth.alpha,
            smooth.V,
            smooth.P_crosslag,
            filt.missing_mask,
        )

        # M-step: Compute unconstrained matrix updates
        Z_new, T_new, H_new, Q_new =
            _mstep_static_matrices(Matrix(ss.p.Z), Matrix(ss.p.T), Matrix(ss.p.R), stats, n)

        # Extract parameters from updated matrices
        params_dict = _extract_params_from_matrices_static(spec, Z_new, T_new, H_new, Q_new)

        # Update parameter vector (unconstrained - warn if bounds violated)
        for (i, prm) in enumerate(spec.params)
            if haskey(params_dict, prm.name)
                val = T(params_dict[prm.name])
                # Check bounds and warn (but don't clamp - unconstrained EM)
                if val < prm.lower || val > prm.upper
                    @warn "Parameter $(prm.name) = $(round(val, digits=6)) violates bounds " *
                          "[$(prm.lower), $(prm.upper)]. Unconstrained EM does not enforce bounds."
                end
                theta_curr[i] = val
            end
            # Parameters not in params_dict keep their current value
        end

        verbose &&
            iter_i % 10 == 0 &&
            println("EM iteration $iter_i: loglik=$(round(loglik_curr, digits=4))")
    end

    # Store final parameters
    theta_nt = NamedTuple{names}(Tuple(theta_curr))
    _ssm_store_theta!(model, theta_nt)

    # Run final filter to populate filter results
    ss = Siphon.build_linear_state_space(spec, theta_nt, y; use_static = true)
    filt = Siphon.kalman_filter(ss.p, y, ss.a1, ss.P1)
    _ssm_store_filter_results!(model, filt)

    model.loglik = filt.loglik
    model.converged = converged
    model.iterations = iter

    return nothing
end

# ============================================
# StateSpaceModel Forecasting
# ============================================

"""
    forecast(model::StateSpaceModel, h::Int)

Forecast h steps ahead from fitted model.

Uses the filtered state at the last observation and iterates forward.

# Arguments
- `model`: Fitted StateSpaceModel
- `h`: Forecast horizon

# Returns
NamedTuple with:
- `yhat`: Forecasted observations (p × h)
- `a`: Forecasted states (m × h)
- `P`: Forecasted state covariances (m × m × h)
- `F`: Forecasted observation covariances (p × p × h)

# Example
```julia
model = StateSpaceModel(local_level(), 100)
fit!(MLE(), model, y)
fc = forecast(model, 10)
fc.yhat  # 1 × 10 forecasts
```
"""
function forecast(model::StateSpaceModel{T}, h::Int) where {T}
    model.fitted || throw(ArgumentError("Model not fitted."))
    model.filter_valid || throw(ArgumentError("Filter not run."))

    # Get system matrices at fitted parameters
    theta_nt = _ssm_get_theta_namedtuple(model)
    kfparms = Siphon.build_kfparms(model.spec, theta_nt)

    # Get final filtered state
    n = model.n_times
    a_final = model.att[:, n]
    P_final = model.Ptt[:, :, n]

    # Forecast forward
    _ssm_forecast_from_state(kfparms, a_final, P_final, h)
end

"""Internal forecast helper."""
function _ssm_forecast_from_state(
    p::KFParms,
    a::AbstractVector{T},
    P::AbstractMatrix{T},
    h::Int,
) where {T}
    m = length(a)
    obs_dim = size(p.Z, 1)

    yhat = Matrix{T}(undef, obs_dim, h)
    a_fc = Matrix{T}(undef, m, h)
    P_fc = Array{T,3}(undef, m, m, h)
    F_fc = Array{T,3}(undef, obs_dim, obs_dim, h)

    # Compute RQR' once
    RQR = p.R * p.Q * p.R'

    # First step: predict from final filtered state
    a_curr = p.T * a
    P_curr = p.T * P * p.T' + RQR

    for j = 1:h
        a_fc[:, j] = a_curr
        P_fc[:, :, j] = P_curr
        yhat[:, j] = p.Z * a_curr
        F_fc[:, :, j] = p.Z * P_curr * p.Z' + p.H

        # Propagate for next step
        if j < h
            a_curr = p.T * a_curr
            P_curr = p.T * P_curr * p.T' + RQR
        end
    end

    return (yhat = yhat, a = a_fc, P = P_fc, F = F_fc)
end

# ============================================
# Dynamic Factor Model
# ============================================

"""
    DynamicFactorModelSpec

Specification for a dynamic factor model with optional dynamic loadings and AR errors.

Model structure (following Stock & Watson notation):
    Xₜ = λ(L) fₜ + eₜ
    fₜ = Ψ(L) fₜ₋₁ + ηₜ
    eᵢₜ = δ(L) eᵢ,ₜ₋₁ + vᵢₜ

where:
    λ(L) = λ₀ + λ₁L + ... + λₚLᵖ  (dynamic factor loadings)
    Ψ(L) = Ψ₁L + ... + ΨᵧLᵧ       (factor VAR dynamics)
    δ(L) = δ₁L + ... + δᵣLʳ       (AR idiosyncratic errors, common across i)

State-space representation:
    State vector: αₜ = [fₜ; fₜ₋₁; ...; fₜ₋ₛ₊₁; eₜ; eₜ₋₁; ...; eₜ₋ᵣ₊₁]

    where s = max(q, p) to accommodate both factor dynamics and dynamic loadings

Dimensions:
    N = number of observables
    k = number of factors
    p = lags in λ(L) (dynamic loadings)
    q = lags in Ψ(L) (factor VAR)
    r = lags in δ(L) (AR errors)

    s = max(q, p+1) for factor block
    m_f = k * s (factor state dimension)
    m_e = N * r (error state dimension)
    m = m_f + m_e (total state dimension)
"""
struct DynamicFactorModelSpec
    n_obs::Int           # N: number of observables
    n_factors::Int       # k: number of factors
    loading_lags::Int    # p: lags in λ(L), p=0 means static loadings
    factor_lags::Int     # q: lags in factor VAR Ψ(L)
    error_lags::Int      # r: lags in AR errors δ(L), r=0 means white noise
end

"""
    DynamicFactorModelWorkspace{T}

Internal workspace for DFM estimation with AR errors.
Stores additional parameters and sufficient statistics for δ estimation.
"""
mutable struct DynamicFactorModelWorkspace{T<:Real}
    # Model specification
    spec::DynamicFactorModelSpec

    # Dimensions
    n_obs::Int           # N
    n_factors::Int       # k
    loading_lags::Int    # p
    factor_lags::Int     # q
    error_lags::Int      # r
    n_times::Int         # n

    # Derived dimensions
    factor_state_dim::Int  # m_f = k * s where s = max(q, p+1)
    error_state_dim::Int   # m_e = N * r
    total_state_dim::Int   # m = m_f + m_e

    # Parameter storage (separate from KalmanWorkspace for clarity)
    # Factor loadings: Λ[j] is N × k for lag j, j = 0, 1, ..., p
    Λ::Vector{Matrix{T}}

    # Factor VAR: Φ[j] is k × k for lag j, j = 1, ..., q
    Φ::Vector{Matrix{T}}

    # Factor innovation covariance
    Σ_η::Matrix{T}  # k × k

    # AR coefficients for errors: δ[j] is scalar for lag j, j = 1, ..., r
    δ::Vector{T}

    # Idiosyncratic innovation variances (diagonal)
    σ²_v::Vector{T}  # N × 1

    # Sufficient statistics for AR error estimation
    S_ee::Matrix{T}    # Σ E[eₜ eₜ' | Y] (N × N) - but we only need diagonal
    S_ee_lag::Array{T,3}  # Σ E[eₜ eₜ₋ⱼ' | Y] for j=1:r (N × N × r)

    # Scratch space
    tmp_nn::Matrix{T}   # N × N
    tmp_kk::Matrix{T}   # k × k
end

"""
    _setup_dfm(spec::DynamicFactorModelSpec, n_times::Int, ::Type{T}=Float64)

Create workspaces for full dynamic factor model estimation.

Returns (kf_ws, em_ws, dfm_ws) where:
- kf_ws: KalmanWorkspace for filter/smoother
- em_ws: EMWorkspace for basic EM sufficient statistics
- dfm_ws: DynamicFactorModelWorkspace for DFM-specific parameters and updates
"""
function _setup_dfm(
    spec::DynamicFactorModelSpec,
    n_times::Int,
    ::Type{T} = Float64,
) where {T}
    N = spec.n_obs
    k = spec.n_factors
    p = spec.loading_lags
    q = spec.factor_lags
    r = spec.error_lags

    # Factor state needs to hold enough lags for both VAR and dynamic loadings
    # fₜ, fₜ₋₁, ..., fₜ₋ₛ₊₁ where s = max(q, p+1)
    # (p+1 because λ(L)fₜ = λ₀fₜ + λ₁fₜ₋₁ + ... + λₚfₜ₋ₚ needs fₜ₋ₚ)
    s = max(q, p + 1)
    s = max(s, 1)  # At least 1

    m_f = k * s           # Factor state dimension
    m_e = N * r           # Error state dimension (0 if r=0)
    m = m_f + m_e         # Total state dimension

    # Shock dimension: k factor shocks + N idiosyncratic shocks
    n_shocks = k + N

    # ========================================
    # Build state-space matrices
    # ========================================

    # State vector: αₜ = [fₜ; fₜ₋₁; ...; fₜ₋ₛ₊₁; eₜ; eₜ₋₁; ...; eₜ₋ᵣ₊₁]
    #               = [factor_block; error_block]

    # --- Observation equation: Xₜ = Z αₜ + noise ---
    # Xₜ = Σⱼ₌₀ᵖ λⱼ fₜ₋ⱼ + eₜ
    # Z = [Λ₀, Λ₁, ..., Λₚ, 0, ..., 0 | I_N, 0, ..., 0]
    #      └─── factor block (N × m_f) ───┘  └─ error block (N × m_e) ─┘

    Z = zeros(T, N, m)
    # Factor loadings will be filled in later (Λⱼ at columns j*k+1:(j+1)*k)
    # For now, random initialization for Λ₀
    for j = 1:k, i = 1:N
        Z[i, j] = randn(T) * T(0.1)
    end
    # Error block: eₜ is read from state (first N elements of error block)
    if r > 0
        for i = 1:N
            Z[i, m_f+i] = one(T)
        end
    end

    # H = 0 when r > 0 (errors are in state), or diagonal σ²_v when r = 0
    if r > 0
        H = zeros(T, N, N)  # Observation noise is 0, errors are in state
    else
        H = Matrix{T}(I, N, N)  # Diagonal idiosyncratic variances
    end

    # --- State equation: αₜ₊₁ = T αₜ + R ηₜ ---
    # Factor block: companion form for VAR(q)
    # [fₜ₊₁  ]   [Φ₁ Φ₂ ... Φᵧ 0 ... 0] [fₜ    ]   [I]
    # [fₜ    ] = [I  0  ... 0  0 ... 0] [fₜ₋₁  ] + [0] ηₜ
    # [fₜ₋₁  ]   [0  I  ... 0  0 ... 0] [fₜ₋₂  ]   [0]
    # ...

    # Error block: companion form for AR(r)
    # [eₜ₊₁  ]   [δ₁I δ₂I ... δᵣI] [eₜ    ]   [I]
    # [eₜ    ] = [I   0   ... 0  ] [eₜ₋₁  ] + [0] vₜ
    # ...

    Tmat = zeros(T, m, m)

    # Factor block: VAR companion form
    # First k rows: Φ₁, Φ₂, ..., Φᵧ (rest zeros)
    for lag = 1:min(q, s)
        col_start = (lag - 1) * k + 1
        for i = 1:k
            # Initialize with small diagonal AR coefficients
            Tmat[i, col_start+i-1] = T(0.5) / lag
        end
    end
    # Identity blocks for state augmentation
    for lag = 1:(s-1)
        row_start = lag * k + 1
        col_start = (lag - 1) * k + 1
        for i = 1:k
            Tmat[row_start+i-1, col_start+i-1] = one(T)
        end
    end

    # Error block: AR companion form (if r > 0)
    if r > 0
        # First N rows of error block: δ₁I, δ₂I, ..., δᵣI
        for lag = 1:r
            col_start = m_f + (lag - 1) * N + 1
            for i = 1:N
                # Initialize with small AR coefficient
                Tmat[m_f+i, col_start+i-1] = T(0.3) / lag
            end
        end
        # Identity blocks for error state augmentation
        for lag = 1:(r-1)
            row_start = m_f + lag * N + 1
            col_start = m_f + (lag - 1) * N + 1
            for i = 1:N
                Tmat[row_start+i-1, col_start+i-1] = one(T)
            end
        end
    end

    # R matrix: selects which states receive shocks
    # Shocks: [ηₜ (k×1); vₜ (N×1)]
    R = zeros(T, m, n_shocks)
    # Factor shocks go to first k states
    for i = 1:k
        R[i, i] = one(T)
    end
    # Idiosyncratic shocks go to first N states of error block (or nowhere if r=0)
    if r > 0
        for i = 1:N
            R[m_f+i, k+i] = one(T)
        end
    end

    # Q matrix: shock covariances
    # [Σ_η  0  ]
    # [0    Σ_v]  where Σ_v is diagonal
    Q = zeros(T, n_shocks, n_shocks)
    # Factor innovation covariance (initialize as identity)
    for i = 1:k
        Q[i, i] = T(0.1)
    end
    # Idiosyncratic innovation variances
    for i = 1:N
        Q[k+i, k+i] = one(T)
    end

    # Initial state
    a1 = zeros(T, m)
    P1 = Matrix{T}(I, m, m) * T(10.0)

    # ========================================
    # Create workspaces
    # ========================================

    kf_ws = KalmanWorkspace(Z, H, Tmat, R, Q, a1, P1, n_times)
    em_ws = EMWorkspace(kf_ws)

    # Set constraints for EM
    # Z: factor loadings are free (columns 1:k*(p+1)), error selection is fixed
    fill!(em_ws.Z_free, false)
    for lag = 0:p
        col_start = lag * k + 1
        for j = 1:k, i = 1:N
            em_ws.Z_free[i, col_start+j-1] = true
        end
    end

    # H: depends on whether errors are in state
    em_ws.H_diag_only = true

    # T: factor VAR coefficients (first k rows, first k*q columns) are free
    #    error AR coefficients are handled separately in DynamicFactorModelWorkspace
    fill!(em_ws.T_free, false)
    for i = 1:k
        for lag = 1:q
            col_start = (lag - 1) * k + 1
            for j = 1:k
                em_ws.T_free[i, col_start+j-1] = true
            end
        end
    end

    # Q: factor covariance is free, idiosyncratic variances handled separately
    fill!(em_ws.Q_free, false)
    for j = 1:k, i = 1:k
        em_ws.Q_free[i, j] = true
    end

    # ========================================
    # Create DFM-specific workspace
    # ========================================

    # Initialize parameter arrays
    Λ = [zeros(T, N, k) for _ = 0:p]
    # Copy initial loadings from Z
    for lag = 0:p
        col_start = lag * k + 1
        Λ[lag+1] .= Z[:, col_start:(col_start+k-1)]
    end

    Φ = [zeros(T, k, k) for _ = 1:q]
    for lag = 1:q
        col_start = (lag - 1) * k + 1
        Φ[lag] .= Tmat[1:k, col_start:(col_start+k-1)]
    end

    Σ_η = Matrix{T}(I, k, k) * T(0.1)

    δ_vec = zeros(T, max(r, 1))
    if r > 0
        for lag = 1:r
            δ_vec[lag] = T(0.3) / lag
        end
    end

    σ²_v = ones(T, N)

    dfm_ws = DynamicFactorModelWorkspace{T}(
        spec,
        N,
        k,
        p,
        q,
        r,
        n_times,
        m_f,
        m_e,
        m,
        Λ,
        Φ,
        Σ_η,
        δ_vec,
        σ²_v,
        zeros(T, N, N),           # S_ee
        zeros(T, N, N, max(r, 1)), # S_ee_lag
        zeros(T, N, N),           # tmp_nn
        zeros(T, k, k),            # tmp_kk
    )

    return kf_ws, em_ws, dfm_ws
end

"""
    sync_params_to_ssm!(kf_ws::KalmanWorkspace, dfm_ws::DynamicFactorModelWorkspace)

Copy parameters from DynamicFactorModelWorkspace to KalmanWorkspace state-space matrices.
"""
function sync_params_to_ssm!(
    kf_ws::KalmanWorkspace{T},
    dfm_ws::DynamicFactorModelWorkspace{T},
) where {T}
    N, k, p, q, r = dfm_ws.n_obs,
    dfm_ws.n_factors,
    dfm_ws.loading_lags,
    dfm_ws.factor_lags,
    dfm_ws.error_lags
    m_f, m_e = dfm_ws.factor_state_dim, dfm_ws.error_state_dim
    s = div(m_f, k)

    # Update Z with factor loadings
    for lag = 0:p
        col_start = lag * k + 1
        for j = 1:k, i = 1:N
            kf_ws.Z[i, col_start+j-1] = dfm_ws.Λ[lag+1][i, j]
        end
    end

    # Update T with factor VAR coefficients
    for lag = 1:q
        col_start = (lag - 1) * k + 1
        for j = 1:k, i = 1:k
            kf_ws.Tmat[i, col_start+j-1] = dfm_ws.Φ[lag][i, j]
        end
    end

    # Update T with AR error coefficients
    if r > 0
        for lag = 1:r
            col_start = m_f + (lag - 1) * N + 1
            for i = 1:N
                kf_ws.Tmat[m_f+i, col_start+i-1] = dfm_ws.δ[lag]
            end
        end
    end

    # Update Q with factor innovation covariance
    for j = 1:k, i = 1:k
        kf_ws.Q[i, j] = dfm_ws.Σ_η[i, j]
    end

    # Update Q with idiosyncratic variances
    for i = 1:N
        kf_ws.Q[k+i, k+i] = dfm_ws.σ²_v[i]
    end

    # Update H (only used if r = 0)
    if r == 0
        for i = 1:N
            kf_ws.H[i, i] = dfm_ws.σ²_v[i]
        end
    end

    # Update RQR
    _update_RQR!(kf_ws)

    return nothing
end

"""
    sync_params_from_ssm!(dfm_ws::DynamicFactorModelWorkspace, kf_ws::KalmanWorkspace)

Copy parameters from KalmanWorkspace back to DynamicFactorModelWorkspace after EM updates.
"""
function sync_params_from_ssm!(
    dfm_ws::DynamicFactorModelWorkspace{T},
    kf_ws::KalmanWorkspace{T},
) where {T}
    N, k, p, q, r = dfm_ws.n_obs,
    dfm_ws.n_factors,
    dfm_ws.loading_lags,
    dfm_ws.factor_lags,
    dfm_ws.error_lags
    m_f = dfm_ws.factor_state_dim

    # Extract factor loadings from Z
    for lag = 0:p
        col_start = lag * k + 1
        for j = 1:k, i = 1:N
            dfm_ws.Λ[lag+1][i, j] = kf_ws.Z[i, col_start+j-1]
        end
    end

    # Extract factor VAR coefficients from T
    for lag = 1:q
        col_start = (lag - 1) * k + 1
        for j = 1:k, i = 1:k
            dfm_ws.Φ[lag][i, j] = kf_ws.Tmat[i, col_start+j-1]
        end
    end

    # Extract factor innovation covariance from Q
    for j = 1:k, i = 1:k
        dfm_ws.Σ_η[i, j] = kf_ws.Q[i, j]
    end

    # Extract idiosyncratic variances from Q
    for i = 1:N
        dfm_ws.σ²_v[i] = kf_ws.Q[k+i, k+i]
    end

    return nothing
end

"""
    compute_error_sufficient_stats!(dfm_ws::DynamicFactorModelWorkspace, kf_ws::KalmanWorkspace, y::AbstractMatrix)

Compute sufficient statistics for AR error parameter estimation.
Extracts smoothed idiosyncratic errors and computes autocovariances.
"""
function compute_error_sufficient_stats!(
    dfm_ws::DynamicFactorModelWorkspace{T},
    kf_ws::KalmanWorkspace{T},
    y::AbstractMatrix,
) where {T}
    N, k, p, r = dfm_ws.n_obs, dfm_ws.n_factors, dfm_ws.loading_lags, dfm_ws.error_lags
    n = dfm_ws.n_times
    m_f = dfm_ws.factor_state_dim

    if r == 0
        return  # No AR errors to estimate
    end

    # Reset sufficient statistics
    fill!(dfm_ws.S_ee, zero(T))
    fill!(dfm_ws.S_ee_lag, zero(T))

    # Extract smoothed errors from state
    # State: [factor_block; error_block]
    # Error at time t: αₜ[m_f+1:m_f+N]

    @inbounds for t = 1:n
        # Current error
        e_t = view(kf_ws.αs, (m_f+1):(m_f+N), t)
        V_t = view(kf_ws.Vs, (m_f+1):(m_f+N), (m_f+1):(m_f+N), t)

        # S_ee: Σ E[eₜ eₜ' | Y] - only diagonal needed
        for i = 1:N
            dfm_ws.S_ee[i, i] += V_t[i, i] + e_t[i]^2
        end

        # S_ee_lag[j]: Σ E[eₜ eₜ₋ⱼ' | Y] for j = 1:r
        for lag = 1:r
            if t > lag
                e_tlag = view(kf_ws.αs, (m_f+1):(m_f+N), t-lag)
                # Cross-covariance from smoother (approximation: use state cross-cov)
                # For simplicity, use E[eₜ]E[eₜ₋ⱼ]' (ignoring cross-variance term)
                # A proper implementation would need Cov[αₜ, αₜ₋ⱼ | Y] for all lags
                for i = 1:N
                    dfm_ws.S_ee_lag[i, i, lag] += e_t[i] * e_tlag[i]
                end
            end
        end
    end

    return nothing
end

"""
    update_ar_errors!(dfm_ws::DynamicFactorModelWorkspace, kf_ws::KalmanWorkspace)

M-step update for AR error coefficients δ and innovation variances σ²_v.

For AR(r) errors with common coefficients:
    eᵢₜ = δ₁ eᵢ,ₜ₋₁ + ... + δᵣ eᵢ,ₜ₋ᵣ + vᵢₜ

Estimates δ by pooled regression across all series.
"""
function update_ar_errors!(
    dfm_ws::DynamicFactorModelWorkspace{T},
    kf_ws::KalmanWorkspace{T},
) where {T}
    N, r = dfm_ws.n_obs, dfm_ws.error_lags
    n = dfm_ws.n_times
    m_f = dfm_ws.factor_state_dim

    if r == 0
        return  # No AR errors
    end

    # Pooled Yule-Walker estimation for common AR coefficients
    # Solve: [γ(1)]     [γ(0)   γ(1)   ... γ(r-1)] [δ₁]
    #        [γ(2)]  =  [γ(1)   γ(0)   ... γ(r-2)] [δ₂]
    #        [...]      [...]                       [...]
    #        [γ(r)]     [γ(r-1) γ(r-2) ... γ(0)  ] [δᵣ]
    #
    # where γ(j) = (1/N) Σᵢ E[eᵢₜ eᵢ,ₜ₋ⱼ] (pooled autocovariance)

    # Compute pooled autocovariances
    γ = zeros(T, r + 1)  # γ[1] = γ(0), γ[2] = γ(1), etc.

    # γ(0) from S_ee diagonal
    γ[1] = sum(diag(dfm_ws.S_ee)) / (N * (n - r))

    # γ(j) from S_ee_lag
    for lag = 1:r
        γ[lag+1] = sum(diag(view(dfm_ws.S_ee_lag,:,:,lag))) / (N * (n - r))
    end

    # Build Yule-Walker system
    Γ = zeros(T, r, r)  # Toeplitz matrix of γ
    for i = 1:r, j = 1:r
        Γ[i, j] = γ[abs(i-j)+1]
    end

    γ_vec = γ[2:(r+1)]  # [γ(1), γ(2), ..., γ(r)]

    # Solve for δ
    # Add regularization for numerical stability
    for i = 1:r
        Γ[i, i] += T(1e-8)
    end

    δ_new = Γ \ γ_vec

    # Ensure stationarity: check roots outside unit circle
    # Simple check: sum of |δᵢ| < 1 (sufficient but not necessary)
    δ_sum = sum(abs, δ_new)
    if δ_sum >= one(T)
        # Scale down to ensure stationarity
        δ_new .*= T(0.95) / δ_sum
    end

    copyto!(dfm_ws.δ, δ_new)

    # Update innovation variances
    # σ²_v,i = E[eᵢₜ²] - 2 Σⱼ δⱼ E[eᵢₜ eᵢ,ₜ₋ⱼ] + Σⱼ Σₖ δⱼ δₖ E[eᵢ,ₜ₋ⱼ eᵢ,ₜ₋ₖ]
    # Simplified: σ²_v,i ≈ γᵢ(0) - Σⱼ δⱼ γᵢ(j)

    for i = 1:N
        γ0_i = dfm_ws.S_ee[i, i] / (n - r)
        sum_δγ = zero(T)
        for lag = 1:r
            γlag_i = dfm_ws.S_ee_lag[i, i, lag] / (n - r)
            sum_δγ += dfm_ws.δ[lag] * γlag_i
        end
        dfm_ws.σ²_v[i] = max(γ0_i - sum_δγ, T(1e-10))
    end

    return nothing
end

"""
    _em_dfm!(kf_ws::KalmanWorkspace, em_ws::EMWorkspace, dfm_ws::DynamicFactorModelWorkspace,
                 y::AbstractMatrix; maxiter=500, tol=1e-6, verbose=false)

EM algorithm for full dynamic factor model.
"""
function _em_dfm!(
    kf_ws::KalmanWorkspace{T},
    em_ws::EMWorkspace{T},
    dfm_ws::DynamicFactorModelWorkspace{T},
    y::AbstractMatrix;
    maxiter::Int = 500,
    tol::Real = 1e-6,
    verbose::Bool = false,
) where {T}

    N, k, p, q, r = dfm_ws.n_obs,
    dfm_ws.n_factors,
    dfm_ws.loading_lags,
    dfm_ws.factor_lags,
    dfm_ws.error_lags

    loglik_history = Vector{T}(undef, maxiter)
    ll_prev = T(-Inf)
    converged = false
    iter = 0

    for i = 1:maxiter
        iter = i

        # Sync parameters to state-space form
        sync_params_to_ssm!(kf_ws, dfm_ws)

        # E-step: Filter and smooth
        filter_and_smooth!(kf_ws, y)
        ll = kf_ws.loglik
        loglik_history[i] = ll

        # Check convergence
        if i > 1
            ll_change = abs(ll - ll_prev)
            rel_change = ll_change / (abs(ll_prev) + T(1e-10))

            if verbose && (i % 10 == 0 || i <= 5)
                println("EM iter $i: loglik = $ll, rel_change = $rel_change")
            end

            if rel_change < tol
                converged = true
                if verbose
                    println("EM converged at iteration $i")
                end
                break
            end
        elseif verbose
            println("EM iter $i: loglik = $ll")
        end

        ll_prev = ll

        # Compute sufficient statistics
        n_valid = compute_sufficient_stats!(em_ws, kf_ws, y)

        # M-step: Update factor loadings and VAR
        update_Z!(kf_ws, em_ws, n_valid)  # Updates Λ
        update_T!(kf_ws, em_ws)           # Updates Φ
        update_Q!(kf_ws, em_ws)           # Updates Σ_η

        # M-step: Update AR error parameters (if r > 0)
        if r > 0
            compute_error_sufficient_stats!(dfm_ws, kf_ws, y)
            update_ar_errors!(dfm_ws, kf_ws)
        else
            # Update idiosyncratic variances directly from H
            update_H!(kf_ws, em_ws, n_valid)
            for i = 1:N
                dfm_ws.σ²_v[i] = kf_ws.H[i, i]
            end
        end

        # Sync back to dfm_ws
        sync_params_from_ssm!(dfm_ws, kf_ws)
    end

    # Final sync and filter
    sync_params_to_ssm!(kf_ws, dfm_ws)
    filter_and_smooth!(kf_ws, y)

    return (
        converged = converged,
        iterations = iter,
        loglik = kf_ws.loglik,
        loglik_history = loglik_history[1:iter],
    )
end

"""
    DynamicFactorModel{T}

Mutable struct holding specification, workspaces, and fitting state for a dynamic factor model.

# Usage
```julia
# Step 1: Create model (allocates workspaces)
model = DynamicFactorModel(n_obs, n_factors, n_times;
                           loading_lags=0, factor_lags=1, error_lags=0)

# Step 2: Fit with EM algorithm
fit!(EM(), model, y; maxiter=500, tol=1e-6, verbose=false)

# Step 3: Check results
isfitted(model)       # true
isconverged(model)    # true/false
loglikelihood(model)  # final log-likelihood

# Step 4: Access factors and forecast
f = factors(model)    # k × n smoothed factors
fc = forecast(model, h)  # h-step ahead forecast
```

# Model Structure (following Stock & Watson notation)
    Xₜ = λ(L) fₜ + eₜ
    fₜ = Ψ(L) fₜ₋₁ + ηₜ
    eᵢₜ = δ(L) eᵢ,ₜ₋₁ + vᵢₜ

# Fields
## Specification
- `spec::DynamicFactorModelSpec` - Model dimensions and lag structure

## Workspaces (allocated at construction)
- `kf_ws::KalmanWorkspace{T}` - Kalman filter/smoother workspace
- `em_ws::EMWorkspace{T}` - EM algorithm workspace
- `dfm_ws::DynamicFactorModelWorkspace{T}` - DFM-specific workspace

## Fitting state (updated by fit!)
- `fitted::Bool` - Whether model has been fitted
- `converged::Bool` - Whether EM converged
- `iterations::Int` - Number of EM iterations
- `loglik::T` - Final log-likelihood
- `loglik_history::Vector{T}` - Log-likelihood at each iteration
"""
mutable struct DynamicFactorModel{T<:Real} <: AbstractStateSpaceModel
    # Specification
    spec::DynamicFactorModelSpec

    # Workspaces (allocated at construction)
    kf_ws::KalmanWorkspace{T}
    em_ws::EMWorkspace{T}
    dfm_ws::DynamicFactorModelWorkspace{T}

    # Fitting state
    fitted::Bool
    converged::Bool
    iterations::Int
    loglik::T
    loglik_history::Vector{T}
end

"""
    DynamicFactorModel(n_obs::Int, n_factors::Int, n_times::Int;
                       loading_lags::Int=0, factor_lags::Int=1, error_lags::Int=0,
                       T::Type{<:Real}=Float64)

Construct a dynamic factor model with pre-allocated workspaces.

# Arguments
- `n_obs`: Number of observables (N)
- `n_factors`: Number of latent factors (k)
- `n_times`: Number of time periods (n)

# Keyword Arguments
- `loading_lags`: Lags in λ(L), p=0 means static loadings (default: 0)
- `factor_lags`: Lags in factor VAR Ψ(L) (default: 1)
- `error_lags`: Lags in AR errors δ(L), r=0 means white noise (default: 0)
- `T`: Element type for matrices (default: Float64)

# Returns
- `DynamicFactorModel{T}` ready for fitting with `fit!(EM(), model, data)`

# Example
```julia
# 100 observables, 6 factors, 200 time periods, VAR(3) dynamics
model = DynamicFactorModel(100, 6, 200; factor_lags=3)
```
"""
function DynamicFactorModel(
    n_obs::Int,
    n_factors::Int,
    n_times::Int;
    loading_lags::Int = 0,
    factor_lags::Int = 1,
    error_lags::Int = 0,
    T::Type{<:Real} = Float64,
)
    spec = DynamicFactorModelSpec(n_obs, n_factors, loading_lags, factor_lags, error_lags)
    kf_ws, em_ws, dfm_ws = _setup_dfm(spec, n_times, T)

    return DynamicFactorModel{T}(
        spec,
        kf_ws,
        em_ws,
        dfm_ws,
        false,           # fitted
        false,           # converged
        0,               # iterations
        T(-Inf),         # loglik
        T[],              # loglik_history
    )
end

# ============================================
# Status Accessors
# ============================================

"""
    isfitted(model::DynamicFactorModel) -> Bool

Check whether the model has been fitted.
"""
isfitted(model::DynamicFactorModel) = model.fitted

"""
    isconverged(model::DynamicFactorModel) -> Bool

Check whether EM converged. Throws error if model not fitted.
"""
function isconverged(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    return model.converged
end

"""
    loglikelihood(model::DynamicFactorModel) -> T

Return final log-likelihood. Throws error if model not fitted.
"""
function loglikelihood(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    return model.loglik
end

"""
    niterations(model::DynamicFactorModel) -> Int

Return number of EM iterations. Throws error if model not fitted.
"""
function niterations(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    return model.iterations
end

# ============================================
# Factor Accessors
# ============================================

"""
    factors(model::DynamicFactorModel)

Extract smoothed factors (k × n) from fitted model.
"""
function factors(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    k = model.spec.n_factors
    return model.kf_ws.αs[1:k, :]
end

"""
    factors_cov(model::DynamicFactorModel)

Extract smoothed factor covariances (k × k × n) from fitted model.
"""
function factors_cov(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    k = model.spec.n_factors
    return model.kf_ws.Vs[1:k, 1:k, :]
end

"""
    filtered_factors(model::DynamicFactorModel)

Extract filtered factors (k × n) from fitted model.
"""
function filtered_factors(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    k = model.spec.n_factors
    return model.kf_ws.att[1:k, :]
end

"""
    filtered_factors_cov(model::DynamicFactorModel)

Extract filtered factor covariances (k × k × n) from fitted model.
"""
function filtered_factors_cov(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    k = model.spec.n_factors
    return model.kf_ws.Ptt[1:k, 1:k, :]
end

# ============================================
# Parameter Accessors
# ============================================

"""
    loadings(model::DynamicFactorModel)

Return factor loadings [Λ₀, Λ₁, ..., Λₚ] from fitted model.
"""
function loadings(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    return model.dfm_ws.Λ
end

"""
    var_coefficients(model::DynamicFactorModel)

Return factor VAR coefficients [Φ₁, ..., Φq] from fitted model.
"""
function var_coefficients(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    return model.dfm_ws.Φ
end

"""
    innovation_cov(model::DynamicFactorModel)

Return factor innovation covariance Σ_η (k × k) from fitted model.
"""
function innovation_cov(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    return model.dfm_ws.Σ_η
end

"""
    ar_coefficients(model::DynamicFactorModel)

Return AR error coefficients [δ₁, ..., δᵣ] from fitted model.
"""
function ar_coefficients(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    return model.dfm_ws.δ
end

"""
    idiosyncratic_variances(model::DynamicFactorModel)

Return idiosyncratic innovation variances σ²_v (length N) from fitted model.
"""
function idiosyncratic_variances(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    return model.dfm_ws.σ²_v
end

# ============================================
# fit! (StatsAPI-style interface)
# ============================================

"""
    fit!(::EM, model::DynamicFactorModel, y::AbstractMatrix;
         maxiter::Int=500, tol::Real=1e-6, verbose::Bool=false)

Fit a dynamic factor model using the EM algorithm.

# Arguments
- `EM()`: Estimation method selector
- `model`: DynamicFactorModel to fit (mutated in-place)
- `y`: Observations (N × n matrix), missing values as NaN

# Keyword Arguments
- `maxiter`: Maximum EM iterations (default: 500)
- `tol`: Convergence tolerance for log-likelihood (default: 1e-6)
- `verbose`: Print progress (default: false)

# Returns
- The fitted `model` (same object, mutated)

# Example
```julia
model = DynamicFactorModel(100, 6, 200; factor_lags=3)
fit!(EM(), model, y; verbose=true)
f = factors(model)  # k × n smoothed factors
```
"""
function fit!(
    ::EM,
    model::DynamicFactorModel{T},
    y::AbstractMatrix;
    maxiter::Int = 500,
    tol::Real = 1e-6,
    verbose::Bool = false,
) where {T}

    N, n = size(y)
    spec = model.spec

    # Validate dimensions
    N == spec.n_obs ||
        throw(DimensionMismatch("Data has $N observables but model expects $(spec.n_obs)"))
    n == model.kf_ws.n_times || throw(
        DimensionMismatch(
            "Data has $n time periods but model workspace allocated for $(model.kf_ws.n_times)",
        ),
    )

    if verbose
        println("Fitting DynamicFactorModel via EM:")
        println("  Observables (N): ", spec.n_obs)
        println("  Factors (k): ", spec.n_factors)
        println("  Loading lags (p): ", spec.loading_lags)
        println("  Factor VAR lags (q): ", spec.factor_lags)
        println("  Error AR lags (r): ", spec.error_lags)
        s = max(spec.factor_lags, spec.loading_lags + 1)
        m_f = spec.n_factors * s
        m_e = spec.n_obs * spec.error_lags
        println("  State dimension: $(m_f + m_e) (factors: $m_f, errors: $m_e)")
    end

    # Run EM algorithm
    result = _em_dfm!(
        model.kf_ws,
        model.em_ws,
        model.dfm_ws,
        y;
        maxiter = maxiter,
        tol = tol,
        verbose = verbose,
    )

    # Update model state
    model.fitted = true
    model.converged = result.converged
    model.iterations = result.iterations
    model.loglik = result.loglik
    model.loglik_history = result.loglik_history

    return model
end

# ============================================
# Forecasting Functions
# ============================================

"""
    DynamicFactorModelForecast{T}

Result of h-step ahead forecasting.

# Fields
- `h::Int` - Forecast horizon
- `state_mean::Matrix{T}` - Forecasted state means E[αₙ₊ₕ|Y₁:ₙ] (m × h)
- `state_cov::Array{T,3}` - Forecasted state covariances Var[αₙ₊ₕ|Y₁:ₙ] (m × m × h)
- `obs_mean::Matrix{T}` - Forecasted observation means E[yₙ₊ₕ|Y₁:ₙ] (N × h)
- `obs_cov::Array{T,3}` - Forecasted observation covariances Var[yₙ₊ₕ|Y₁:ₙ] (N × N × h)
- `factor_mean::Matrix{T}` - Forecasted factor means (k × h)
- `factor_cov::Array{T,3}` - Forecasted factor covariances (k × k × h)
"""
struct DynamicFactorModelForecast{T<:Real}
    h::Int
    state_mean::Matrix{T}      # m × h
    state_cov::Array{T,3}      # m × m × h
    obs_mean::Matrix{T}        # N × h
    obs_cov::Array{T,3}        # N × N × h
    factor_mean::Matrix{T}     # k × h
    factor_cov::Array{T,3}     # k × k × h
end

"""
    forecast(model::DynamicFactorModel, h::Int)

Compute h-step ahead forecasts from a fitted DFM.

Uses the filtered state at the last observation (αₙ|ₙ, Pₙ|ₙ) and iterates
the state equation forward:
    E[αₙ₊ₕ|Y₁:ₙ] = Tʰ αₙ|ₙ
    Var[αₙ₊ₕ|Y₁:ₙ] = Tʰ Pₙ|ₙ (T')ʰ + Σⱼ₌₀ʰ⁻¹ Tʲ RQR' (T')ʲ

# Arguments
- `model::DynamicFactorModel` - Fitted DFM model
- `h::Int` - Forecast horizon

# Returns
- `DynamicFactorModelForecast` with forecasted states and observations

# Example
```julia
model = DynamicFactorModel(100, 6, 200; factor_lags=3)
fit!(EM(), model, y)
fc = forecast(model, 4)
ci = forecast_interval(fc, 0.05)
```
"""
function forecast(model::DynamicFactorModel{T}, h::Int) where {T}
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))

    k = model.spec.n_factors
    N = model.spec.n_obs
    kf_ws = model.kf_ws
    m = kf_ws.state_dim
    n = kf_ws.n_times

    # Initialize with filtered state at time n
    a_h = copy(kf_ws.att[:, n])
    P_h = copy(kf_ws.Ptt[:, :, n])

    # Storage for forecasts
    state_mean = Matrix{T}(undef, m, h)
    state_cov = Array{T,3}(undef, m, m, h)
    obs_mean = Matrix{T}(undef, N, h)
    obs_cov = Array{T,3}(undef, N, N, h)

    # Temporaries
    tmp_mm = similar(P_h)
    tmp_nm = Matrix{T}(undef, N, m)

    Tmat = kf_ws.Tmat
    Z = kf_ws.Z
    H = kf_ws.H
    RQR = kf_ws.RQR

    @inbounds for j = 1:h
        # State forecast: αₙ₊ⱼ|ₙ = T αₙ₊ⱼ₋₁|ₙ
        a_new = Tmat * a_h

        # Covariance forecast: Pₙ₊ⱼ|ₙ = T Pₙ₊ⱼ₋₁|ₙ T' + RQR'
        mul!(tmp_mm, Tmat, P_h)
        mul!(P_h, tmp_mm, Tmat')
        P_h .+= RQR

        # Store state forecast
        state_mean[:, j] .= a_new
        state_cov[:, :, j] .= P_h

        # Observation forecast: yₙ₊ⱼ|ₙ = Z αₙ₊ⱼ|ₙ
        mul!(view(obs_mean, :, j), Z, a_new)

        # Observation covariance: Fₙ₊ⱼ|ₙ = Z Pₙ₊ⱼ|ₙ Z' + H
        mul!(tmp_nm, Z, P_h)
        mul!(view(obs_cov,:,:,j), tmp_nm, Z')
        obs_cov[:, :, j] .+= H

        # Update for next iteration
        a_h .= a_new
    end

    # Extract factor forecasts (first k elements of state)
    factor_mean = state_mean[1:k, :]
    factor_cov = state_cov[1:k, 1:k, :]

    return DynamicFactorModelForecast{T}(
        h,
        state_mean,
        state_cov,
        obs_mean,
        obs_cov,
        factor_mean,
        factor_cov,
    )
end

"""
    forecast_interval(fc::DynamicFactorModelForecast, α::Real=0.05)

Compute (1-α) confidence intervals for forecasts.

# Returns
- NamedTuple with `lower` and `upper` for observation forecasts, each N × h
"""
function forecast_interval(fc::DynamicFactorModelForecast{T}, α::Real = 0.05) where {T}
    # z-value for (1-α) confidence interval: Φ⁻¹(1 - α/2) ≈ 1.96 for α=0.05
    z = quantile_normal(one(T) - T(α) / 2)

    N, h = size(fc.obs_mean)
    lower = similar(fc.obs_mean)
    upper = similar(fc.obs_mean)

    @inbounds for j = 1:h
        for i = 1:N
            se = sqrt(fc.obs_cov[i, i, j])
            lower[i, j] = fc.obs_mean[i, j] - z * se
            upper[i, j] = fc.obs_mean[i, j] + z * se
        end
    end

    return (lower = lower, upper = upper)
end

"""
    factor_forecast_interval(fc::DynamicFactorModelForecast, α::Real=0.05)

Compute (1-α) confidence intervals for factor forecasts.

# Returns
- NamedTuple with `lower` and `upper` for factor forecasts, each k × h
"""
function factor_forecast_interval(
    fc::DynamicFactorModelForecast{T},
    α::Real = 0.05,
) where {T}
    z = quantile_normal(one(T) - T(α) / 2)

    k, h = size(fc.factor_mean)
    lower = similar(fc.factor_mean)
    upper = similar(fc.factor_mean)

    @inbounds for j = 1:h
        for i = 1:k
            se = sqrt(fc.factor_cov[i, i, j])
            lower[i, j] = fc.factor_mean[i, j] - z * se
            upper[i, j] = fc.factor_mean[i, j] + z * se
        end
    end

    return (lower = lower, upper = upper)
end

# Simple approximation for normal quantile (avoid dependency)
function quantile_normal(p::T) where {T}
    # Rational approximation for Φ⁻¹(p)
    # Uses Abramowitz & Stegun 26.2.23
    # For p > 0.5, use symmetry: Φ⁻¹(p) = -Φ⁻¹(1-p)
    if p > 0.5
        return -quantile_normal(one(T) - p)
    end
    # For p ≤ 0.5, compute negative quantile
    t = sqrt(-2 * log(p))
    c0, c1, c2 = T(2.515517), T(0.802853), T(0.010328)
    d1, d2, d3 = T(1.432788), T(0.189269), T(0.001308)
    # This gives the negative of what we want, so negate it
    return -(t - (c0 + c1*t + c2*t^2) / (one(T) + d1*t + d2*t^2 + d3*t^3))
end
