"""
    filter_ad.jl

AD-compatible Kalman filter implementation for use with automatic differentiation.

This module provides pure functional filter implementations that:
- Avoid mutable state (no GrowableArrays)
- Avoid try-catch blocks
- Use element type propagation for AD compatibility
- Handle missing observations (marked as NaN)
- Use StaticArrays for small state dimensions (≤ STATIC_THRESHOLD)

Missing Data Handling:
When an observation contains NaN, the filter skips the measurement update
and propagates the state using only the transition equation:
    a_{t+1} = T * a_t
    P_{t+1} = T * P_t * T' + R * Q * R'
This is equivalent to having infinite observation noise for that period.

StaticArrays Optimization:
When KFParms contains StaticArrays and initial state (a1, P1) are also static,
the filter uses fully static arithmetic in the inner loop, avoiding all heap
allocations and enabling compiler optimizations like loop unrolling and SIMD.
"""

using LinearAlgebra
using StaticArrays

# ============================================
# Missing observation detection
# ============================================

"""
    _has_missing(y_t::AbstractVector) -> Bool

Check if observation vector contains any NaN (missing) values.
"""
@inline _has_missing(y_t::AbstractVector) = any(isnan, y_t)

"""
    _has_missing(y_t::Real) -> Bool

Check if scalar observation is NaN (missing).
"""
@inline _has_missing(y_t::Real) = isnan(y_t)

"""
    kalman_loglik(p::KFParms, y, a1, P1) -> loglik

Compute the log-likelihood of a linear Gaussian state space model using the Kalman filter.

This is an AD-compatible implementation that:
- Uses pure functional operations (no in-place mutations)
- Avoids try-catch blocks
- Propagates element types correctly for automatic differentiation
- Handles missing observations (NaN values)

# Arguments
- `p::KFParms`: State space parameters (Z, H, T, R, Q)
- `y::AbstractMatrix`: Observations (p × n matrix, where p is observation dim, n is time).
                       Missing values should be marked as `NaN`.
- `a1::AbstractVector`: Initial state mean
- `P1::AbstractMatrix`: Initial state covariance

# Returns
- `loglik::Real`: Log-likelihood of the observed (non-missing) data

# State Space Model
```
y_t = Z * α_t + ε_t,    ε_t ~ N(0, H)
α_{t+1} = T * α_t + R * η_t,    η_t ~ N(0, Q)
```

# Missing Data
When `y[:, t]` contains any NaN, the observation is treated as missing.
The filter skips the measurement update and propagates the state:
    a_{t+1} = T * a_t
    P_{t+1} = T * P_t * T' + R * Q * R'
"""
function kalman_loglik(
    p::KFParms,
    y::AbstractMatrix,
    a1::AbstractVector,
    P1::AbstractMatrix,
)
    n = size(y, 2)
    obs_dim = size(y, 1)

    # Get element type from parameters for AD compatibility
    T = promote_type(
        eltype(p.Z),
        eltype(p.H),
        eltype(p.T),
        eltype(p.R),
        eltype(p.Q),
        eltype(a1),
        eltype(P1),
    )

    # Initialize state
    a = Vector{T}(a1)
    P = Matrix{T}(P1)

    # Initialize log-likelihood (without constant term)
    loglik = zero(T)

    # Count non-missing observations for constant term
    n_obs = 0

    for t = 1:n
        y_t = y[:, t]

        # Check for missing observation
        if _has_missing(y_t)
            # Missing: skip update, just propagate state
            a = p.T * a
            P = p.T * P * p.T' + p.R * p.Q * p.R'
            continue
        end

        # Observation is present
        n_obs += 1

        # Innovation
        v = y_t - p.Z * a

        # Innovation covariance
        F = p.Z * P * p.Z' + p.H

        # For numerical stability with scalars
        if obs_dim == 1
            F_val = F[1, 1]
            # Check for non-positive variance
            if F_val <= zero(T)
                return T(-Inf)
            end
            Finv = one(T) / F_val
            logdetF = log(F_val)
            quad_form = v[1]^2 * Finv
        else
            # For multivariate case, use Cholesky for stability
            F_sym = Symmetric((F + F') / 2)  # Ensure symmetry
            chol_result = cholesky(F_sym; check = false)
            if !issuccess(chol_result)
                # F is not positive definite - return -Inf
                return T(-Inf)
            end
            Finv = inv(chol_result)
            logdetF = 2 * sum(log.(diag(chol_result.U)))
            quad_form = dot(v, Finv * v)
        end

        # Log-likelihood contribution
        loglik += -T(0.5) * (logdetF + quad_form)

        # Check for NaN/Inf
        if !isfinite(loglik)
            return T(-Inf)
        end

        # Kalman gain
        K = p.T * P * p.Z' * Finv

        # State update (predicted state for next period)
        a = p.T * a + K * v

        # Covariance update
        # P = T * P * T' + R * Q * R' - K * Z * P * T'
        # More numerically stable form:
        L = p.T - K * p.Z
        P = L * P * p.T' + p.R * p.Q * p.R'
    end

    # Constant term based on actual observations
    const_term = -obs_dim * n_obs * log(T(2π)) / 2

    return loglik + const_term
end

# ============================================
# Static specialization for kalman_loglik
# ============================================

"""
    _is_static_kfparms(p::KFParms) -> Bool

Check if all KFParms matrices are StaticArrays.
"""
@inline _is_static_kfparms(::KFParms{<:SMatrix,<:SMatrix,<:SMatrix,<:SMatrix,<:SMatrix}) =
    true
@inline _is_static_kfparms(::KFParms) = false

"""
    kalman_loglik(p::KFParms{<:SMatrix,...}, y, a1::SVector, P1::SMatrix) -> loglik

Fully static specialization of Kalman filter log-likelihood for small state dimensions.

When all system matrices in KFParms are SMatrix and the initial state (a1, P1) are
SVector/SMatrix, this method keeps all intermediate computations static, avoiding
heap allocations in the inner loop.

This provides significant speedup (3x-17x depending on dimensions) for models where
both state dimension m and observation dimension p are small (≤ STATIC_THRESHOLD).

# Performance Notes
- Zero allocations in the inner loop
- Enables loop unrolling and SIMD vectorization
- Best for m ≤ 8 and p ≤ 5 (speedup decreases for larger dimensions)
"""
function kalman_loglik(
    p::KFParms{<:SMatrix{P,M},<:SMatrix{P,P},<:SMatrix{M,M},<:SMatrix{M,R},<:SMatrix{R,R}},
    y::AbstractMatrix,
    a1::SVector{M},
    P1::SMatrix{M,M},
) where {P,M,R}
    n = size(y, 2)

    # Get element type for AD compatibility
    ET = promote_type(
        eltype(p.Z),
        eltype(p.H),
        eltype(p.T),
        eltype(p.R),
        eltype(p.Q),
        eltype(a1),
        eltype(P1),
    )

    # Initialize state as static types
    a = SVector{M,ET}(a1)
    P_state = SMatrix{M,M,ET}(P1)

    # Static system matrices (ensure correct element type for AD)
    Z = SMatrix{P,M,ET}(p.Z)
    H = SMatrix{P,P,ET}(p.H)
    T_mat = SMatrix{M,M,ET}(p.T)
    R_mat = SMatrix{M,R,ET}(p.R)
    Q = SMatrix{R,R,ET}(p.Q)

    # Precompute R*Q*R' (constant across iterations)
    RQR = R_mat * Q * transpose(R_mat)

    loglik = zero(ET)
    n_obs = 0

    @inbounds for t = 1:n
        # Extract observation as SVector
        y_t = SVector{P,ET}(ntuple(i -> ET(y[i, t]), Val(P)))

        # Check for missing observation
        if _has_missing(y_t)
            # Missing: skip update, just propagate state
            a = T_mat * a
            P_state = T_mat * P_state * transpose(T_mat) + RQR
            continue
        end

        n_obs += 1

        # Innovation (static)
        v = y_t - Z * a

        # Innovation covariance (static)
        F = Z * P_state * transpose(Z) + H

        # Compute inverse and log-determinant
        if P == 1
            # Scalar observation: direct computation
            F_val = F[1, 1]
            if F_val <= zero(ET)
                return ET(-Inf)
            end
            Finv = SMatrix{1,1,ET}((one(ET) / F_val,))
            logdetF = log(F_val)
            quad_form = v[1]^2 / F_val
        else
            # Multivariate: use Cholesky
            F_sym = Symmetric(SMatrix{P,P,ET}((F + transpose(F)) / 2))
            chol_result = cholesky(F_sym; check = false)
            if !issuccess(chol_result)
                return ET(-Inf)
            end
            # For static matrices, inv is efficient
            Finv = SMatrix{P,P,ET}(inv(chol_result))
            # logdet from Cholesky: 2 * sum(log(diag(U)))
            logdetF = 2 * sum(log.(diag(chol_result.U)))
            quad_form = dot(v, Finv * v)
        end

        # Log-likelihood contribution
        loglik += -ET(0.5) * (logdetF + quad_form)

        if !isfinite(loglik)
            return ET(-Inf)
        end

        # Kalman gain (static)
        K = T_mat * P_state * transpose(Z) * Finv

        # State update
        a = T_mat * a + K * v

        # Covariance update (Joseph form for stability)
        L = T_mat - K * Z
        P_state = L * P_state * transpose(T_mat) + RQR
    end

    # Constant term
    const_term = -P * n_obs * log(ET(2π)) / 2

    return loglik + const_term
end

"""
    kalman_loglik_scalar(Z, H, T, R, Q, a1, P1, y) -> loglik

Scalar (univariate) version of the Kalman filter log-likelihood.

Optimized for the common case of scalar state and observation.
Handles missing observations marked as NaN.
"""
function kalman_loglik_scalar(
    Z::Real,
    H::Real,
    Tmat::Real,
    R::Real,
    Q::Real,
    a1::Real,
    P1::Real,
    y::AbstractVector,
)
    n = length(y)
    T = promote_type(
        typeof(Z),
        typeof(H),
        typeof(Tmat),
        typeof(R),
        typeof(Q),
        typeof(a1),
        typeof(P1),
    )

    a = convert(T, a1)
    P = convert(T, P1)
    loglik = zero(T)
    n_obs = 0

    for t = 1:n
        y_t = y[t]

        # Check for missing observation
        if _has_missing(y_t)
            # Missing: skip update, just propagate state
            a = Tmat * a
            P = Tmat * P * Tmat + R * Q * R
            continue
        end

        n_obs += 1

        v = y_t - Z * a
        F = Z * P * Z + H

        # Check for non-positive variance
        if F <= zero(T)
            return T(-Inf)
        end

        Finv = one(T) / F
        loglik += -T(0.5) * (log(F) + v^2 * Finv)

        # Check for NaN/Inf
        if !isfinite(loglik)
            return T(-Inf)
        end

        K = Tmat * P * Z * Finv
        a = Tmat * a + K * v
        L = Tmat - K * Z
        P = L * P * Tmat + R * Q * R
    end

    const_term = -n_obs * log(T(2π)) / 2
    return loglik + const_term
end

# ============================================
# Convenience wrapper for automatic static conversion
# ============================================

"""
    kalman_loglik_static(p::KFParms, y, a1, P1) -> loglik

Compute log-likelihood using StaticArrays if dimensions are small enough.

Automatically converts inputs to StaticArrays when dimensions ≤ STATIC_THRESHOLD,
then dispatches to the appropriate specialized method.

# Example
```julia
# These will use static inner loop if dimensions are small:
ll = kalman_loglik_static(p, y, a1, P1)

# Equivalent to manually converting:
p_static = KFParms_static(p.Z, p.H, p.T, p.R, p.Q)
a1_static = SVector{m}(a1)
P1_static = SMatrix{m,m}(P1)
ll = kalman_loglik(p_static, y, a1_static, P1_static)
```
"""
function kalman_loglik_static(
    p::KFParms,
    y::AbstractMatrix,
    a1::AbstractVector,
    P1::AbstractMatrix,
)
    m = length(a1)
    obs_dim = size(y, 1)
    r = size(p.Q, 1)

    # Only convert if all dimensions are within threshold
    if m ≤ STATIC_THRESHOLD && obs_dim ≤ STATIC_THRESHOLD && r ≤ STATIC_THRESHOLD
        p_static = KFParms_static(p.Z, p.H, p.T, p.R, p.Q)
        a1_static = to_static_if_small(a1)
        P1_static = to_static_if_small(P1)
        return kalman_loglik(p_static, y, a1_static, P1_static)
    else
        return kalman_loglik(p, y, a1, P1)
    end
end

"""
    kalman_filter(p::KFParms, y, a1, P1) -> KalmanFilterResult

Run full Kalman filter returning both predicted and filtered states.

For `n` observations, returns `n` time points:
- `at[:, t]` = E[αₜ | y₁:ₜ₋₁] (predicted state before seeing yₜ)
- `att[:, t]` = E[αₜ | y₁:ₜ] (filtered state after seeing yₜ)
- `Pt[:, :, t]` = Var[αₜ | y₁:ₜ₋₁] (predicted covariance)
- `Ptt[:, :, t]` = Var[αₜ | y₁:ₜ] (filtered covariance)

Returns a `KalmanFilterResult` with:
- `loglik`: Log-likelihood (of non-missing observations)
- `at`: Predicted state means (m × n)
- `Pt`: Predicted state covariances (m × m × n)
- `att`: Filtered state means (m × n)
- `Ptt`: Filtered state covariances (m × m × n)
- `vt`: Innovations (p × n), NaN for missing observations
- `Ft`: Innovation covariances (p × p × n)
- `Kt`: Kalman gains (m × p × n), zero for missing observations
- `missing_mask`: BitVector indicating missing observations (length n)

Use accessor methods: `predicted_states`, `filtered_states`, `variances_predicted_states`,
`variances_filtered_states`, `prediction_errors`, `variances_prediction_errors`,
`kalman_gains`, `loglikelihood`.

# Missing Data
When `y[:, t]` contains any NaN, the observation is treated as missing.
The filter skips the measurement update and propagates the state.
"""
function kalman_filter(
    p::KFParms,
    y::AbstractMatrix,
    a1::AbstractVector,
    P1::AbstractMatrix,
)
    n = size(y, 2)
    obs_dim = size(y, 1)
    state_dim = length(a1)

    ET = promote_type(
        eltype(p.Z),
        eltype(p.H),
        eltype(p.T),
        eltype(p.R),
        eltype(p.Q),
        eltype(a1),
        eltype(P1),
    )

    # Predicted state storage: n entries (at = a_{t|t-1})
    at_store = Matrix{ET}(undef, state_dim, n)
    Pt_store = Array{ET}(undef, state_dim, state_dim, n)

    # Filtered state storage: n entries (att = a_{t|t})
    att_store = Matrix{ET}(undef, state_dim, n)
    Ptt_store = Array{ET}(undef, state_dim, state_dim, n)

    # Innovation storage: n entries
    vt_store = Matrix{ET}(undef, obs_dim, n)
    Ft_store = Array{ET}(undef, obs_dim, obs_dim, n)
    Kt_store = Array{ET}(undef, state_dim, obs_dim, n)
    missing_mask = BitVector(undef, n)

    # Initialize: at[1] = a1, Pt[1] = P1 (initial state is prediction for t=1)
    a_pred = Vector{ET}(a1)
    P_pred = Matrix{ET}(P1)

    loglik = zero(ET)
    n_obs = 0

    for t = 1:n
        y_t = y[:, t]

        # Store predicted state (before measurement update)
        at_store[:, t] = a_pred
        Pt_store[:, :, t] = P_pred

        # Check for missing observation
        if _has_missing(y_t)
            missing_mask[t] = true
            vt_store[:, t] .= ET(NaN)
            Ft_store[:, :, t] = p.Z * P_pred * p.Z' + p.H
            Kt_store[:, :, t] .= zero(ET)

            # For missing: filtered = predicted (no update)
            att_store[:, t] = a_pred
            Ptt_store[:, :, t] = P_pred

            # Propagate state without update
            a_pred = p.T * a_pred
            P_pred = p.T * P_pred * p.T' + p.R * p.Q * p.R'
        else
            missing_mask[t] = false
            n_obs += 1

            # Innovation
            v = y_t - p.Z * a_pred
            F = p.Z * P_pred * p.Z' + p.H

            if obs_dim == 1
                F_val = F[1, 1]
                Finv = fill(one(ET) / F_val, 1, 1)
                logdetF = log(F_val)
                quad_form = v[1]^2 / F_val
            else
                Finv = inv(F)
                logdetF = logdet(F)
                quad_form = dot(v, Finv * v)
            end

            loglik += -ET(0.5) * (logdetF + quad_form)

            # Kalman gain (for predicting next state: K = T * P * Z' * Finv)
            K = p.T * P_pred * p.Z' * Finv

            vt_store[:, t] = v
            Ft_store[:, :, t] = F
            Kt_store[:, :, t] = K

            # Filtered state (measurement update)
            a_filt = a_pred + P_pred * p.Z' * Finv * v
            P_filt = P_pred - P_pred * p.Z' * Finv * p.Z * P_pred

            att_store[:, t] = a_filt
            Ptt_store[:, :, t] = P_filt

            # Predict next state
            a_pred = p.T * a_filt
            P_pred = p.T * P_filt * p.T' + p.R * p.Q * p.R'
        end
    end

    const_term = -obs_dim * n_obs * log(ET(2π)) / 2

    return KalmanFilterResult(
        p,
        loglik + const_term,
        at_store,
        Pt_store,
        att_store,
        Ptt_store,
        vt_store,
        Ft_store,
        Kt_store,
        missing_mask,
    )
end

# ============================================
# Static specialization for kalman_filter
# ============================================

"""
    kalman_filter(p::KFParms{<:SMatrix,...}, y, a1::SVector, P1::SMatrix) -> KalmanFilterResult

Fully static specialization of Kalman filter for small state dimensions.

When all system matrices in KFParms are SMatrix and the initial state (a1, P1) are
SVector/SMatrix, this method uses static arithmetic in the inner loop while storing
results in regular arrays (which must be heap-allocated for the full filter).

The speedup comes from keeping intermediate state computations (a, P, v, F, K) as
StaticArrays, avoiding temporary allocations within each iteration.
"""
function kalman_filter(
    p::KFParms{<:SMatrix{P,M},<:SMatrix{P,P},<:SMatrix{M,M},<:SMatrix{M,R},<:SMatrix{R,R}},
    y::AbstractMatrix,
    a1::SVector{M},
    P1::SMatrix{M,M},
) where {P,M,R}
    n = size(y, 2)

    ET = promote_type(
        eltype(p.Z),
        eltype(p.H),
        eltype(p.T),
        eltype(p.R),
        eltype(p.Q),
        eltype(a1),
        eltype(P1),
    )

    # Output storage (heap-allocated, but filled with static values)
    at_store = Matrix{ET}(undef, M, n)
    Pt_store = Array{ET}(undef, M, M, n)
    att_store = Matrix{ET}(undef, M, n)
    Ptt_store = Array{ET}(undef, M, M, n)
    vt_store = Matrix{ET}(undef, P, n)
    Ft_store = Array{ET}(undef, P, P, n)
    Kt_store = Array{ET}(undef, M, P, n)
    missing_mask = BitVector(undef, n)

    # Static system matrices
    Z = SMatrix{P,M,ET}(p.Z)
    H = SMatrix{P,P,ET}(p.H)
    T_mat = SMatrix{M,M,ET}(p.T)
    R_mat = SMatrix{M,R,ET}(p.R)
    Q = SMatrix{R,R,ET}(p.Q)

    # Precompute R*Q*R'
    RQR = R_mat * Q * transpose(R_mat)

    # Initialize state as static
    a_pred = SVector{M,ET}(a1)
    P_pred = SMatrix{M,M,ET}(P1)

    loglik = zero(ET)
    n_obs = 0

    @inbounds for t = 1:n
        # Extract observation as SVector
        y_t = SVector{P,ET}(ntuple(i -> ET(y[i, t]), Val(P)))

        # Store predicted state
        for i = 1:M
            at_store[i, t] = a_pred[i]
        end
        for j = 1:M, i = 1:M
            Pt_store[i, j, t] = P_pred[i, j]
        end

        if _has_missing(y_t)
            missing_mask[t] = true
            for i = 1:P
                vt_store[i, t] = ET(NaN)
            end
            F_miss = Z * P_pred * transpose(Z) + H
            for j = 1:P, i = 1:P
                Ft_store[i, j, t] = F_miss[i, j]
            end
            for j = 1:P, i = 1:M
                Kt_store[i, j, t] = zero(ET)
            end

            # Filtered = predicted for missing
            for i = 1:M
                att_store[i, t] = a_pred[i]
            end
            for j = 1:M, i = 1:M
                Ptt_store[i, j, t] = P_pred[i, j]
            end

            # Propagate
            a_pred = T_mat * a_pred
            P_pred = T_mat * P_pred * transpose(T_mat) + RQR
        else
            missing_mask[t] = false
            n_obs += 1

            # Innovation (static)
            v = y_t - Z * a_pred
            F = Z * P_pred * transpose(Z) + H

            # Inverse and log-det
            if P == 1
                F_val = F[1, 1]
                Finv = SMatrix{1,1,ET}((one(ET) / F_val,))
                logdetF = log(F_val)
                quad_form = v[1]^2 / F_val
            else
                F_sym = Symmetric(SMatrix{P,P,ET}((F + transpose(F)) / 2))
                chol_result = cholesky(F_sym; check = false)
                if !issuccess(chol_result)
                    # Return early with -Inf likelihood
                    return KalmanFilterResult(
                        p,
                        ET(-Inf),
                        at_store,
                        Pt_store,
                        att_store,
                        Ptt_store,
                        vt_store,
                        Ft_store,
                        Kt_store,
                        missing_mask,
                    )
                end
                Finv = SMatrix{P,P,ET}(inv(chol_result))
                logdetF = 2 * sum(log.(diag(chol_result.U)))
                quad_form = dot(v, Finv * v)
            end

            loglik += -ET(0.5) * (logdetF + quad_form)

            # Kalman gain (static)
            K = T_mat * P_pred * transpose(Z) * Finv

            # Store
            for i = 1:P
                vt_store[i, t] = v[i]
            end
            for j = 1:P, i = 1:P
                Ft_store[i, j, t] = F[i, j]
            end
            for j = 1:P, i = 1:M
                Kt_store[i, j, t] = K[i, j]
            end

            # Filtered state (static)
            a_filt = a_pred + P_pred * transpose(Z) * Finv * v
            P_filt = P_pred - P_pred * transpose(Z) * Finv * Z * P_pred

            for i = 1:M
                att_store[i, t] = a_filt[i]
            end
            for j = 1:M, i = 1:M
                Ptt_store[i, j, t] = P_filt[i, j]
            end

            # Predict next
            a_pred = T_mat * a_filt
            P_pred = T_mat * P_filt * transpose(T_mat) + RQR
        end
    end

    const_term = -P * n_obs * log(ET(2π)) / 2

    return KalmanFilterResult(
        p,
        loglik + const_term,
        at_store,
        Pt_store,
        att_store,
        Ptt_store,
        vt_store,
        Ft_store,
        Kt_store,
        missing_mask,
    )
end

"""
    kalman_filter_static(p::KFParms, y, a1, P1) -> KalmanFilterResult

Run full Kalman filter using StaticArrays if dimensions are small enough.

Automatically converts inputs to StaticArrays when dimensions ≤ STATIC_THRESHOLD.
"""
function kalman_filter_static(
    p::KFParms,
    y::AbstractMatrix,
    a1::AbstractVector,
    P1::AbstractMatrix,
)
    m = length(a1)
    obs_dim = size(y, 1)
    r = size(p.Q, 1)

    if m ≤ STATIC_THRESHOLD && obs_dim ≤ STATIC_THRESHOLD && r ≤ STATIC_THRESHOLD
        p_static = KFParms_static(p.Z, p.H, p.T, p.R, p.Q)
        a1_static = to_static_if_small(a1)
        P1_static = to_static_if_small(P1)
        return kalman_filter(p_static, y, a1_static, P1_static)
    else
        return kalman_filter(p, y, a1, P1)
    end
end

"""
    kalman_filter_scalar(Z, H, T, R, Q, a1, P1, y) -> KalmanFilterResultScalar

Scalar version of full Kalman filter for univariate state-space models.
Handles missing observations marked as NaN.

For `n` observations, returns `n` time points:
- `at[t]` = E[αₜ | y₁:ₜ₋₁] (predicted state)
- `att[t]` = E[αₜ | y₁:ₜ] (filtered state)
"""
function kalman_filter_scalar(
    Z::Real,
    H::Real,
    Tmat::Real,
    R::Real,
    Q::Real,
    a1::Real,
    P1::Real,
    y::AbstractVector,
)
    n = length(y)
    ET = promote_type(
        typeof(Z),
        typeof(H),
        typeof(Tmat),
        typeof(R),
        typeof(Q),
        typeof(a1),
        typeof(P1),
    )

    # Predicted state storage: n entries
    at_store = Vector{ET}(undef, n)
    Pt_store = Vector{ET}(undef, n)

    # Filtered state storage: n entries
    att_store = Vector{ET}(undef, n)
    Ptt_store = Vector{ET}(undef, n)

    # Innovation storage: n entries
    vt_store = Vector{ET}(undef, n)
    Ft_store = Vector{ET}(undef, n)
    missing_mask = BitVector(undef, n)

    a_pred = convert(ET, a1)
    P_pred = convert(ET, P1)

    loglik = zero(ET)
    n_obs = 0

    for t = 1:n
        y_t = y[t]

        # Store predicted state
        at_store[t] = a_pred
        Pt_store[t] = P_pred

        if _has_missing(y_t)
            missing_mask[t] = true
            vt_store[t] = ET(NaN)
            Ft_store[t] = Z * P_pred * Z + H

            # For missing: filtered = predicted
            att_store[t] = a_pred
            Ptt_store[t] = P_pred

            # Propagate without update
            a_pred = Tmat * a_pred
            P_pred = Tmat * P_pred * Tmat + R * Q * R
        else
            missing_mask[t] = false
            n_obs += 1

            v = y_t - Z * a_pred
            F = Z * P_pred * Z + H
            Finv = one(ET) / F

            loglik += -ET(0.5) * (log(F) + v^2 * Finv)

            vt_store[t] = v
            Ft_store[t] = F

            # Filtered state
            a_filt = a_pred + P_pred * Z * Finv * v
            P_filt = P_pred - P_pred * Z * Finv * Z * P_pred

            att_store[t] = a_filt
            Ptt_store[t] = P_filt

            # Predict next state
            a_pred = Tmat * a_filt
            P_pred = Tmat * P_filt * Tmat + R * Q * R
        end
    end

    const_term = -n_obs * log(ET(2π)) / 2

    return KalmanFilterResultScalar{ET}(
        loglik + const_term,
        at_store,
        Pt_store,
        att_store,
        Ptt_store,
        vt_store,
        Ft_store,
        missing_mask,
    )
end

# ============================================
# Exact Diffuse Initialization (Durbin-Koopman)
# ============================================

"""
    _safe_inverse(F::AbstractMatrix, tol=1e-10) -> (flag, result)

Attempt to invert matrix F. Returns (1, F⁻¹) if successful, (0, F) if singular.

The flag indicates:
- 1: F was invertible (det(F) > tol)
- 0: F was singular or near-singular
"""
@inline function _safe_inverse(F::AbstractMatrix{T}, tol::Real = 1e-10) where {T}
    d = det(F)
    if abs(d) > tol
        return (1, inv(F))
    else
        return (0, F)  # Return F itself as placeholder (won't be used)
    end
end

@inline function _safe_inverse(F::Real, tol::Real = 1e-10)
    if abs(F) > tol
        return (1, one(F) / F)
    else
        return (0, F)
    end
end

"""
    kalman_loglik_diffuse(p::KFParms, y, a1, P1_star, P1_inf; tol=1e-8) -> loglik

Compute log-likelihood using exact diffuse initialization (Durbin-Koopman method).

The initial state covariance is P1 = P1_star + κ * P1_inf where κ → ∞.
The filter runs in two phases:
1. Diffuse period: While norm(Pinf) > tol, uses exact diffuse recursion
2. Non-diffuse period: Standard Kalman filter recursion

Only observations after the diffuse period contribute to the log-likelihood.

# Arguments
- `p::KFParms`: State space parameters (Z, H, T, R, Q)
- `y::AbstractMatrix`: Observations (p × n matrix)
- `a1::AbstractVector`: Initial state mean
- `P1_star::AbstractMatrix`: Finite part of initial covariance
- `P1_inf::AbstractMatrix`: Diffuse part of initial covariance (typically I or diagonal)
- `tol::Real=1e-8`: Tolerance for detecting end of diffuse period

# Returns
- `loglik::Real`: Log-likelihood (non-diffuse observations only)

# Example
```julia
# Local level model with exact diffuse initialization
p = KFParms([1.0;;], [σ²_obs;;], [1.0;;], [1.0;;], [σ²_state;;])
y = randn(1, 100)
a1 = [0.0]
P1_star = [0.0;;]  # No finite uncertainty initially
P1_inf = [1.0;;]   # Full diffuse on the level

ll = kalman_loglik_diffuse(p, y, a1, P1_star, P1_inf)
```
"""
function kalman_loglik_diffuse(
    p::KFParms,
    y::AbstractMatrix,
    a1::AbstractVector,
    P1_star::AbstractMatrix,
    P1_inf::AbstractMatrix;
    tol::Real = 1e-8,
)
    n = size(y, 2)
    obs_dim = size(y, 1)

    # Get element type for AD compatibility
    ET = promote_type(
        eltype(p.Z),
        eltype(p.H),
        eltype(p.T),
        eltype(p.R),
        eltype(p.Q),
        eltype(a1),
        eltype(P1_star),
        eltype(P1_inf),
    )

    # Initialize state
    a = Vector{ET}(a1)
    Pstar = Matrix{ET}(P1_star)
    Pinf = Matrix{ET}(P1_inf)

    # Precompute RQR'
    RQR = p.R * p.Q * p.R'

    # Log-likelihood accumulator
    loglik = zero(ET)
    n_obs = 0  # Count non-diffuse, non-missing observations

    # Track if diffuse period has ended
    diffuse_ended = false

    for t = 1:n
        y_t = y[:, t]

        # Check for missing observation
        if _has_missing(y_t)
            if diffuse_ended
                # Standard propagation
                a = p.T * a
                P = p.T * Pstar * p.T' + RQR
                Pstar = P
            else
                # Diffuse propagation (no measurement update)
                a = p.T * a
                Pinf = p.T * Pinf * p.T'
                Pstar = p.T * Pstar * p.T' + RQR
                # Check if diffuse period ends
                if norm(Pinf) <= tol
                    diffuse_ended = true
                end
            end
            continue
        end

        if diffuse_ended
            # === Standard Kalman filter step ===
            n_obs += 1
            v = y_t - p.Z * a
            F = p.Z * Pstar * p.Z' + p.H

            if obs_dim == 1
                F_val = F[1, 1]
                if F_val <= zero(ET)
                    return ET(-Inf)
                end
                Finv = one(ET) / F_val
                logdetF = log(F_val)
                quad_form = v[1]^2 * Finv
            else
                F_sym = Symmetric((F + F') / 2)
                chol_result = cholesky(F_sym; check = false)
                if !issuccess(chol_result)
                    return ET(-Inf)
                end
                Finv = inv(chol_result)
                logdetF = 2 * sum(log.(diag(chol_result.U)))
                quad_form = dot(v, Finv * v)
            end

            loglik += -ET(0.5) * (logdetF + quad_form)

            if !isfinite(loglik)
                return ET(-Inf)
            end

            # Kalman gain and state update
            K = p.T * Pstar * p.Z' * Finv
            a = p.T * a + K * v
            L = p.T - K * p.Z
            Pstar = L * Pstar * p.T' + RQR
        else
            # === Diffuse filter step ===
            v = y_t - p.Z * a
            Finf = p.Z * Pinf * p.Z'

            # Check if Finf is invertible
            (flag, Finf_inv) = _safe_inverse(Finf, tol)

            if flag == 1
                # Finf is invertible: exact diffuse update
                Kinf = p.T * Pinf * p.Z' * Finf_inv
                Linf = p.T - Kinf * p.Z
                Fstar = p.Z * Pstar * p.Z' + p.H
                Kstar = (p.T * Pstar * p.Z' + Kinf * Fstar) * Finf_inv

                # State update
                a = p.T * a + Kinf * v

                # Covariance updates
                Pinf_new = p.T * Pinf * Linf'
                Pstar_new = p.T * Pstar * Linf' + Kinf * Finf * Kstar' + RQR

                Pinf = Pinf_new
                Pstar = Pstar_new

                # No likelihood contribution during diffuse period with Finf invertible
            else
                # Finf is singular: use Fstar-based update
                Fstar = p.Z * Pstar * p.Z' + p.H

                if obs_dim == 1
                    Fstar_val = Fstar[1, 1]
                    if Fstar_val <= zero(ET)
                        return ET(-Inf)
                    end
                    Fstar_inv = one(ET) / Fstar_val
                    logdetFstar = log(Fstar_val)
                    quad_form = v[1]^2 * Fstar_inv
                else
                    Fstar_sym = Symmetric((Fstar + Fstar') / 2)
                    chol_result = cholesky(Fstar_sym; check = false)
                    if !issuccess(chol_result)
                        return ET(-Inf)
                    end
                    Fstar_inv = inv(chol_result)
                    logdetFstar = 2 * sum(log.(diag(chol_result.U)))
                    quad_form = dot(v, Fstar_inv * v)
                end

                # This observation contributes to likelihood
                n_obs += 1
                loglik += -ET(0.5) * (logdetFstar + quad_form)

                if !isfinite(loglik)
                    return ET(-Inf)
                end

                Kstar = p.T * Pstar * p.Z' * Fstar_inv
                Lstar = p.T - Kstar * p.Z

                # State update
                a = p.T * a + Kstar * v

                # Covariance updates
                Pinf = p.T * Pinf * p.T'
                Pstar = p.T * Pstar * Lstar' + RQR
            end

            # Check if diffuse period ends
            if norm(Pinf) <= tol
                diffuse_ended = true
            end
        end
    end

    # Add constant term
    const_term = -obs_dim * n_obs * log(ET(2π)) / 2

    return loglik + const_term
end

"""
    kalman_filter_diffuse(p::KFParms, y, a1, P1_star, P1_inf; tol=1e-8) -> DiffuseFilterResult

Run full Kalman filter with exact diffuse initialization.

Returns filtered and predicted states along with all intermediate quantities,
including the diffuse covariances (Pinf, Pstar) during the diffuse period.

# Arguments
- `p::KFParms`: State space parameters (Z, H, T, R, Q)
- `y::AbstractMatrix`: Observations (p × n matrix)
- `a1::AbstractVector`: Initial state mean
- `P1_star::AbstractMatrix`: Finite part of initial covariance
- `P1_inf::AbstractMatrix`: Diffuse part of initial covariance
- `tol::Real=1e-8`: Tolerance for detecting end of diffuse period

# Returns
- `DiffuseFilterResult`: Contains all filter outputs plus diffuse-specific quantities

# See Also
- `kalman_loglik_diffuse`: Log-likelihood only (faster if states not needed)
- `kalman_filter`: Standard filter without diffuse initialization
"""
function kalman_filter_diffuse(
    p::KFParms,
    y::AbstractMatrix,
    a1::AbstractVector,
    P1_star::AbstractMatrix,
    P1_inf::AbstractMatrix;
    tol::Real = 1e-8,
)
    n = size(y, 2)
    obs_dim = size(y, 1)
    state_dim = length(a1)

    ET = promote_type(
        eltype(p.Z),
        eltype(p.H),
        eltype(p.T),
        eltype(p.R),
        eltype(p.Q),
        eltype(a1),
        eltype(P1_star),
        eltype(P1_inf),
    )

    # Allocate storage
    at_store = Matrix{ET}(undef, state_dim, n)
    Pt_store = Array{ET}(undef, state_dim, state_dim, n)
    att_store = Matrix{ET}(undef, state_dim, n)
    Ptt_store = Array{ET}(undef, state_dim, state_dim, n)
    vt_store = Matrix{ET}(undef, obs_dim, n)
    Ft_store = Array{ET}(undef, obs_dim, obs_dim, n)
    Kt_store = Array{ET}(undef, state_dim, obs_dim, n)
    missing_mask = BitVector(undef, n)

    # Temporary storage for diffuse period (will resize at end)
    Pinf_list = Vector{Matrix{ET}}()
    Pstar_list = Vector{Matrix{ET}}()
    flag_list = Vector{Int}()

    # Initialize state
    a = Vector{ET}(a1)
    Pstar = Matrix{ET}(P1_star)
    Pinf = Matrix{ET}(P1_inf)

    # Precompute RQR'
    RQR = p.R * p.Q * p.R'

    loglik = zero(ET)
    n_obs = 0
    diffuse_ended = false
    d = 0  # Diffuse period length

    for t = 1:n
        y_t = y[:, t]

        # Store predicted state
        at_store[:, t] = a
        # During diffuse period, P = Pstar (finite part); after, P = Pstar
        Pt_store[:, :, t] = Pstar

        if _has_missing(y_t)
            missing_mask[t] = true
            vt_store[:, t] .= ET(NaN)
            Ft_store[:, :, t] = p.Z * Pstar * p.Z' + p.H
            Kt_store[:, :, t] .= zero(ET)
            att_store[:, t] = a
            Ptt_store[:, :, t] = Pstar

            if diffuse_ended
                a = p.T * a
                Pstar = p.T * Pstar * p.T' + RQR
            else
                # Store diffuse quantities
                push!(Pinf_list, copy(Pinf))
                push!(Pstar_list, copy(Pstar))
                push!(flag_list, -1)  # -1 indicates missing
                d += 1

                a = p.T * a
                Pinf = p.T * Pinf * p.T'
                Pstar = p.T * Pstar * p.T' + RQR

                if norm(Pinf) <= tol
                    diffuse_ended = true
                end
            end
            continue
        end

        missing_mask[t] = false

        if diffuse_ended
            # === Standard Kalman filter step ===
            n_obs += 1
            v = y_t - p.Z * a
            F = p.Z * Pstar * p.Z' + p.H

            if obs_dim == 1
                F_val = F[1, 1]
                Finv = fill(one(ET) / F_val, 1, 1)
                logdetF = log(F_val)
                quad_form = v[1]^2 / F_val
            else
                Finv = inv(F)
                logdetF = logdet(F)
                quad_form = dot(v, Finv * v)
            end

            loglik += -ET(0.5) * (logdetF + quad_form)

            K = p.T * Pstar * p.Z' * Finv

            vt_store[:, t] = v
            Ft_store[:, :, t] = F
            Kt_store[:, :, t] = K

            # Filtered state
            a_filt = a + Pstar * p.Z' * Finv * v
            P_filt = Pstar - Pstar * p.Z' * Finv * p.Z * Pstar

            att_store[:, t] = a_filt
            Ptt_store[:, :, t] = P_filt

            # Predict next
            a = p.T * a_filt
            Pstar = p.T * P_filt * p.T' + RQR
        else
            # === Diffuse filter step ===
            v = y_t - p.Z * a
            Finf = p.Z * Pinf * p.Z'

            (flag, Finf_inv) = _safe_inverse(Finf, tol)

            # Store diffuse quantities
            push!(Pinf_list, copy(Pinf))
            push!(Pstar_list, copy(Pstar))
            push!(flag_list, flag)
            d += 1

            if flag == 1
                # Finf invertible
                Kinf = p.T * Pinf * p.Z' * Finf_inv
                Linf = p.T - Kinf * p.Z
                Fstar = p.Z * Pstar * p.Z' + p.H
                Kstar = (p.T * Pstar * p.Z' + Kinf * Fstar) * Finf_inv

                vt_store[:, t] = v
                Ft_store[:, :, t] = Finf  # Store Finf during diffuse
                Kt_store[:, :, t] = Kinf

                # Filtered state (use Kinf for gain in filtered estimate)
                # During diffuse: a_filt = a + Pinf * Z' * Finf_inv * v
                a_filt = a + Pinf * p.Z' * Finf_inv * v
                # P_filt approximation during diffuse
                P_filt = Pstar - Pstar * p.Z' * Finf_inv * p.Z * Pstar

                att_store[:, t] = a_filt
                Ptt_store[:, :, t] = P_filt

                a = p.T * a + Kinf * v
                Pinf = p.T * Pinf * Linf'
                Pstar = p.T * Pstar * Linf' + Kinf * Finf * Kstar' + RQR

            else
                # Finf singular, use Fstar
                Fstar = p.Z * Pstar * p.Z' + p.H

                if obs_dim == 1
                    Fstar_val = Fstar[1, 1]
                    Fstar_inv = fill(one(ET) / Fstar_val, 1, 1)
                    logdetFstar = log(Fstar_val)
                    quad_form = v[1]^2 / Fstar_val
                else
                    Fstar_inv = inv(Fstar)
                    logdetFstar = logdet(Fstar)
                    quad_form = dot(v, Fstar_inv * v)
                end

                n_obs += 1
                loglik += -ET(0.5) * (logdetFstar + quad_form)

                Kstar = p.T * Pstar * p.Z' * Fstar_inv
                Lstar = p.T - Kstar * p.Z

                vt_store[:, t] = v
                Ft_store[:, :, t] = Fstar
                Kt_store[:, :, t] = Kstar

                a_filt = a + Pstar * p.Z' * Fstar_inv * v
                P_filt = Pstar - Pstar * p.Z' * Fstar_inv * p.Z * Pstar

                att_store[:, t] = a_filt
                Ptt_store[:, :, t] = P_filt

                a = p.T * a + Kstar * v
                Pinf = p.T * Pinf * p.T'
                Pstar = p.T * Pstar * Lstar' + RQR
            end

            if norm(Pinf) <= tol
                diffuse_ended = true
            end
        end
    end

    # Convert diffuse storage to arrays
    if d > 0
        Pinf_store = Array{ET}(undef, state_dim, state_dim, d)
        Pstar_store = Array{ET}(undef, state_dim, state_dim, d)
        for i = 1:d
            Pinf_store[:, :, i] = Pinf_list[i]
            Pstar_store[:, :, i] = Pstar_list[i]
        end
    else
        Pinf_store = Array{ET}(undef, state_dim, state_dim, 0)
        Pstar_store = Array{ET}(undef, state_dim, state_dim, 0)
    end

    const_term = -obs_dim * n_obs * log(ET(2π)) / 2

    return DiffuseFilterResult(
        p,
        loglik + const_term,
        d,
        at_store,
        Pt_store,
        att_store,
        Ptt_store,
        Pinf_store,
        Pstar_store,
        vt_store,
        Ft_store,
        Kt_store,
        flag_list,
        missing_mask,
    )
end

# ============================================
# Unified API: 5-arg signatures trigger exact diffuse
# ============================================

"""
    kalman_loglik(p::KFParms, y, a1, P1_star, P1_inf; tol=1e-8) -> loglik

Compute log-likelihood using exact diffuse initialization (Durbin-Koopman method).

This is the unified API: passing 5 positional arguments (with P1_inf as the 5th)
triggers exact diffuse initialization. For standard (non-diffuse) filtering,
use the 4-arg version `kalman_loglik(p, y, a1, P1)`.

The initial state covariance is P1 = P1_star + κ * P1_inf where κ → ∞.
Only observations after the diffuse period contribute to the log-likelihood.

# Arguments
- `p::KFParms`: State space parameters (Z, H, T, R, Q)
- `y::AbstractMatrix`: Observations (p × n matrix)
- `a1::AbstractVector`: Initial state mean
- `P1_star::AbstractMatrix`: Finite part of initial covariance
- `P1_inf::AbstractMatrix`: Diffuse part of initial covariance (typically I or diagonal)
- `tol::Real=1e-8`: Tolerance for detecting end of diffuse period

# Example
```julia
# Local level model with exact diffuse initialization
p = KFParms([1.0;;], [σ²_obs;;], [1.0;;], [1.0;;], [σ²_state;;])
y = randn(1, 100)
a1 = [0.0]
P1_star = [0.0;;]  # No finite uncertainty initially
P1_inf = [1.0;;]   # Full diffuse on the level

ll = kalman_loglik(p, y, a1, P1_star, P1_inf)
```

See also: [`kalman_filter`](@ref) (5-arg version for full filter output)
"""
function kalman_loglik(
    p::KFParms,
    y::AbstractMatrix,
    a1::AbstractVector,
    P1_star::AbstractMatrix,
    P1_inf::AbstractMatrix;
    tol::Real = 1e-8,
)
    return kalman_loglik_diffuse(p, y, a1, P1_star, P1_inf; tol = tol)
end

"""
    kalman_filter(p::KFParms, y, a1, P1_star, P1_inf; tol=1e-8) -> DiffuseFilterResult

Run full Kalman filter with exact diffuse initialization.

This is the unified API: passing 5 positional arguments (with P1_inf as the 5th)
triggers exact diffuse initialization. For standard (non-diffuse) filtering,
use the 4-arg version `kalman_filter(p, y, a1, P1)`.

Returns a `DiffuseFilterResult` containing all filter outputs plus diffuse-specific
quantities (Pinf, Pstar during diffuse period, diffuse flags).

# Arguments
- `p::KFParms`: State space parameters (Z, H, T, R, Q)
- `y::AbstractMatrix`: Observations (p × n matrix)
- `a1::AbstractVector`: Initial state mean
- `P1_star::AbstractMatrix`: Finite part of initial covariance
- `P1_inf::AbstractMatrix`: Diffuse part of initial covariance
- `tol::Real=1e-8`: Tolerance for detecting end of diffuse period

# Example
```julia
p = KFParms([1.0;;], [100.0;;], [1.0;;], [1.0;;], [10.0;;])
y = randn(1, 50)
a1 = [0.0]
P1_star = [0.0;;]
P1_inf = [1.0;;]

result = kalman_filter(p, y, a1, P1_star, P1_inf)
d = diffuse_period(result)  # Number of diffuse observations
```

See also: [`kalman_loglik`](@ref) (5-arg version for log-likelihood only)
"""
function kalman_filter(
    p::KFParms,
    y::AbstractMatrix,
    a1::AbstractVector,
    P1_star::AbstractMatrix,
    P1_inf::AbstractMatrix;
    tol::Real = 1e-8,
)
    return kalman_filter_diffuse(p, y, a1, P1_star, P1_inf; tol = tol)
end
