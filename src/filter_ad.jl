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
NaN marks a missing observation, one element at a time. Where only some rows of a
period are present, the measurement update runs on the reduced system formed by
those rows. Where every row is NaN the update is skipped and the state propagates
through the transition equation alone:
    a_{t+1} = T * a_t
    P_{t+1} = T * P_t * T' + R * Q * R'
which is equivalent to infinite observation noise for that period.

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
    _all_missing(y_t::AbstractVector) -> Bool

Whether every element of the observation vector is NaN, so the period carries no
information and the measurement update is skipped entirely.
"""
@inline _all_missing(y_t::AbstractVector) = all(isnan, y_t)

@inline _all_missing(y_t::Real) = isnan(y_t)

"""
    _observed_row_indices(y_t) -> Vector{Int}

The indices of the non-NaN elements of `y_t`.

The return type is concrete rather than a range-or-vector union: the callers feed
it straight into the matrices of a long filter body, and a union there propagates
through every downstream expression and makes type inference intractable.
"""
function _observed_row_indices(y_t::AbstractVector)
    rows = Vector{Int}(undef, count(!isnan, y_t))
    k = 0
    for (i, yi) in pairs(y_t)
        if !isnan(yi)
            k += 1
            rows[k] = i
        end
    end
    return rows
end

"""
    _select_rows(A, rows) -> Matrix

Rows `rows` of `A`. Always materializes, so the result type does not depend on
whether the period was fully observed; a union there would defeat inference in
the filter bodies that consume it.
"""
_select_rows(A::AbstractMatrix, rows) = A[rows, :]

"""
    _select_rows_cols(A, rows) -> Matrix

The principal submatrix of `A` on `rows`, materialized for the same reason as
[`_select_rows`](@ref).
"""
_select_rows_cols(A::AbstractMatrix, rows) = A[rows, rows]

"""
    _period_obs_rows(observed_mask, vt, t) -> AbstractVector{Int}

The rows observed in period `t`. Taken from `observed_mask` when one is supplied;
otherwise every row of `vt` is assumed observed, which is what a filter result
without a per-element mask records.
"""
function _period_obs_rows(observed_mask, vt::AbstractMatrix, t::Integer)
    observed_mask === nothing && return collect(axes(vt, 1))
    return findall(view(observed_mask, :, t))
end

"""
    _period_obs_dim(observed_mask, vt, t) -> Int

The number of rows observed in period `t`.
"""
function _period_obs_dim(observed_mask, vt::AbstractMatrix, t::Integer)
    observed_mask === nothing && return size(vt, 1)
    return count(view(observed_mask, :, t))
end

# The first period of `y` in which some but not all rows are observed, or `nothing`.
function _first_partial_period(y::AbstractMatrix)
    for t in axes(y, 2)
        n_obs = count(!isnan, view(y, :, t))
        n_obs != 0 && n_obs != size(y, 1) && return t
    end
    return nothing
end

"""
    _require_no_partial_periods(y, caller::AbstractString, hint::AbstractString = "")

Throw an `ArgumentError` if any period of `y` is partially observed, naming the
first such period and the rows involved, followed by `hint`.

Paths that handle missingness one period at a time rather than one element at a
time call this on entry, so a partially observed panel fails immediately instead
of silently discarding the observed rows of that period.
"""
function _require_no_partial_periods(y::AbstractMatrix, caller::AbstractString,
        hint::AbstractString = "")
    t = _first_partial_period(y)
    t === nothing && return nothing
    observed = findall(!isnan, view(y, :, t))
    throw(ArgumentError(
        "$caller does not support partially observed periods, but in period $t only " *
        "rows $observed of $(size(y, 1)) are observed." *
        (isempty(hint) ? "" : " " * hint)))
end

# Static-array system matrices converted to ordinary arrays, for the per-element
# missing-data filters.
function _dynamic_kfparms(p::KFParms)
    KFParms(Matrix(p.Z), Matrix(p.H), Matrix(p.T),
        Matrix(p.R), Matrix(p.Q))
end

"""
    _observed_rows!(rows, y_t) -> Int

Write the indices of the non-NaN elements of `y_t` into the leading entries of
`rows` and return how many there were. `rows` must be at least as long as `y_t`.
"""
function _observed_rows!(rows::AbstractVector{Int}, y_t::AbstractVector)
    Base.require_one_based_indexing(rows)
    length(rows) ≥ length(y_t) || throw(DimensionMismatch(
        "row buffer holds $(length(rows)) entries; need $(length(y_t))"))
    k = 0
    for (i, yi) in pairs(y_t)
        if !isnan(yi)
            k += 1
            rows[k] = i
        end
    end
    return k
end

# ============================================
# Element type helper for AD compatibility
# ============================================

@inline _filter_eltype(p::KFParms,
    a1,
    P1) = promote_type(
    eltype(p.Z), eltype(p.H), eltype(p.T), eltype(p.R), eltype(p.Q), eltype(a1), eltype(P1))

@inline _filter_eltype(p::KFParms,
    a1,
    P1_star,
    P1_inf) = promote_type(eltype(p.Z), eltype(p.H), eltype(p.T), eltype(p.R),
    eltype(p.Q), eltype(a1), eltype(P1_star), eltype(P1_inf))

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
NaN marks a missing observation, one element at a time. In a period where only
some rows are present the measurement update runs on the reduced system formed by
those rows, so the observed rows still inform the state. A period in which every
row is NaN carries no information and the update is skipped:
    a_{t+1} = T * a_t
    P_{t+1} = T * P_t * T' + R * Q * R'
"""
function kalman_loglik(
        p::KFParms,
        y::AbstractMatrix,
        a1::AbstractVector,
        P1::AbstractMatrix
)
    y = _nan_obs(y)
    n = size(y, 2)
    ET = _filter_eltype(p, a1, P1)

    a = Vector{ET}(a1)
    P = Matrix{ET}(P1)
    # RQR' is constant across iterations.
    RQR = p.R * p.Q * transpose(p.R)
    Tt = transpose(p.T)
    loglik = zero(ET)
    n_obs_rows = 0

    for t in 1:n
        y_t = view(y, :, t)

        if _all_missing(y_t)
            a = p.T * a
            P = p.T * P * Tt + RQR
            continue
        end

        # Restrict to the observed rows. When every row is present these are the
        # full Z, H and y_t and the arithmetic is unchanged.
        rows = _observed_row_indices(y_t)
        obs_dim = length(rows)
        n_obs_rows += obs_dim
        Zr = _select_rows(p.Z, rows)
        Hr = _select_rows_cols(p.H, rows)
        Zrt = transpose(Zr)

        v = y_t[rows] - Zr * a
        F = Zr * P * Zrt + Hr

        if obs_dim == 1
            F_val = F[1, 1]
            if F_val <= zero(ET)
                return ET(-Inf)
            end
            Finv_val = one(ET) / F_val
            logdetF = log(F_val)
            quad_form = v[1]^2 * Finv_val
            loglik += -ET(0.5) * (logdetF + quad_form)
            if !isfinite(loglik)
                return ET(-Inf)
            end
            PZt = P * Zrt
            K = p.T * PZt * Finv_val
            a = p.T * a + K * v
            # P_filt = P - PZt * Finv_val * (PZt)'
            P = p.T * (P - PZt * (Finv_val * transpose(PZt))) * Tt + RQR
        else
            F_sym = Symmetric((F + F') / 2)
            chol_result = cholesky(F_sym; check = false)
            if !issuccess(chol_result)
                return ET(-Inf)
            end
            # log|F| = 2 Σ log L[i,i]; avoid diag() allocation.
            logdetF = zero(ET)
            L_tri = chol_result.L
            @inbounds for i in 1:obs_dim
                logdetF += log(L_tri[i, i])
            end
            logdetF += logdetF

            # Solve Finv*v once via Cholesky — no dense inverse.
            Finv_v = chol_result \ v
            quad_form = dot(v, Finv_v)
            loglik += -ET(0.5) * (logdetF + quad_form)
            if !isfinite(loglik)
                return ET(-Inf)
            end

            # M = P * Z' * F^{-1} (m×pt) from Cholesky solves on Z*P.
            PZt = P * Zrt
            M = transpose(chol_result \ transpose(PZt))
            K = p.T * M
            a = p.T * a + K * v
            # P_filt = P - M * (Z*P)
            P = p.T * (P - M * (Zr * P)) * Tt + RQR
        end
    end

    # The Gaussian constant counts every observed scalar across the sample.
    const_term = -n_obs_rows * log(ET(2π)) / 2
    return loglik + const_term
end

# ============================================
# Static specialization for kalman_loglik
# ============================================

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
        p::KFParms{<:SMatrix{P, M}, <:SMatrix{P, P}, <:SMatrix{M, M},
            <:SMatrix{M, R}, <:SMatrix{R, R}},
        y::AbstractMatrix,
        a1::SVector{M},
        P1::SMatrix{M, M}
) where {P, M, R}
    y = _nan_obs(y)
    # The static kernel drops whole periods; partially observed ones need the
    # per-element filter.
    _first_partial_period(y) === nothing ||
        return kalman_loglik(_dynamic_kfparms(p), y, Vector(a1), Matrix(P1))
    n = size(y, 2)
    ET = _filter_eltype(p, a1, P1)

    # Initialize state as static types
    a = SVector{M, ET}(a1)
    P_state = SMatrix{M, M, ET}(P1)

    # Static system matrices (ensure correct element type for AD)
    Z = SMatrix{P, M, ET}(p.Z)
    H = SMatrix{P, P, ET}(p.H)
    T_mat = SMatrix{M, M, ET}(p.T)
    R_mat = SMatrix{M, R, ET}(p.R)
    Q = SMatrix{R, R, ET}(p.Q)

    # Precompute R*Q*R' (constant across iterations)
    RQR = R_mat * Q * transpose(R_mat)

    loglik = zero(ET)
    n_obs = 0

    @inbounds for t in 1:n
        # Extract observation as SVector
        y_t = SVector{P, ET}(ntuple(i -> ET(y[i, t]), Val(P)))

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
            Finv = SMatrix{1, 1, ET}((one(ET) / F_val,))
            logdetF = log(F_val)
            quad_form = v[1]^2 / F_val
        else
            # Multivariate: use Cholesky
            F_sym = Symmetric(SMatrix{P, P, ET}((F + transpose(F)) / 2))
            chol_result = cholesky(F_sym; check = false)
            if !issuccess(chol_result)
                return ET(-Inf)
            end
            # For static matrices, inv is efficient
            Finv = SMatrix{P, P, ET}(inv(chol_result))
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

# A system whose matrices are all scalars: one state and one series.
const _ScalarKFParms = KFParms{<:Real, <:Real, <:Real, <:Real, <:Real}

# The series of a one-series observation array, as a vector.
_scalar_series(y::AbstractVector) = y
function _scalar_series(y::AbstractMatrix)
    size(y, 1) == 1 || throw(DimensionMismatch(
        "a KFParms with scalar fields has one series, but y has $(size(y, 1)) rows"))
    return vec(y)
end

"""
    kalman_loglik(p::KFParms, y, a1::Real, P1::Real) -> loglik

Log-likelihood of a model with one state and one series, given as a `KFParms` whose
fields `Z`, `H`, `T`, `R` and `Q` are numbers. `y` is a vector or a 1 × n matrix.
This runs a scalar kernel that avoids matrix arithmetic. AD-compatible.

# Example
```julia
p = KFParms(1.0, 15099.0, 1.0, 1.0, 1469.1)
kalman_loglik(p, nile, 0.0, 1e7)
```
"""
function kalman_loglik(p::_ScalarKFParms, y::AbstractVecOrMat, a1::Real, P1::Real)
    y = _scalar_series(_nan_obs(y))
    return _kalman_loglik_scalar(p.Z, p.H, p.T, p.R, p.Q, a1, P1, y)
end

function _kalman_loglik_scalar(
        Z::Real,
        H::Real,
        Tmat::Real,
        R::Real,
        Q::Real,
        a1::Real,
        P1::Real,
        y::AbstractVector
)
    y = _nan_obs(y)
    n = length(y)
    ET = promote_type(
        typeof(Z), typeof(H), typeof(Tmat), typeof(R), typeof(Q), typeof(a1), typeof(P1))

    a = convert(ET, a1)
    P = convert(ET, P1)
    loglik = zero(ET)
    n_obs = 0

    for t in 1:n
        y_t = y[t]

        if _has_missing(y_t)
            a = Tmat * a
            P = Tmat * P * Tmat + R * Q * R
            continue
        end

        n_obs += 1

        v = y_t - Z * a
        F = Z * P * Z + H

        if F <= zero(ET)
            return ET(-Inf)
        end

        Finv = one(ET) / F
        loglik += -ET(0.5) * (log(F) + v^2 * Finv)

        if !isfinite(loglik)
            return ET(-Inf)
        end

        K = Tmat * P * Z * Finv
        a = Tmat * a + K * v
        L = Tmat - K * Z
        P = L * P * Tmat + R * Q * R
    end

    const_term = -n_obs * log(ET(2π)) / 2
    return loglik + const_term
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
- `missing_mask`: BitVector, true where a period has no observed row (length n)
- `observed_mask`: BitMatrix (p × n), true where an individual value is present

Use accessor methods: `predicted_states`, `filtered_states`, `predicted_states_cov`,
`filtered_states_cov`, `prediction_errors`, `prediction_errors_cov`,
`kalman_gains`, `loglikelihood`.

# Missing Data
NaN marks a missing observation, one element at a time. Where only some rows of a
period are present, the update runs on the reduced system formed by those rows and
`vt`, `Ft` and `Kt` carry pₜ-sized blocks in their leading rows and columns for
that period. Where every row is NaN the update is skipped.
"""
function kalman_filter(
        p::KFParms,
        y::AbstractMatrix,
        a1::AbstractVector,
        P1::AbstractMatrix
)
    y = _nan_obs(y)
    n = size(y, 2)
    obs_dim = size(y, 1)
    state_dim = length(a1)
    ET = _filter_eltype(p, a1, P1)

    at_store = fill(ET(NaN), state_dim, n)
    Pt_store = fill(ET(NaN), state_dim, state_dim, n)
    att_store = fill(ET(NaN), state_dim, n)
    Ptt_store = fill(ET(NaN), state_dim, state_dim, n)
    vt_store = fill(ET(NaN), obs_dim, n)
    Ft_store = fill(ET(NaN), obs_dim, obs_dim, n)
    Kt_store = fill(ET(NaN), state_dim, obs_dim, n)
    missing_mask = falses(n)
    observed_mask = falses(obs_dim, n)

    a_pred = Vector{ET}(a1)
    P_pred = Matrix{ET}(P1)
    loglik = zero(ET)
    n_obs_rows = 0

    for t in 1:n
        y_t = y[:, t]

        at_store[:, t] = a_pred
        Pt_store[:, :, t] = P_pred

        for i in 1:obs_dim
            observed_mask[i, t] = !isnan(y_t[i])
        end

        if _all_missing(y_t)
            missing_mask[t] = true
            vt_store[:, t] .= ET(NaN)
            Ft_store[:, :, t] = p.Z * P_pred * p.Z' + p.H
            Kt_store[:, :, t] .= zero(ET)
            att_store[:, t] = a_pred
            Ptt_store[:, :, t] = P_pred
            a_pred = p.T * a_pred
            P_pred = p.T * P_pred * p.T' + p.R * p.Q * p.R'
        else
            missing_mask[t] = false

            # Restrict to the observed rows; the fully observed case selects all
            # of them and the arithmetic below is unchanged.
            rows = _observed_row_indices(y_t)
            pt = length(rows)
            n_obs_rows += pt
            Zr = _select_rows(p.Z, rows)
            Hr = _select_rows_cols(p.H, rows)

            v = y_t[rows] - Zr * a_pred
            F = Zr * P_pred * Zr' + Hr

            if pt == 1
                F_val = F[1, 1]
                if F_val <= zero(ET)
                    return KalmanFilterResult(
                        p, ET(-Inf), at_store, Pt_store, att_store, Ptt_store,
                        vt_store, Ft_store, Kt_store, missing_mask, observed_mask, t)
                end
                Finv = fill(one(ET) / F_val, 1, 1)
                logdetF = log(F_val)
                quad_form = v[1]^2 / F_val
            else
                F_sym = Symmetric((F + F') / 2)
                chol_result = cholesky(F_sym; check = false)
                if !issuccess(chol_result)
                    return KalmanFilterResult(
                        p, ET(-Inf), at_store, Pt_store, att_store, Ptt_store,
                        vt_store, Ft_store, Kt_store, missing_mask, observed_mask, t)
                end
                Finv = inv(chol_result)
                logdetF = 2 * sum(log.(diag(chol_result.U)))
                quad_form = dot(v, Finv * v)
            end

            loglik += -ET(0.5) * (logdetF + quad_form)

            PZt = P_pred * Zr'
            K = p.T * PZt * Finv

            # Reduced-system quantities occupy the leading rows and columns; the
            # rest of the slice marks rows that were not observed.
            vt_store[:, t] .= ET(NaN)
            vt_store[1:pt, t] = v
            Ft_store[:, :, t] .= ET(NaN)
            Ft_store[1:pt, 1:pt, t] = F
            Kt_store[:, :, t] .= zero(ET)
            Kt_store[:, 1:pt, t] = K

            a_filt = a_pred + PZt * Finv * v
            P_filt = P_pred - PZt * Finv * Zr * P_pred
            att_store[:, t] = a_filt
            Ptt_store[:, :, t] = P_filt

            a_pred = p.T * a_filt
            P_pred = p.T * P_filt * p.T' + p.R * p.Q * p.R'
        end
    end

    # The Gaussian constant counts every observed scalar across the sample.
    const_term = -n_obs_rows * log(ET(2π)) / 2

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
        observed_mask
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
        p::KFParms{<:SMatrix{P, M}, <:SMatrix{P, P}, <:SMatrix{M, M},
            <:SMatrix{M, R}, <:SMatrix{R, R}},
        y::AbstractMatrix,
        a1::SVector{M},
        P1::SMatrix{M, M}
) where {P, M, R}
    y = _nan_obs(y)
    # The static kernel drops whole periods; partially observed ones need the
    # per-element filter.
    _first_partial_period(y) === nothing ||
        return kalman_filter(_dynamic_kfparms(p), y, Vector(a1), Matrix(P1))
    n = size(y, 2)
    ET = _filter_eltype(p, a1, P1)

    # Output storage (heap-allocated, filled with static values)
    at_store = fill(ET(NaN), M, n)
    Pt_store = fill(ET(NaN), M, M, n)
    att_store = fill(ET(NaN), M, n)
    Ptt_store = fill(ET(NaN), M, M, n)
    vt_store = fill(ET(NaN), P, n)
    Ft_store = fill(ET(NaN), P, P, n)
    Kt_store = fill(ET(NaN), M, P, n)
    missing_mask = falses(n)

    Z = SMatrix{P, M, ET}(p.Z)
    H = SMatrix{P, P, ET}(p.H)
    T_mat = SMatrix{M, M, ET}(p.T)
    R_mat = SMatrix{M, R, ET}(p.R)
    Q = SMatrix{R, R, ET}(p.Q)
    RQR = R_mat * Q * transpose(R_mat)

    a_pred = SVector{M, ET}(a1)
    P_pred = SMatrix{M, M, ET}(P1)
    loglik = zero(ET)
    n_obs = 0

    @inbounds for t in 1:n
        y_t = SVector{P, ET}(ntuple(i -> ET(y[i, t]), Val(P)))

        at_store[:, t] .= a_pred
        Pt_store[:, :, t] .= P_pred

        if _has_missing(y_t)
            missing_mask[t] = true
            vt_store[:, t] .= ET(NaN)
            Ft_store[:, :, t] .= Z * P_pred * transpose(Z) + H
            Kt_store[:, :, t] .= zero(ET)
            att_store[:, t] .= a_pred
            Ptt_store[:, :, t] .= P_pred
            a_pred = T_mat * a_pred
            P_pred = T_mat * P_pred * transpose(T_mat) + RQR
        else
            missing_mask[t] = false
            n_obs += 1

            v = y_t - Z * a_pred
            F = Z * P_pred * transpose(Z) + H

            if P == 1
                F_val = F[1, 1]
                if F_val <= zero(ET)
                    return KalmanFilterResult(
                        p, ET(-Inf), at_store, Pt_store, att_store, Ptt_store,
                        vt_store, Ft_store, Kt_store, missing_mask; failed_at = t)
                end
                Finv = SMatrix{1, 1, ET}((one(ET) / F_val,))
                logdetF = log(F_val)
                quad_form = v[1]^2 / F_val
            else
                F_sym = Symmetric(SMatrix{P, P, ET}((F + transpose(F)) / 2))
                chol_result = cholesky(F_sym; check = false)
                if !issuccess(chol_result)
                    return KalmanFilterResult(
                        p, ET(-Inf), at_store, Pt_store, att_store, Ptt_store,
                        vt_store, Ft_store, Kt_store, missing_mask; failed_at = t)
                end
                Finv = SMatrix{P, P, ET}(inv(chol_result))
                logdetF = 2 * sum(log.(diag(chol_result.U)))
                quad_form = dot(v, Finv * v)
            end

            loglik += -ET(0.5) * (logdetF + quad_form)

            K = T_mat * P_pred * transpose(Z) * Finv
            vt_store[:, t] .= v
            Ft_store[:, :, t] .= F
            Kt_store[:, :, t] .= K

            a_filt = a_pred + P_pred * transpose(Z) * Finv * v
            P_filt = P_pred - P_pred * transpose(Z) * Finv * Z * P_pred
            att_store[:, t] .= a_filt
            Ptt_store[:, :, t] .= P_filt

            a_pred = T_mat * a_filt
            P_pred = T_mat * P_filt * transpose(T_mat) + RQR
        end
    end

    const_term = -P * n_obs * log(ET(2π)) / 2
    return KalmanFilterResult(p, loglik + const_term, at_store, Pt_store, att_store,
        Ptt_store, vt_store, Ft_store, Kt_store, missing_mask)
end

"""
    kalman_filter(p::KFParms, y, a1::Real, P1::Real) -> KalmanFilterResult

Filter a model with one state and one series, given as a `KFParms` whose fields are
numbers. `y` is a vector or a 1 × n matrix. This runs a scalar kernel; the result is an
ordinary `KalmanFilterResult` with one row (`at` is 1 × n, `Pt` is 1 × 1 × n), so every
accessor, `kalman_smoother` and the plot recipes work on it.
"""
function kalman_filter(p::_ScalarKFParms, y::AbstractVecOrMat, a1::Real, P1::Real)
    y = _scalar_series(_nan_obs(y))
    return _kalman_filter_scalar(p, a1, P1, y)
end

function _kalman_filter_scalar(p::_ScalarKFParms, a1::Real, P1::Real, y::AbstractVector)
    Z, H, Tmat, R, Q = p.Z, p.H, p.T, p.R, p.Q
    n = length(y)
    ET = promote_type(
        typeof(Z),
        typeof(H),
        typeof(Tmat),
        typeof(R),
        typeof(Q),
        typeof(a1),
        typeof(P1)
    )

    # Predicted state storage: n entries
    at_store = fill(ET(NaN), n)
    Pt_store = fill(ET(NaN), n)

    # Filtered state storage: n entries
    att_store = fill(ET(NaN), n)
    Ptt_store = fill(ET(NaN), n)

    # Innovation storage: n entries
    vt_store = fill(ET(NaN), n)
    Ft_store = fill(ET(NaN), n)
    Kt_store = fill(ET(NaN), n)
    missing_mask = falses(n)

    # One-row arrays in the layout of KalmanFilterResult
    result(loglik, failed_at) = KalmanFilterResult(p, loglik, reshape(at_store, 1, n),
        reshape(Pt_store, 1, 1, n), reshape(att_store, 1, n), reshape(Ptt_store, 1, 1, n),
        reshape(vt_store, 1, n), reshape(Ft_store, 1, 1, n), reshape(Kt_store, 1, 1, n),
        missing_mask; failed_at = failed_at)

    a_pred = convert(ET, a1)
    P_pred = convert(ET, P1)

    loglik = zero(ET)
    n_obs = 0

    for t in 1:n
        y_t = y[t]

        # Store predicted state
        at_store[t] = a_pred
        Pt_store[t] = P_pred

        if _has_missing(y_t)
            missing_mask[t] = true
            Kt_store[t] = zero(ET)

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
            if F <= zero(ET)
                return result(ET(-Inf), t)
            end
            Finv = one(ET) / F

            loglik += -ET(0.5) * (log(F) + v^2 * Finv)

            vt_store[t] = v
            Ft_store[t] = F
            Kt_store[t] = Tmat * P_pred * Z * Finv

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

    return result(loglik + const_term, 0)
end

# ============================================
# Exact Diffuse Initialization (Durbin-Koopman)
# ============================================

"""
    _diffuse_step_flag(Finf, tol) -> flag

Classify a diffuse-period step from `Finf = Z_t Pinf Z_t'`: `1` when Finf is
invertible, `0` when it is zero (no observed row loads on a diffuse direction) and
`2` when it is singular but nonzero, which needs the univariate treatment of
[`_univariate_diffuse_update!`](@ref).
"""
function _diffuse_step_flag(Finf::AbstractMatrix, tol::Real)
    abs(det(Finf)) > tol && return 1
    return maximum(abs, Finf) > tol ? 2 : 0
end

"""
    _univariate_diffuse_update!(δ, G, Pstar, Pinf, Zs, w, h, Hr, C, Minf, Mstar, tol)
        -> (loglik, nrows, ok)

Process one period of the diffuse filter row by row (Koopman & Durbin 2000;
Durbin & Koopman 2012, §6.4). Used when `Finf = Z_t Pinf Z_t'` is singular but
nonzero, where the multivariate step cannot separate the diffuse directions from
the finite ones.

On entry `Zs` holds the observed rows `Z_t` (pₜ × m), `w` the innovation
`v = y_t - Z_t a` and `Hr` the observation covariance `H_t`. When `H_t` is not
diagonal, the rows are first transformed by the unit lower-triangular factor of
`H_t = C D C'`, so the transformed rows have independent errors with variances
`h = diag(D)`; the transform has unit determinant and leaves the likelihood
unchanged.

Row `i` then has `Finf,ᵢ = zᵢ Pinf zᵢ'` and `Fstar,ᵢ = zᵢ Pstar zᵢ' + hᵢ`. When
`Finf,ᵢ > tol` the row takes the exact diffuse update and adds nothing to the
likelihood; otherwise it takes the standard update on `Pstar` and adds its
Gaussian term (without the 2π constant, which the caller adds for `nrows`
scalars).

On exit `Pstar` and `Pinf` hold the finite and diffuse parts of the filtered
covariance, `δ = a_{t|t} - a` and `G` (m × pₜ) is the gain with `δ = G v`. `ok`
is `false` when some `Fstar,ᵢ` is negative.
"""
function _univariate_diffuse_update!(δ, G, Pstar, Pinf, Zs, w, h, Hr, C, Minf, Mstar, tol)
    pt, m = size(Zs)
    ET = eltype(Pstar)

    # Decorrelate the rows: H_t = C D C' with C unit lower triangular.
    if isdiag(Hr)
        for i in 1:pt
            h[i] = Hr[i, i]
        end
    else
        fill!(C, zero(ET))
        for j in 1:pt
            s = Hr[j, j]
            for k in 1:(j - 1)
                s -= C[j, k]^2 * h[k]
            end
            h[j] = s
            C[j, j] = one(ET)
            for i in (j + 1):pt
                s = Hr[i, j]
                for k in 1:(j - 1)
                    s -= C[i, k] * C[j, k] * h[k]
                end
                # A zero pivot of a positive semidefinite H_t has a zero column.
                C[i, j] = h[j] > 0 ? s / h[j] : zero(ET)
            end
        end
        Cl = UnitLowerTriangular(C)
        ldiv!(Cl, Zs)
        ldiv!(Cl, w)
    end

    # G maps the transformed innovation w to δ until the end, where it is
    # converted to a gain on v.
    fill!(δ, zero(ET))
    fill!(G, zero(ET))
    loglik = zero(ET)
    nrows = 0

    for i in 1:pt
        z = view(Zs, i, :)
        mul!(Minf, Pinf, z)
        mul!(Mstar, Pstar, z)
        Finf_i = dot(z, Minf)
        Fstar_i = dot(z, Mstar) + h[i]
        v_i = w[i] - dot(z, δ)

        if Finf_i > tol
            # Exact diffuse step for one row.
            c = Fstar_i / Finf_i^2
            for k in 1:m, l in 1:m

                Pstar[l, k] += c * Minf[l] * Minf[k] -
                               (Mstar[l] * Minf[k] + Minf[l] * Mstar[k]) / Finf_i
                Pinf[l, k] -= Minf[l] * Minf[k] / Finf_i
            end
            gain = Minf
            F_i = Finf_i
        elseif Fstar_i > tol
            # Standard step on Pstar; the row contributes to the likelihood.
            loglik -= (log(Fstar_i) + v_i^2 / Fstar_i) / 2
            nrows += 1
            for k in 1:m, l in 1:m

                Pstar[l, k] -= Mstar[l] * Mstar[k] / Fstar_i
            end
            gain = Mstar
            F_i = Fstar_i
        elseif Fstar_i < -tol
            return (loglik, nrows, false)
        else
            # The row is predicted exactly and carries no information.
            continue
        end

        # δ += kᵢ vᵢ and G += kᵢ (eᵢ' - zᵢ G), with kᵢ = gain / Fᵢ.
        for l in 1:m
            δ[l] += gain[l] * v_i / F_i
        end
        for j in 1:pt
            g_j = (j == i ? one(ET) : zero(ET)) - dot(z, view(G, :, j))
            for l in 1:m
                G[l, j] += gain[l] * g_j / F_i
            end
        end
    end

    isdiag(Hr) || rdiv!(G, UnitLowerTriangular(C))
    return (loglik, nrows, true)
end

"""
    _univariate_diffuse_step(a, Pstar, Pinf, Zr, Hr, v, tol)
        -> (a_filt, Pstar_filt, Pinf_filt, G, loglik, nrows, ok)

Allocating form of [`_univariate_diffuse_update!`](@ref) for the functional
filters: returns the filtered state, the finite and diffuse parts of the filtered
covariance, the gain `G` with `a_filt = a + G v`, and the period's likelihood
terms.
"""
function _univariate_diffuse_step(a, Pstar, Pinf, Zr, Hr, v, tol)
    pt, m = size(Zr)
    ET = promote_type(eltype(a), eltype(Pstar), eltype(Pinf), eltype(Zr), eltype(Hr),
        eltype(v))
    Pstar_f = Matrix{ET}(Pstar)
    Pinf_f = Matrix{ET}(Pinf)
    δ = Vector{ET}(undef, m)
    G = Matrix{ET}(undef, m, pt)
    (ll, nrows, ok) = _univariate_diffuse_update!(δ, G, Pstar_f, Pinf_f,
        Matrix{ET}(Zr), Vector{ET}(v), Vector{ET}(undef, pt), Hr,
        Matrix{ET}(undef, pt, pt), Vector{ET}(undef, m), Vector{ET}(undef, m), tol)
    return (a + δ, Pstar_f, Pinf_f, G, ll, nrows, ok)
end

"""
    kalman_loglik(p::KFParms, y, a1, P1_star, P1_inf; tol=1e-8) -> loglik

Compute log-likelihood using exact diffuse initialization (Durbin-Koopman method).
Passing the diffuse part `P1_inf` is what selects it; the 4-argument
`kalman_loglik(p, y, a1, P1)` is the ordinary filter.

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

NaN marks a missing value, one element at a time: in a partially observed period
both phases use the reduced system formed by the observed rows.

A diffuse-period step where `Finf = Z_t Pinf Z_t'` is singular but nonzero (more
observed rows than diffuse directions they load on) is processed one row at a time
(Koopman & Durbin 2000). A non-diagonal `H` is decorrelated first with its unit LDL
factor.

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
"""
function kalman_loglik(
        p::KFParms,
        y::AbstractMatrix,
        a1::AbstractVector,
        P1_star::AbstractMatrix,
        P1_inf::AbstractMatrix;
        tol::Real = 1e-8
)
    y = _nan_obs(y)
    n = size(y, 2)

    ET = _filter_eltype(p, a1, P1_star, P1_inf)

    # Initialize state
    a = Vector{ET}(a1)
    Pstar = Matrix{ET}(P1_star)
    Pinf = Matrix{ET}(P1_inf)

    # Precompute RQR'
    RQR = p.R * p.Q * p.R'

    loglik = zero(ET)
    n_obs_rows = 0  # observed scalars that contribute to the likelihood

    # Track if diffuse period has ended
    diffuse_ended = false

    for t in 1:n
        y_t = view(y, :, t)

        if _all_missing(y_t)
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

        # Restrict to the observed rows. When every row is present these are the
        # full Z, H and y_t and the arithmetic is unchanged.
        rows = _observed_row_indices(y_t)
        pt = length(rows)
        Zr = _select_rows(p.Z, rows)
        Hr = _select_rows_cols(p.H, rows)
        yr = y_t[rows]

        if diffuse_ended
            # === Standard Kalman filter step ===
            n_obs_rows += pt
            v = yr - Zr * a
            F = Zr * Pstar * Zr' + Hr

            if pt == 1
                F_val = F[1, 1]
                if F_val <= zero(ET)
                    return ET(-Inf)
                end
                Finv = fill(one(ET) / F_val, 1, 1)
                logdetF = log(F_val)
            else
                F_sym = Symmetric((F + F') / 2)
                chol_result = cholesky(F_sym; check = false)
                if !issuccess(chol_result)
                    return ET(-Inf)
                end
                Finv = inv(chol_result)
                logdetF = 2 * sum(log.(diag(chol_result.U)))
            end

            quad_form = dot(v, Finv * v)
            loglik += -ET(0.5) * (logdetF + quad_form)

            if !isfinite(loglik)
                return ET(-Inf)
            end

            # Kalman gain and state update
            K = p.T * Pstar * Zr' * Finv
            a = p.T * a + K * v
            L = p.T - K * Zr
            Pstar = L * Pstar * p.T' + RQR
        else
            # === Diffuse filter step ===
            v = yr - Zr * a
            Finf = Zr * Pinf * Zr'

            flag = _diffuse_step_flag(Finf, tol)

            if flag == 2
                # Finf is singular but nonzero: process the rows one at a time.
                (a_filt, Pstar_filt, Pinf_filt, _, ll,
                    nrows, ok) = _univariate_diffuse_step(
                    a, Pstar, Pinf, Zr, Hr, v, tol)
                ok || return ET(-Inf)
                loglik += ll
                n_obs_rows += nrows
                a = p.T * a_filt
                Pinf = p.T * Pinf_filt * p.T'
                Pstar = p.T * Pstar_filt * p.T' + RQR
            elseif flag == 1
                # Finf is invertible: exact diffuse update (Durbin & Koopman 2012,
                # §5.2), with Kinf = K⁽⁰⁾ and Kstar = K⁽¹⁾
                Finf_inv = inv(Finf)
                Kinf = p.T * Pinf * Zr' * Finf_inv
                Linf = p.T - Kinf * Zr
                Fstar = Zr * Pstar * Zr' + Hr
                Kstar = (p.T * Pstar * Zr' - Kinf * Fstar) * Finf_inv

                # State update
                a = p.T * a + Kinf * v

                # Covariance updates
                Pinf_new = p.T * Pinf * Linf'
                Pstar_new = p.T * Pstar * Linf' - Kinf * Finf * Kstar' + RQR

                Pinf = Pinf_new
                Pstar = Pstar_new

                # No likelihood contribution during diffuse period with Finf invertible
            else
                # Finf is zero: no observed row loads on a diffuse direction, so
                # the period takes the standard update on Pstar.
                Fstar = Zr * Pstar * Zr' + Hr

                if pt == 1
                    Fstar_val = Fstar[1, 1]
                    if Fstar_val <= zero(ET)
                        return ET(-Inf)
                    end
                    Fstar_inv = fill(one(ET) / Fstar_val, 1, 1)
                    logdetFstar = log(Fstar_val)
                else
                    Fstar_sym = Symmetric((Fstar + Fstar') / 2)
                    chol_result = cholesky(Fstar_sym; check = false)
                    if !issuccess(chol_result)
                        return ET(-Inf)
                    end
                    Fstar_inv = inv(chol_result)
                    logdetFstar = 2 * sum(log.(diag(chol_result.U)))
                end

                quad_form = dot(v, Fstar_inv * v)
                # This observation contributes to likelihood
                n_obs_rows += pt
                loglik += -ET(0.5) * (logdetFstar + quad_form)

                if !isfinite(loglik)
                    return ET(-Inf)
                end

                Kstar = p.T * Pstar * Zr' * Fstar_inv
                Lstar = p.T - Kstar * Zr

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

    # The Gaussian constant counts every observed scalar that contributed.
    const_term = -n_obs_rows * log(ET(2π)) / 2

    return loglik + const_term
end

"""
    _store_innovations!(vt, Ft, Kt, t, v, F, K)

Store the period-`t` innovation `v`, its covariance `F` and the gain `K` of a
reduced system with `pt = length(v)` observed rows. They occupy the leading rows
and columns of the period's slices; the remaining entries are `NaN` for `vt` and
`Ft` and zero for `Kt`.
"""
function _store_innovations!(vt, Ft, Kt, t, v, F, K)
    pt = length(v)
    vt[:, t] .= NaN
    vt[1:pt, t] = v
    Ft[:, :, t] .= NaN
    Ft[1:pt, 1:pt, t] = F
    Kt[:, :, t] .= 0
    Kt[:, 1:pt, t] = K
    return nothing
end

"""
    kalman_filter(p::KFParms, y, a1, P1_star, P1_inf; tol=1e-8) -> DiffuseFilterResult

Run full Kalman filter with exact diffuse initialization. Passing the diffuse part
`P1_inf` is what selects it; the 4-argument `kalman_filter(p, y, a1, P1)` is the
ordinary filter.

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

NaN marks a missing value, one element at a time. In a partially observed period
both phases use the reduced system formed by the observed rows, and `vt`, `Ft` and
`Kt` hold its pₜ-sized blocks in their leading rows and columns; `observed_mask`
records which rows those are.

A diffuse-period step where `Finf = Z_t Pinf Z_t'` is singular but nonzero (more
observed rows than diffuse directions they load on) is processed one row at a time
(Koopman & Durbin 2000). A non-diagonal `H` is decorrelated first with its unit LDL
factor.

# See Also
- `kalman_loglik(p, y, a1, P1_star, P1_inf)`: log-likelihood only (faster if states
  are not needed)
- `diffuse_period`, `diffuse_flags`, `diffuse_covariances`: the diffuse-period output
"""
function kalman_filter(
        p::KFParms,
        y::AbstractMatrix,
        a1::AbstractVector,
        P1_star::AbstractMatrix,
        P1_inf::AbstractMatrix;
        tol::Real = 1e-8
)
    y = _nan_obs(y)
    n = size(y, 2)
    obs_dim = size(y, 1)
    state_dim = length(a1)

    ET = _filter_eltype(p, a1, P1_star, P1_inf)

    # Allocate storage
    at_store = fill(ET(NaN), state_dim, n)
    Pt_store = fill(ET(NaN), state_dim, state_dim, n)
    att_store = fill(ET(NaN), state_dim, n)
    Ptt_store = fill(ET(NaN), state_dim, state_dim, n)
    vt_store = fill(ET(NaN), obs_dim, n)
    Ft_store = fill(ET(NaN), obs_dim, obs_dim, n)
    Kt_store = fill(ET(NaN), state_dim, obs_dim, n)
    missing_mask = falses(n)
    observed_mask = falses(obs_dim, n)

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
    n_obs_rows = 0
    diffuse_ended = false
    d = 0  # Diffuse period length
    failed_at = 0

    for t in 1:n
        y_t = view(y, :, t)
        for i in 1:obs_dim
            observed_mask[i, t] = !isnan(y_t[i])
        end

        # Store predicted state
        at_store[:, t] = a
        # During diffuse period, P = Pstar (finite part); after, P = Pstar
        Pt_store[:, :, t] = Pstar

        if _all_missing(y_t)
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

        # Restrict to the observed rows; the fully observed case selects all of
        # them and the arithmetic below is unchanged.
        rows = _observed_row_indices(y_t)
        pt = length(rows)
        Zr = _select_rows(p.Z, rows)
        Hr = _select_rows_cols(p.H, rows)
        yr = y_t[rows]

        if diffuse_ended
            # === Standard Kalman filter step ===
            n_obs_rows += pt
            v = yr - Zr * a
            F = Zr * Pstar * Zr' + Hr

            if pt == 1
                F_val = F[1, 1]
                if F_val <= zero(ET)
                    failed_at = t
                    break
                end
                Finv = fill(one(ET) / F_val, 1, 1)
                logdetF = log(F_val)
                quad_form = v[1]^2 / F_val
            else
                F_sym = Symmetric((F + F') / 2)
                chol_result = cholesky(F_sym; check = false)
                if !issuccess(chol_result)
                    failed_at = t
                    break
                end
                Finv = inv(chol_result)
                logdetF = 2 * sum(log.(diag(chol_result.U)))
                quad_form = dot(v, Finv * v)
            end

            loglik += -ET(0.5) * (logdetF + quad_form)

            K = p.T * Pstar * Zr' * Finv
            _store_innovations!(vt_store, Ft_store, Kt_store, t, v, F, K)

            # Filtered state
            a_filt = a + Pstar * Zr' * Finv * v
            P_filt = Pstar - Pstar * Zr' * Finv * Zr * Pstar

            att_store[:, t] = a_filt
            Ptt_store[:, :, t] = P_filt

            # Predict next
            a = p.T * a_filt
            Pstar = p.T * P_filt * p.T' + RQR
        else
            # === Diffuse filter step ===
            v = yr - Zr * a
            Finf = Zr * Pinf * Zr'

            flag = _diffuse_step_flag(Finf, tol)

            # Store diffuse quantities
            push!(Pinf_list, copy(Pinf))
            push!(Pstar_list, copy(Pstar))
            push!(flag_list, flag)
            d += 1

            if flag == 2
                # Finf singular but nonzero: process the rows one at a time. The
                # stored gain is the one that updates the state, a_filt = a + G v.
                (a_filt, Pstar_filt, Pinf_filt, G, ll,
                    nrows, ok) = _univariate_diffuse_step(
                    a, Pstar, Pinf, Zr, Hr, v, tol)
                if !ok
                    failed_at = t
                    break
                end
                loglik += ll
                n_obs_rows += nrows
                _store_innovations!(vt_store, Ft_store, Kt_store, t, v, Finf, p.T * G)

                att_store[:, t] = a_filt
                Ptt_store[:, :, t] = Pstar_filt

                a = p.T * a_filt
                Pinf = p.T * Pinf_filt * p.T'
                Pstar = p.T * Pstar_filt * p.T' + RQR
            elseif flag == 1
                # Finf invertible: exact diffuse update (Durbin & Koopman 2012,
                # §5.2), with Kinf = K⁽⁰⁾ and Kstar = K⁽¹⁾
                Finf_inv = inv(Finf)
                Kinf = p.T * Pinf * Zr' * Finf_inv
                Linf = p.T - Kinf * Zr
                Fstar = Zr * Pstar * Zr' + Hr
                Kstar = (p.T * Pstar * Zr' - Kinf * Fstar) * Finf_inv

                # Finf is stored as the innovation covariance during the diffuse period
                _store_innovations!(vt_store, Ft_store, Kt_store, t, v, Finf, Kinf)

                # Filtered state and the finite part of its covariance: the κ → ∞
                # limit of a + P Z' F⁻¹ v and P - P Z' F⁻¹ Z P with
                # P = κ Pinf + Pstar, where A = Pinf Z' Finf⁻¹.
                M_star = Pstar * Zr'
                A = Pinf * Zr' * Finf_inv
                a_filt = a + A * v
                P_filt = Pstar - A * M_star' - M_star * A' + A * Fstar * A'

                att_store[:, t] = a_filt
                Ptt_store[:, :, t] = P_filt

                a = p.T * a + Kinf * v
                Pinf = p.T * Pinf * Linf'
                Pstar = p.T * Pstar * Linf' - Kinf * Finf * Kstar' + RQR

            else
                # Finf zero: standard update on Pstar
                Fstar = Zr * Pstar * Zr' + Hr

                if pt == 1
                    Fstar_val = Fstar[1, 1]
                    if Fstar_val <= zero(ET)
                        failed_at = t
                        break
                    end
                    Fstar_inv = fill(one(ET) / Fstar_val, 1, 1)
                    logdetFstar = log(Fstar_val)
                    quad_form = v[1]^2 / Fstar_val
                else
                    Fstar_sym = Symmetric((Fstar + Fstar') / 2)
                    chol_result = cholesky(Fstar_sym; check = false)
                    if !issuccess(chol_result)
                        failed_at = t
                        break
                    end
                    Fstar_inv = inv(chol_result)
                    logdetFstar = 2 * sum(log.(diag(chol_result.U)))
                    quad_form = dot(v, Fstar_inv * v)
                end

                n_obs_rows += pt
                loglik += -ET(0.5) * (logdetFstar + quad_form)

                Kstar = p.T * Pstar * Zr' * Fstar_inv
                Lstar = p.T - Kstar * Zr
                _store_innovations!(vt_store, Ft_store, Kt_store, t, v, Fstar, Kstar)

                a_filt = a + Pstar * Zr' * Fstar_inv * v
                P_filt = Pstar - Pstar * Zr' * Fstar_inv * Zr * Pstar

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
        for i in 1:d
            Pinf_store[:, :, i] = Pinf_list[i]
            Pstar_store[:, :, i] = Pstar_list[i]
        end
    else
        Pinf_store = Array{ET}(undef, state_dim, state_dim, 0)
        Pstar_store = Array{ET}(undef, state_dim, state_dim, 0)
    end

    # The Gaussian constant counts every observed scalar that contributed.
    const_term = -n_obs_rows * log(ET(2π)) / 2
    failed_at > 0 && (loglik = ET(-Inf))

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
        observed_mask,
        failed_at
    )
end
