"""
    smoother_ad.jl

AD-compatible Rauch-Tung-Striebel (RTS) smoother for state-space models.

Pure functional implementation that works with automatic differentiation.
"""

"""
    SmootherResult{T<:Real}

Output of the Kalman smoother: the smoothed state means and covariances, and the
cross-lag covariances when they were requested.

# Fields
- `αs`: Smoothed states E[αₜ | y₁:ₙ] (m × n)
- `Vs`: Smoothed covariances Var[αₜ | y₁:ₙ] (m × m × n)
- `Pcross`: Cross-lag covariances Cov[αₜ₊₁, αₜ | y₁:ₙ] (m × m × (n−1)), or `nothing`
  when the smoother ran with `crosscov = false`

# Accessors
`smoothed_states`, `smoothed_states_cov` and `crosslag_covs`.

# Construction
- `kalman_smoother(result)`: smooth the output of `kalman_filter`
- `kalman_smoother(p, y, a1, P1)`: filter and smooth in one call
- `kalman_filter_and_smooth(p, y, a1, P1)`: both results, as a `(filter, smoother)` tuple
- `SmootherResult(ws)`: copy the smoother output of a workspace after
  `kalman_smoother!(ws)`, so later runs on `ws` don't change it
- `SmootherResult(model)`: copy the smoothed states of a fitted `StateSpaceModel`
"""
struct SmootherResult{T <: Real}
    αs::Matrix{T}
    Vs::Array{T, 3}
    Pcross::Union{Nothing, Array{T, 3}}
end

smoothed_states(r::SmootherResult) = r.αs
smoothed_states_cov(r::SmootherResult) = r.Vs

function crosslag_covs(r::SmootherResult)
    r.Pcross === nothing && throw(ArgumentError(
        "crosslag_covs: the smoother ran without cross-lag covariances. " *
        "Call kalman_smoother(result; crosscov = true)."))
    return r.Pcross
end

# A scalar system's matrices as 1 × 1 matrices.
_as_matrix(x::Real) = fill(x, 1, 1)
_as_matrix(x::AbstractMatrix) = x

# The RTS recursion on the arrays of a filter output. Durbin & Koopman (2012), §4.4,
# eqs. (4.32)-(4.44), on the predicted states `at = a_{t|t-1}`. Where a period was only
# partly observed, the filter stored the reduced-system v_t and F_t in the leading
# positions, so the recursion uses the matching rows of Z. With `crosscov = true` it
# also returns the lag-one covariances of Shumway & Stoffer (2017):
#     J_t = P_{t|t} T' P_{t+1|t}⁻¹,  P_{t+1,t|n} = V_{t+1} J_t',
# which EM needs for E[αₜ αₜ₋₁' | y₁:ₙ]. `Ptt` supplies P_{t|t}; without it P_{t|t} is
# rebuilt from the predicted covariances. `Z` is either one p × m matrix or, for the
# linearization of an extended Kalman filter, a p × m × n array with one Zₜ per period.
function _kalman_smoother(
        Z::AbstractArray,
        T::AbstractMatrix,
        at::AbstractMatrix,
        Pt::AbstractArray,
        vt::AbstractMatrix,
        Ft::AbstractArray;
        crosscov::Bool = false,
        missing_mask::Union{Nothing, BitVector} = nothing,
        observed_mask::Union{Nothing, AbstractMatrix{Bool}, BitMatrix} = nothing,
        Ptt::Union{Nothing, AbstractArray} = nothing
)
    state_dim = size(T, 1)
    n_obs = size(vt, 2)

    # Determine element type for AD compatibility
    ET = promote_type(eltype(Z), eltype(T), eltype(at), eltype(Pt), eltype(vt), eltype(Ft))

    # Allocate output arrays
    alpha_smooth = Matrix{ET}(undef, state_dim, n_obs)
    V_smooth = Array{ET}(undef, state_dim, state_dim, n_obs)

    # Initialize smoothing recursion: r_n = 0, N_n = 0
    r_vec = zeros(ET, state_dim)
    N_mat = zeros(ET, state_dim, state_dim)

    # Backward recursion: t = n, n-1, ..., 1
    for t in n_obs:-1:1
        (α_t, V_t, r_vec, N_mat) = _rts_step(_period_Z(Z, t), T, at, Pt, vt, Ft, t,
            r_vec, N_mat;
            missing_mask, observed_mask)
        alpha_smooth[:, t] = α_t
        V_smooth[:, :, t] = V_t
    end

    # Compute cross-lag covariances if requested
    # Using Shumway & Stoffer (2017) formula:
    #   J_t = P_{t|t} * T' * inv(P_{t+1|t})
    #   P_{t+1,t|n} = V_{t+1} * J_t'
    #
    # where P_{t|t} is the filtered (updated) covariance.
    # For missing observations, P_{t|t} = P_{t|t-1}.
    if crosscov
        P_crosslag = Array{ET}(undef, state_dim, state_dim, n_obs - 1)

        @inbounds for t in 1:(n_obs - 1)
            # Pt[:,:,t] is P_{t|t-1}
            # Pt[:,:,t+1] is P_{t+1|t}
            P_pred_t = view(Pt, :, :, t)      # P_{t|t-1}
            P_pred_tp1 = view(Pt, :, :, (t + 1))  # P_{t+1|t}

            # Check if observation at time t is missing
            is_missing_t = if missing_mask !== nothing
                missing_mask[t]
            else
                any(isnan, view(vt, :, t))
            end

            # Get P_{t|t} (filtered covariance)
            if is_missing_t
                # For missing: P_{t|t} = P_{t|t-1}
                P_upd_t = P_pred_t
            elseif Ptt !== nothing
                # Use provided filtered covariances
                P_upd_t = view(Ptt, :, :, t)
            else
                # Compute from predicted: P_{t|t} = P_{t|t-1} - P_{t|t-1} * Zr' * inv(F_t) * Zr * P_{t|t-1}
                pt = _period_obs_dim(observed_mask, vt, t)
                Zr = _select_rows(_period_Z(Z, t), _period_obs_rows(observed_mask, vt, t))
                F_t = view(Ft, 1:pt, 1:pt, t)
                F_inv = inv(F_t)
                P_upd_t = P_pred_t - P_pred_t * Zr' * F_inv * Zr * P_pred_t
            end

            # J_t = P_{t|t} * T' * inv(P_{t+1|t}), regularized for numerical stability
            eps_reg = ET(1e-10) * max(one(ET), tr(P_pred_tp1) / state_dim)
            J_t = P_upd_t * T' * inv(Symmetric(Matrix(P_pred_tp1) + eps_reg * I))

            # V_{t+1} is the smoothed covariance at t+1
            V_tp1 = view(V_smooth, :, :, (t + 1))

            # P_{t+1,t|n} = V_{t+1} * J_t'
            P_crosslag[:, :, t] = V_tp1 * J_t'
        end

        return SmootherResult(alpha_smooth, V_smooth, P_crosslag)
    else
        return SmootherResult(alpha_smooth, V_smooth, nothing)
    end
end

_period_Z(Z::AbstractMatrix, t) = Z
_period_Z(Z::AbstractArray{<:Any, 3}, t) = view(Z, :, :, t)

# One period of the backward recursion of `_kalman_smoother`: from `r = r_t` and
# `N = N_t` it returns the smoothed mean and covariance of period `t` with
# `r_{t-1}` and `N_{t-1}`.
function _rts_step(Z, T, at, Pt, vt, Ft, t, r_vec, N_mat; missing_mask, observed_mask)
    a_t = view(at, :, t)
    P_t = view(Pt, :, :, t)

    is_missing = if missing_mask !== nothing
        missing_mask[t]
    else
        any(isnan, view(vt, :, t))
    end

    if is_missing
        # r_{t-1} = T' r_t, N_{t-1} = T' N_t T
        r_new = T' * r_vec
        N_new = T' * N_mat * T
    else
        # Where the period was only partly observed, the filter stored the
        # reduced-system v_t and F_t in the leading positions, so the recursion
        # uses the matching rows of Z.
        pt = _period_obs_dim(observed_mask, vt, t)
        rows = _period_obs_rows(observed_mask, vt, t)
        Zr = _select_rows(Z, rows)
        v_t = view(vt, 1:pt, t)
        F_t = view(Ft, 1:pt, 1:pt, t)

        F_inv = inv(F_t)
        # K_t = T P_t Z_t' F_t⁻¹, L_t = T - K_t Z_t
        K_t = T * P_t * Zr' * F_inv
        L_t = T - K_t * Zr

        # r_{t-1} = Z_t' F_t⁻¹ v_t + L_t' r_t, N_{t-1} = Z_t' F_t⁻¹ Z_t + L_t' N_t L_t
        r_new = Zr' * F_inv * v_t + L_t' * r_vec
        N_new = Zr' * F_inv * Zr + L_t' * N_mat * L_t
    end

    # α̂_t = a_t + P_t r_{t-1}, V_t = P_t - P_t N_{t-1} P_t
    return (a_t + P_t * r_new, P_t - P_t * N_new * P_t, r_new, N_new)
end

"""
    kalman_smoother(result::KalmanFilterResult; crosscov=false) -> SmootherResult

Run the RTS smoother on the output of [`kalman_filter`](@ref). The system matrices,
missing-data masks and filtered covariances come from `result`.

With `crosscov = true` the result also holds the cross-lag covariances
Cov[αₜ₊₁, αₜ | y₁:ₙ], which EM needs.

Throws an `ArgumentError` if the filter stopped on an innovation covariance that was not
positive definite.

# Example
```julia
filt = kalman_filter(p, y, a1, P1)
sm = kalman_smoother(filt)
smoothed_states(sm)
```
"""
function kalman_smoother(result::KalmanFilterResult; crosscov::Bool = false)
    _require_output(result, :kalman_smoother)
    return _kalman_smoother(
        _as_matrix(result.p.Z),
        _as_matrix(result.p.T),
        result.at,
        result.Pt,
        result.vt,
        result.Ft;
        crosscov = crosscov,
        missing_mask = result.missing_mask,
        observed_mask = result.observed_mask,
        Ptt = result.Ptt
    )
end

"""
    kalman_smoother(p::KFParms, y, a1, P1; crosscov=false) -> SmootherResult

Run [`kalman_filter`](@ref) and then the smoother on its output. Same as
`kalman_smoother(kalman_filter(p, y, a1, P1); crosscov)`.

# Example
```julia
p = KFParms(Z, H, T, R, Q)
sm = kalman_smoother(p, y, a1, P1)
smoothed_states(sm)
```
"""
function kalman_smoother(p::KFParms, y, a1, P1; crosscov::Bool = false)
    return kalman_smoother(kalman_filter(p, y, a1, P1); crosscov = crosscov)
end

"""
    kalman_filter_and_smooth(p::KFParms, y, a1, P1; crosscov=false)
        -> (KalmanFilterResult, SmootherResult)

Run the filter and the smoother, and return both results. AD-compatible.

The tuple can be destructured, or passed to `plot` to compare filtered and smoothed
states.

# Example
```julia
filt, sm = kalman_filter_and_smooth(p, y, a1, P1)
loglikelihood(filt), smoothed_states(sm)
```
"""
function kalman_filter_and_smooth(p::KFParms, y, a1, P1; crosscov::Bool = false)
    filt = kalman_filter(p, y, a1, P1)
    return filt, kalman_smoother(filt; crosscov = crosscov)
end
