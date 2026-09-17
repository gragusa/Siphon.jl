"""
    ekf_ad.jl

Functional, AD-compatible Extended Kalman Filter likelihood and filter.

The implementation mirrors `kalman_loglik` / `kalman_filter` in `filter_ad.jl`
with two changes per step:

  1. The linear `Z * a` is replaced by the nonlinear pure call
     `measurement(model, a, t)`.
  2. The constant `Z` is replaced by the local Jacobian `Z_t = ∂h/∂a |_{a_t}`,
     either supplied via `measurement_jacobian` (`AnalyticJacobian()`) or
     computed with `ForwardDiff.jacobian` on the pure measurement map
     (`ADJacobian()`).

Missing observations behave as in the linear filter: the measurement update is
skipped and the state is propagated using only the transition equation.
"""

using LinearAlgebra
using ForwardDiff

# ============================================================================
# Jacobian dispatch (functional path)
# ============================================================================

@inline function _ekf_jacobian(::ADJacobian, model, a, t)
    return ForwardDiff.jacobian(x -> measurement(model, x, t), a)
end

@inline function _ekf_jacobian(::AnalyticJacobian, model, a, t)
    return measurement_jacobian(model, a, t)
end

@inline function _ekf_jacobian(::FiniteDiffJacobian, model, a, t)
    # Central differences using a step that scales with |a|.
    ETa = eltype(a)
    n = length(a)
    h_step = sqrt(eps(real(ETa)))
    y0 = measurement(model, a, t)
    p_dim = length(y0)
    J = Matrix{ETa}(undef, p_dim, n)
    a_copy = collect(a)
    @inbounds for j in 1:n
        aj = a_copy[j]
        δ = max(h_step * abs(aj), h_step)
        a_copy[j] = aj + δ
        yp = measurement(model, a_copy, t)
        a_copy[j] = aj - δ
        ym = measurement(model, a_copy, t)
        a_copy[j] = aj
        for i in 1:p_dim
            J[i, j] = (yp[i] - ym[i]) / (2δ)
        end
    end
    return J
end

# ============================================================================
# ekf_loglik
# ============================================================================

"""
    ekf_loglik(p::EKFParms, y, a1, P1) -> loglik

Compute the log-likelihood of the nonlinear-measurement state-space model

```
y_t = h(α_t, t) + ε_t,     ε_t ~ N(0, H)
α_{t+1} = T α_t + R η_t,    η_t ~ N(0, Q)
```

via the Extended Kalman Filter prediction-error decomposition.

# Arguments
- `p::EKFParms`: state-space parameters and measurement model.
- `y::AbstractMatrix`: observations (p × n), missing values as `NaN`.
- `a1::AbstractVector`: initial state mean.
- `P1::AbstractMatrix`: initial state covariance.

# Notes
- This is the AD-compatible path: when `p.jacobian_mode === ADJacobian()` the
  measurement Jacobian is computed by `ForwardDiff.jacobian` on the pure
  `measurement(model, a, t)` map.
- Missing observations skip the measurement update and propagate the state
  via the transition equation, matching `kalman_loglik`.
"""
function ekf_loglik(
        p::EKFParms,
        y::AbstractMatrix,
        a1::AbstractVector,
        P1::AbstractMatrix
)
    n = size(y, 2)
    obs_dim = size(y, 1)
    ET = _ekf_filter_eltype(p, a1, P1)

    a = Vector{ET}(a1)
    P = Matrix{ET}(P1)
    RQR = p.R * p.Q * transpose(p.R)
    Tt = transpose(p.T)
    loglik = zero(ET)
    n_obs = 0

    for t in 1:n
        y_t = view(y, :, t)

        if _has_missing(y_t)
            a = p.T * a
            P = p.T * P * Tt + RQR
            continue
        end

        n_obs += 1

        # Nonlinear measurement and Jacobian at the predicted state
        ŷ = measurement(p.measurement, a, t)
        Zt = _ekf_jacobian(p.jacobian_mode, p.measurement, a, t)
        Ztp = transpose(Zt)

        v = y_t - ŷ
        F = Zt * P * Ztp + p.H

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
            PZt = P * Ztp
            K = p.T * PZt * Finv_val
            a = p.T * a + K * v
            P = p.T * (P - PZt * (Finv_val * transpose(PZt))) * Tt + RQR
        else
            F_sym = Symmetric(F, :L)
            chol_result = cholesky(F_sym; check = false)
            if !issuccess(chol_result)
                return ET(-Inf)
            end
            logdetF = zero(ET)
            L_tri = chol_result.L
            @inbounds for i in 1:obs_dim
                logdetF += log(L_tri[i, i])
            end
            logdetF += logdetF

            Finv_v = chol_result \ v
            quad_form = dot(v, Finv_v)
            loglik += -ET(0.5) * (logdetF + quad_form)
            if !isfinite(loglik)
                return ET(-Inf)
            end

            PZt = P * Ztp
            M = transpose(chol_result \ transpose(PZt))
            K = p.T * M
            a = p.T * a + K * v
            P = p.T * (P - M * (Zt * P)) * Tt + RQR
        end
    end

    const_term = -obs_dim * n_obs * log(ET(2π)) / 2
    return loglik + const_term
end

# ============================================================================
# ekf_filter
# ============================================================================

"""
    ekf_filter(p::EKFParms, y, a1, P1) -> EKFFilterResult

Run the Extended Kalman Filter and return per-step predicted/filtered
moments, innovations, gains, and stored measurement linearisations.

Field semantics match `KalmanFilterResult`; see `EKFFilterResult` for the two
EKF-specific fields `yt` (predicted measurement) and `Zt` (per-step Jacobian).
"""
function ekf_filter(
        p::EKFParms,
        y::AbstractMatrix,
        a1::AbstractVector,
        P1::AbstractMatrix
)
    n = size(y, 2)
    obs_dim = size(y, 1)
    state_dim = length(a1)
    ET = _ekf_filter_eltype(p, a1, P1)

    at_store = Matrix{ET}(undef, state_dim, n)
    Pt_store = Array{ET}(undef, state_dim, state_dim, n)
    att_store = Matrix{ET}(undef, state_dim, n)
    Ptt_store = Array{ET}(undef, state_dim, state_dim, n)
    yt_store = Matrix{ET}(undef, obs_dim, n)
    Zt_store = Array{ET}(undef, obs_dim, state_dim, n)
    vt_store = Matrix{ET}(undef, obs_dim, n)
    Ft_store = Array{ET}(undef, obs_dim, obs_dim, n)
    Kt_store = Array{ET}(undef, state_dim, obs_dim, n)
    missing_mask = BitVector(undef, n)

    a_pred = Vector{ET}(a1)
    P_pred = Matrix{ET}(P1)
    Tt = transpose(p.T)
    RQR = p.R * p.Q * transpose(p.R)
    loglik = zero(ET)
    n_obs = 0

    @inbounds for t in 1:n
        y_t = y[:, t]

        at_store[:, t] = a_pred
        Pt_store[:, :, t] = P_pred

        ŷ = measurement(p.measurement, a_pred, t)
        Zt_local = _ekf_jacobian(p.jacobian_mode, p.measurement, a_pred, t)
        yt_store[:, t] = ŷ
        Zt_store[:, :, t] = Zt_local

        if _has_missing(y_t)
            missing_mask[t] = true
            vt_store[:, t] .= ET(NaN)
            Ft_store[:, :, t] = Zt_local * P_pred * transpose(Zt_local) + p.H
            Kt_store[:, :, t] .= zero(ET)
            att_store[:, t] = a_pred
            Ptt_store[:, :, t] = P_pred
            a_pred = p.T * a_pred
            P_pred = p.T * P_pred * Tt + RQR
            continue
        end

        missing_mask[t] = false
        n_obs += 1

        Ztp = transpose(Zt_local)
        v = y_t - ŷ
        F = Zt_local * P_pred * Ztp + p.H

        if obs_dim == 1
            F_val = F[1, 1]
            if F_val <= zero(ET)
                return EKFFilterResult(
                    p, ET(-Inf), at_store, Pt_store, att_store, Ptt_store,
                    yt_store, Zt_store, vt_store, Ft_store, Kt_store, missing_mask)
            end
            Finv = fill(one(ET) / F_val, 1, 1)
            logdetF = log(F_val)
            quad_form = v[1]^2 / F_val
        else
            F_sym = Symmetric((F + transpose(F)) / 2)
            chol_result = cholesky(F_sym; check = false)
            if !issuccess(chol_result)
                return EKFFilterResult(
                    p, ET(-Inf), at_store, Pt_store, att_store, Ptt_store,
                    yt_store, Zt_store, vt_store, Ft_store, Kt_store, missing_mask)
            end
            Finv = inv(chol_result)
            logdetF = 2 * sum(log.(diag(chol_result.U)))
            quad_form = dot(v, Finv * v)
        end

        loglik += -ET(0.5) * (logdetF + quad_form)

        K = p.T * P_pred * Ztp * Finv
        vt_store[:, t] = v
        Ft_store[:, :, t] = F
        Kt_store[:, :, t] = K

        a_filt = a_pred + P_pred * Ztp * Finv * v
        P_filt = P_pred - P_pred * Ztp * Finv * Zt_local * P_pred
        att_store[:, t] = a_filt
        Ptt_store[:, :, t] = P_filt

        a_pred = p.T * a_filt
        P_pred = p.T * P_filt * Tt + RQR
    end

    const_term = -obs_dim * n_obs * log(ET(2π)) / 2
    return EKFFilterResult(
        p, loglik + const_term, at_store, Pt_store, att_store, Ptt_store,
        yt_store, Zt_store, vt_store, Ft_store, Kt_store, missing_mask
    )
end
