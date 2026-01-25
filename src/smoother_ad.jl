"""
    smoother_ad.jl

AD-compatible Rauch-Tung-Striebel (RTS) smoother for state-space models.

Pure functional implementation that works with automatic differentiation.
"""

"""
    kalman_smoother(Z, T, at, Pt, vt, Ft; compute_crosscov=false, missing_mask=nothing, Ptt=nothing)

AD-compatible RTS smoother. Takes filter outputs and returns smoothed states.

# Arguments
- `Z`: Observation matrix (p × m)
- `T`: Transition matrix (m × m)
- `at`: Predicted states (m × n), where at[:, t] = E[αₜ | y₁:ₜ₋₁]
- `Pt`: Predicted covariances (m × m × n)
- `vt`: Innovations (p × n), where vt[:, t] = yₜ - Z * at[:, t]
- `Ft`: Innovation covariances (p × p × n)
- `compute_crosscov`: If true, compute lag-one covariances P_{t+1,t|n} (default: false)
- `missing_mask`: Optional BitVector indicating missing observations (length n).
  If not provided, inferred from NaN values in vt.
- `Ptt`: Optional filtered covariances (m × m × n). Required for cross-cov with missing data.
  If not provided and compute_crosscov=true, computed from predicted covariances.

# Returns
Named tuple with:
- `alpha`: Smoothed states (m × n), where alpha[:, t] = E[αₜ | y₁:ₙ]
- `V`: Smoothed covariances (m × m × n)
- `P_crosslag`: Cross-lag covariances P_{t+1,t|n} (m × m × (n-1)), only if compute_crosscov=true

# Missing Data Handling
When an observation is missing (indicated by `missing_mask[t] == true` or NaN in `vt[:, t]`),
the smoother uses a simplified recursion that skips the measurement update:
    r_{t-1} = T' * r_t
    N_{t-1} = T' * N_t * T
This is consistent with treating missing observations as having infinite variance.

# Cross-lag covariance formula
Using Shumway & Stoffer (2017), the lag-one covariance smoother gives:
    J_t = P_{t|t} * T' * inv(P_{t+1|t})
    P_{t+1,t|n} = V_{t+1} * J_t'
where P_{t|t} is the filtered (updated) covariance.

For missing observations at time t, P_{t|t} = P_{t|t-1} (no update).

# Notes
This follows Durbin & Koopman (2012), Chapter 4, equations (4.32)-(4.44).
Uses predicted states (at = a_{t|t-1}) not filtered states for the recursion.
This is used by the EM algorithm which requires E[αₜ αₜ₋₁' | y₁:ₙ] for M-step updates.
"""
function kalman_smoother(Z::AbstractMatrix, T::AbstractMatrix,
                          at::AbstractMatrix, Pt::AbstractArray,
                          vt::AbstractMatrix, Ft::AbstractArray;
                          compute_crosscov::Bool=false,
                          missing_mask::Union{Nothing, BitVector}=nothing,
                          Ptt::Union{Nothing, AbstractArray}=nothing)

    state_dim = size(T, 1)
    n_obs = size(vt, 2)

    # Determine element type for AD compatibility
    ET = promote_type(eltype(Z), eltype(T), eltype(at),
                      eltype(Pt), eltype(vt), eltype(Ft))

    # Allocate output arrays
    alpha_smooth = Matrix{ET}(undef, state_dim, n_obs)
    V_smooth = Array{ET}(undef, state_dim, state_dim, n_obs)

    # Initialize smoothing recursion: r_n = 0, N_n = 0
    r_vec = zeros(ET, state_dim)
    N_mat = zeros(ET, state_dim, state_dim)

    # Backward recursion: t = n, n-1, ..., 1
    @inbounds for t in n_obs:-1:1
        # Get predicted state and covariance at time t
        a_t = view(at, :, t)
        P_t = view(Pt, :, :, t)

        # Check for missing observation
        is_missing = if missing_mask !== nothing
            missing_mask[t]
        else
            any(isnan, view(vt, :, t))
        end

        if is_missing
            # Missing observation: use simplified recursion
            # r_{t-1} = T' * r_t
            r_new = T' * r_vec

            # N_{t-1} = T' * N_t * T
            N_new = T' * N_mat * T

            # Smoothed state: α̂_t = a_t + P_t * r_{t-1}
            alpha_smooth[:, t] = a_t + P_t * r_new

            # Smoothed covariance: V_t = P_t - P_t * N_{t-1} * P_t
            V_smooth[:, :, t] = P_t - P_t * N_new * P_t

            # Update for next iteration
            r_vec = r_new
            N_mat = N_new
        else
            # Valid observation: full recursion
            v_t = view(vt, :, t)
            F_t = view(Ft, :, :, t)

            # F_t^{-1}
            if size(F_t, 1) == 1
                F_inv = one(ET) / F_t[1, 1]
                F_inv_mat = fill(F_inv, 1, 1)
            else
                # Multivariate: use Cholesky
                F_sym = Symmetric((F_t + F_t') / 2)
                chol_F = cholesky(F_sym; check=false)
                if issuccess(chol_F)
                    F_inv_mat = inv(chol_F)
                else
                    F_inv_mat = inv(F_t) # Fallback
                end
            end

            # Kalman gain: K_t = T * P_t * Z' * F_t^{-1}
            K_t = T * P_t * Z' * F_inv_mat

            # L_t = T - K_t * Z
            L_t = T - K_t * Z

            # Compute r_{t-1} = Z' * F_t^{-1} * v_t + L_t' * r_t
            r_new = Z' * F_inv_mat * v_t + L_t' * r_vec

            # Compute N_{t-1} = Z' * F_t^{-1} * Z + L_t' * N_t * L_t
            N_new = Z' * F_inv_mat * Z + L_t' * N_mat * L_t

            # Smoothed state: α̂_t = a_t + P_t * r_{t-1}
            alpha_smooth[:, t] = a_t + P_t * r_new

            # Smoothed covariance: V_t = P_t - P_t * N_{t-1} * P_t
            V_smooth[:, :, t] = P_t - P_t * N_new * P_t

            # Update for next iteration
            r_vec = r_new
            N_mat = N_new
        end
    end

    # Compute cross-lag covariances if requested
    # Using Shumway & Stoffer (2017) formula:
    #   J_t = P_{t|t} * T' * inv(P_{t+1|t})
    #   P_{t+1,t|n} = V_{t+1} * J_t'
    #
    # where P_{t|t} is the filtered (updated) covariance.
    # For missing observations, P_{t|t} = P_{t|t-1}.
    if compute_crosscov
        P_crosslag = Array{ET}(undef, state_dim, state_dim, n_obs - 1)

        @inbounds for t in 1:(n_obs - 1)
            # Pt[:,:,t] is P_{t|t-1}
            # Pt[:,:,t+1] is P_{t+1|t}
            P_pred_t = view(Pt, :, :, t)      # P_{t|t-1}
            P_pred_tp1 = view(Pt, :, :, t+1)  # P_{t+1|t}

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
                # Compute from predicted: P_{t|t} = P_{t|t-1} - P_{t|t-1} * Z' * inv(F_t) * Z * P_{t|t-1}
                F_t = view(Ft, :, :, t)
                if size(F_t, 1) == 1
                    F_inv_val = one(ET) / F_t[1,1]
                    F_inv = fill(F_inv_val, 1, 1)
                else
                    chol_F = cholesky(Symmetric(F_t); check=false)
                    F_inv = issuccess(chol_F) ? inv(chol_F) : inv(F_t)
                end
                P_upd_t = P_pred_t - P_pred_t * Z' * F_inv * Z * P_pred_t
            end

            # J_t = P_{t|t} * T' * inv(P_{t+1|t})
            # Use regularized inverse for numerical stability
            P_pred_tp1_reg = Matrix(P_pred_tp1)
            eps_reg = ET(1e-10) * max(one(ET), tr(P_pred_tp1_reg) / state_dim)
            for i in 1:state_dim
                P_pred_tp1_reg[i, i] += eps_reg
            end
            
            # Use Cholesky for inversion
            P_sym = Symmetric(P_pred_tp1_reg)
            chol_P = cholesky(P_sym; check=false)
            if issuccess(chol_P)
                J_t = P_upd_t * T' / chol_P
            else
                J_t = P_upd_t * T' * inv(P_sym)
            end

            # V_{t+1} is the smoothed covariance at t+1
            V_tp1 = view(V_smooth, :, :, t+1)

            # P_{t+1,t|n} = V_{t+1} * J_t'
            P_crosslag[:, :, t] = V_tp1 * J_t'
        end

        return (alpha=alpha_smooth, V=V_smooth, P_crosslag=P_crosslag)
    else
        return (alpha=alpha_smooth, V=V_smooth)
    end
end

"""
    kalman_smoother(result::KalmanFilterResult, Z, T; compute_crosscov=false)

Run RTS smoother using filter result. Convenience method.

Uses `missing_mask` from the filter result for proper handling of missing observations.

# Returns
Named tuple with:
- `alpha`: Smoothed states (m × n)
- `V`: Smoothed covariances (m × m × n)
- `P_crosslag`: Cross-lag covariances (only if compute_crosscov=true)
"""
function kalman_smoother(result::KalmanFilterResult, Z::AbstractMatrix, T::AbstractMatrix;
                          compute_crosscov::Bool=false)
    return kalman_smoother(Z, T, result.at, result.Pt, result.vt, result.Ft;
                            compute_crosscov=compute_crosscov,
                            missing_mask=result.missing_mask,
                            Ptt=result.Ptt)
end

"""
    kalman_smoother(p::KFParms, y, a1, P1; compute_crosscov=false) -> NamedTuple

High-level Kalman smoother that runs filter and smoother in one call.

# Arguments
- `p::KFParms`: State-space model parameters (Z, H, T, R, Q)
- `y`: Observations (p × n matrix)
- `a1`: Initial state mean (m-vector)
- `P1`: Initial state covariance (m × m matrix)
- `compute_crosscov`: If true, compute lag-one covariances (default: false)

# Returns
Named tuple with:
- `α`: Smoothed states (m × n), where α[:, t] = E[αₜ | y₁:ₙ]
- `V`: Smoothed state covariances (m × m × n)
- `P_crosslag`: Cross-lag covariances (only if compute_crosscov=true)

# Example
```julia
p = KFParms(Z, H, T, R, Q)
result = kalman_smoother(p, y, a1, P1)
smoothed_states = result.α
```
"""
function kalman_smoother(p::KFParms, y::AbstractMatrix,
                          a1::AbstractVector, P1::AbstractMatrix;
                          compute_crosscov::Bool=false)
    # Run filter first
    filt = kalman_filter(p, y, a1, P1)

    # Run smoother using predicted states (at, Pt)
    result = kalman_smoother(p.Z, p.T, filt.at, filt.Pt, filt.vt, filt.Ft;
                              compute_crosscov=compute_crosscov,
                              missing_mask=filt.missing_mask,
                              Ptt=filt.Ptt)

    # Return results with α alias for backwards compatibility
    if compute_crosscov
        return (α=result.alpha, V=result.V, P_crosslag=result.P_crosslag)
    else
        return (α=result.alpha, V=result.V)
    end
end

"""
    kalman_filter_and_smooth(p::KFParms, y, a1, P1)

Combined filter and smoother for state-space models.

Returns filtered states, smoothed states, and log-likelihood.
AD-compatible.

# Arguments
- `p::KFParms`: State-space model parameters (Z, H, T, R, Q)
- `y`: Observations (p × n matrix)
- `a1`: Initial state mean (m-vector)
- `P1`: Initial state covariance (m × m matrix)

# Returns
Named tuple with:
- `loglik`: Log-likelihood
- `a_filtered`: Filtered states (m × n) - same as att from filter result
- `P_filtered`: Filtered covariances (m × m × n) - same as Ptt from filter result
- `alpha_smooth`: Smoothed states (m × n)
- `V_smooth`: Smoothed covariances (m × m × n)
"""
function kalman_filter_and_smooth(p::KFParms, y::AbstractMatrix,
                                  a1::AbstractVector, P1::AbstractMatrix)
    # Run filter
    filt = kalman_filter(p, y, a1, P1)

    # Run smoother using predicted states, passing missing_mask
    result = kalman_smoother(p.Z, p.T, filt.at, filt.Pt, filt.vt, filt.Ft;
                              missing_mask=filt.missing_mask)

    return (loglik=filt.loglik, a_filtered=filt.att, P_filtered=filt.Ptt,
            alpha_smooth=result.alpha, V_smooth=result.V)
end

"""
    kalman_smoother_scalar(Z, T, at, Pt, vt, Ft; missing_mask=nothing) -> (alpha, V)

Scalar version of RTS smoother for univariate state-space models.

# Arguments
- `Z::Real`: Observation coefficient
- `T::Real`: State transition coefficient
- `at::AbstractVector`: Predicted states from filter (length n)
- `Pt::AbstractVector`: Predicted variances from filter (length n)
- `vt::AbstractVector`: Innovations from filter (length n), NaN for missing
- `Ft::AbstractVector`: Innovation variances from filter (length n)
- `missing_mask`: Optional BitVector indicating missing observations

# Returns
- `alpha::Vector`: Smoothed states E[αₜ | y₁:ₙ] (length n)
- `V::Vector`: Smoothed variances Var[αₜ | y₁:ₙ] (length n)

# Example
```julia
# Scalar local level: α_{t+1} = α_t + η_t, y_t = α_t + ε_t
Z, T = 1.0, 1.0
filt = kalman_filter_scalar(Z, 100.0, T, 1.0, 10.0, 0.0, 1e7, y)
alpha, V = kalman_smoother_scalar(Z, T, filt.at, filt.Pt, filt.vt, filt.Ft)
```
"""
function kalman_smoother_scalar(Z::Real, T::Real,
                                at::AbstractVector, Pt::AbstractVector,
                                vt::AbstractVector, Ft::AbstractVector;
                                missing_mask::Union{Nothing, BitVector}=nothing)

    n_obs = length(vt)
    ET = promote_type(typeof(Z), typeof(T), eltype(at),
                      eltype(Pt), eltype(vt), eltype(Ft))

    alpha_smooth = Vector{ET}(undef, n_obs)
    V_smooth = Vector{ET}(undef, n_obs)

    # Initialize: r_n = 0, N_n = 0
    r = zero(ET)
    N = zero(ET)

    @inbounds for t in n_obs:-1:1
        a_t = at[t]
        P_t = Pt[t]
        v_t = vt[t]

        # Check for missing observation
        is_missing = if missing_mask !== nothing
            missing_mask[t]
        else
            isnan(v_t)
        end

        if is_missing
            # Missing observation: simplified recursion
            r_new = T * r
            N_new = T * N * T

            # Smoothed estimates
            alpha_smooth[t] = a_t + P_t * r_new
            V_smooth[t] = P_t - P_t * N_new * P_t

            r = r_new
            N = N_new
        else
            # Valid observation
            F_inv = one(ET) / Ft[t]

            # Kalman gain and L
            K_t = T * P_t * Z * F_inv
            L_t = T - K_t * Z

            # Recursion
            r_new = Z * F_inv * v_t + L_t * r
            N_new = Z * F_inv * Z + L_t * N * L_t

            # Smoothed estimates
            alpha_smooth[t] = a_t + P_t * r_new
            V_smooth[t] = P_t - P_t * N_new * P_t

            r = r_new
            N = N_new
        end
    end

    return alpha_smooth, V_smooth
end
