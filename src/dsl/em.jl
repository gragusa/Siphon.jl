"""
    em.jl

Internal EM algorithm implementations for state-space models.

Main user API: `fit!(EM(), model, y)` in inplace.jl

This file provides:
- `EMResult` type for EM results
- `profile_em_ssm()` for DNS models with λ grid search
- Internal M-step functions used by various EM implementations
"""

# Only export profile_em_ssm and EMResult (used by profile_em_ssm)
export profile_em_ssm, ProfileEMResult, EMResult
export _mstep_Z, _mstep_T, _mstep_T_diag, _mstep_H_diag, _mstep_Q_diag, _mstep_H_full, _mstep_Q_full

# ============================================
# Result Type
# ============================================

"""
    EMResult{T, NT}

Result from EM algorithm estimation.

# Fields
- `theta`: Optimal parameters (NamedTuple)
- `theta_vec`: Optimal parameters (Vector)
- `loglik`: Log-likelihood at optimum
- `loglik_history`: Log-likelihood at each iteration
- `converged`: Whether algorithm converged
- `iterations`: Number of iterations performed
- `smoothed_states`: Final smoothed states (m × n)
- `smoothed_cov`: Final smoothed covariances (m × m × n)
- `a1`: Final initial state mean (m-vector). When `update_initial_state=true`,
  this is the estimated initial state; otherwise it's the input value.
- `P1`: Final initial state covariance (m × m matrix). When `update_initial_state=true`,
  this is the estimated initial covariance; otherwise it's the input value.
"""
struct EMResult{T<:Real, NT<:NamedTuple}
    theta::NT
    theta_vec::Vector{T}
    loglik::T
    loglik_history::Vector{T}
    converged::Bool
    iterations::Int
    smoothed_states::Matrix{T}
    smoothed_cov::Array{T,3}
    a1::Vector{T}
    P1::Matrix{T}
end

# ============================================
# Internal EM implementations
# ============================================
# These are used by fit!(EM(), ...) in inplace.jl and profile_em_ssm below

# ============================================
# Local Level EM Implementation
# ============================================

"""
    _em_local_level(spec, y; kwargs...)

EM algorithm specifically for the local level model.

Model:
    yₜ = μₜ + εₜ,  εₜ ~ N(0, var_obs)
    μₜ₊₁ = μₜ + ηₜ,  ηₜ ~ N(0, var_level)

M-step closed forms:
    var_obs = (1/n) * Σₜ [(yₜ - μ̂ₜ)² + Vₜ]
    var_level = (1/(n-1)) * Σₜ [(μ̂ₜ - μ̂ₜ₋₁)² + Vₜ + Vₜ₋₁ - 2Pₜ,ₜ₋₁]

# Keyword Arguments
- `update_initial_state`: If true, update initial state (a1, P1) at each EM iteration
  using the smoothed state estimates (MARSS-style). Default: false.
"""
function _em_local_level(spec::SSMSpec, y::AbstractMatrix;
                          maxiter::Int=100,
                          tol_ll::Real=1e-6,
                          tol_param::Real=1e-6,
                          verbose::Bool=false,
                          update_initial_state::Bool=false)

    n = size(y, 2)

    # Determine which parameters are free vs fixed
    names = param_names(spec)
    var_obs_free = :var_obs in names
    var_level_free = :var_level in names

    # Initialize parameters from spec
    theta_init = initial_values(spec)

    # Get initial values for free parameters
    var_obs = 1.0
    var_level = 1.0

    if var_obs_free
        idx = findfirst(==(:var_obs), names)
        var_obs = theta_init[idx]
    else
        # Extract fixed value from spec's H matrix
        var_obs = _get_fixed_value(spec.H, (1, 1))
    end

    if var_level_free
        idx = findfirst(==(:var_level), names)
        var_level = theta_init[idx]
    else
        # Extract fixed value from spec's Q matrix
        var_level = _get_fixed_value(spec.Q, (1, 1))
    end

    # Get initial state from spec (make mutable copies for potential updating)
    a1_spec = build_initial_state(spec, _make_theta_nt(names, theta_init))
    a1 = copy(a1_spec[1])
    P1 = copy(a1_spec[2])

    # Iteration history
    ll_history = Float64[]

    # EM iterations
    converged = false
    iter = 0
    ll_prev = -Inf

    for iter_i in 1:maxiter
        iter = iter_i

        # Build current KFParms
        p = _build_local_level_kfparms(var_obs, var_level)

        # E-step: Kalman filter and smoother with cross-covariances
        filt = kalman_filter(p, y, a1, P1)
        smooth_result = kalman_smoother(p.Z, p.T, filt.at, filt.Pt, filt.vt, filt.Ft;
                                         compute_crosscov=true)

        alpha_hat = smooth_result.alpha      # m × n
        V_hat = smooth_result.V              # m × m × n
        P_crosslag = smooth_result.P_crosslag  # m × m × (n-1)

        # Current log-likelihood
        ll = filt.loglik
        push!(ll_history, ll)

        if verbose
            println("EM iter $iter: loglik = $(round(ll, digits=4)), " *
                    "var_obs = $(round(var_obs, digits=4)), " *
                    "var_level = $(round(var_level, digits=4))")
        end

        # Check convergence (log-likelihood should increase)
        if iter > 1
            ll_change = ll - ll_prev
            if ll_change < -1e-8
                @warn "Log-likelihood decreased at iteration $iter: $(round(ll_change, sigdigits=4))"
            end
            if abs(ll_change) < tol_ll
                converged = true
                if verbose
                    println("Converged: log-likelihood change < $tol_ll")
                end
                break
            end
        end
        ll_prev = ll

        # M-step: Update parameters
        var_obs_new, var_level_new = _mstep_local_level(y, alpha_hat, V_hat, P_crosslag)

        # Apply updates only for free parameters, enforce positivity
        if var_obs_free
            var_obs_new = max(var_obs_new, 1e-10)
        else
            var_obs_new = var_obs  # Keep fixed value
        end

        if var_level_free
            var_level_new = max(var_level_new, 1e-10)
        else
            var_level_new = var_level  # Keep fixed value
        end

        # Check parameter convergence
        param_change = abs(var_obs_new - var_obs) + abs(var_level_new - var_level)

        var_obs = var_obs_new
        var_level = var_level_new

        # Update initial state (MARSS-style, optional)
        if update_initial_state
            a1_new, P1_new = _mstep_initial_state(alpha_hat, V_hat)
            # Ensure P1 stays positive definite
            P1_new = Matrix(project_psd(P1_new, 1e-8))
            param_change += sum(abs, a1_new - a1) + sum(abs, P1_new - P1)
            a1 = a1_new
            P1 = P1_new
        end

        if param_change < tol_param
            converged = true
            if verbose
                println("Converged: parameter change < $tol_param")
            end
        end

        if converged
            break
        end
    end

    # Build final result
    theta_vec = Float64[]
    for name in names
        if name == :var_obs
            push!(theta_vec, var_obs)
        elseif name == :var_level
            push!(theta_vec, var_level)
        end
    end

    theta_nt = _make_theta_nt(names, theta_vec)

    # Final smoother run for output
    p_final = _build_local_level_kfparms(var_obs, var_level)
    filt_final = kalman_filter(p_final, y, a1, P1)
    smooth_final = kalman_smoother(p_final.Z, p_final.T, filt_final.at, filt_final.Pt,
                                    filt_final.vt, filt_final.Ft; compute_crosscov=false)

    EMResult(
        theta_nt,
        theta_vec,
        filt_final.loglik,
        ll_history,
        converged,
        iter,
        smooth_final.alpha,
        smooth_final.V,
        vec(a1),
        Matrix(P1)
    )
end

"""
    _build_local_level_kfparms(var_obs, var_level)

Build KFParms for local level model with given variances.
"""
function _build_local_level_kfparms(var_obs::Real, var_level::Real)
    T = promote_type(typeof(var_obs), typeof(var_level))
    Z = T[1.0;;]
    H = T[var_obs;;]
    Tm = T[1.0;;]
    R = T[1.0;;]
    Q = T[var_level;;]
    KFParms(Z, H, Tm, R, Q)
end

"""
    _mstep_local_level(y, alpha_hat, V_hat, P_crosslag)

M-step for local level model. Returns updated (var_obs, var_level).

Formulas:
    var_obs = (1/n) * Σₜ [(yₜ - α̂ₜ)² + Vₜ]
    var_level = (1/(n-1)) * Σₜ₌₂ⁿ [(α̂ₜ - α̂ₜ₋₁)² + Vₜ + Vₜ₋₁ - 2Pₜ,ₜ₋₁]
"""
function _mstep_local_level(y::AbstractMatrix, alpha_hat::AbstractMatrix,
                            V_hat::AbstractArray, P_crosslag::AbstractArray)
    n = size(y, 2)

    # Observation variance: var_obs
    sum_obs = zero(eltype(alpha_hat))
    @inbounds for t in 1:n
        residual = y[1, t] - alpha_hat[1, t]
        sum_obs += residual^2 + V_hat[1, 1, t]
    end
    var_obs = sum_obs / n

    # State variance: var_level
    # Σₜ₌₂ⁿ [(α̂ₜ - α̂ₜ₋₁)² + Vₜ + Vₜ₋₁ - 2Pₜ,ₜ₋₁|n]
    sum_state = zero(eltype(alpha_hat))
    @inbounds for t in 2:n
        state_diff = alpha_hat[1, t] - alpha_hat[1, t-1]
        # P_crosslag[:, :, t-1] = P_{t,t-1|n}
        cross_cov = P_crosslag[1, 1, t-1]
        sum_state += state_diff^2 + V_hat[1, 1, t] + V_hat[1, 1, t-1] - 2*cross_cov
    end
    var_level = sum_state / (n - 1)

    return (var_obs, var_level)
end

"""
    _get_fixed_value(mat_spec::SSMMatrixSpec, idx::Tuple{Int,Int})

Extract fixed value from matrix specification.
"""
function _get_fixed_value(mat_spec::SSMMatrixSpec, idx::Tuple{Int,Int})
    elem = get(mat_spec.elements, idx, mat_spec.default)
    if elem isa FixedValue
        return elem.value
    else
        return 1.0  # Default fallback
    end
end

"""
    _make_theta_nt(names, values)

Create a NamedTuple from parameter names and values.
"""
function _make_theta_nt(names::Vector{Symbol}, values::AbstractVector)
    NamedTuple{Tuple(names)}(Tuple(values))
end

# ============================================
# Generalized EM for Diagonal Covariance Models
# ============================================

"""
    _em_diagonal_ssm(Z, T, R, H_diag_params, Q_diag_params, y, a1, P1; kwargs...)

EM algorithm for state-space models with diagonal H and Q covariances.

Model:
    yₜ = Z αₜ + εₜ,    εₜ ~ N(0, H)  where H is diagonal
    αₜ₊₁ = T αₜ + R ηₜ,  ηₜ ~ N(0, Q)  where Q is diagonal

# Arguments
- `Z`: Fixed observation matrix (p × m)
- `T`: Fixed transition matrix (m × m), typically identity for random walk
- `R`: Fixed selection matrix (m × r)
- `H_diag_params`: Vector of Symbols for diagonal H parameters (free ones only)
- `Q_diag_params`: Vector of Symbols for diagonal Q parameters (free ones only)
- `H_diag_fixed`: Vector of (index, value) for fixed H diagonal elements
- `Q_diag_fixed`: Vector of (index, value) for fixed Q diagonal elements
- `y`: Observations (p × n)
- `a1`: Initial state mean (m-vector)
- `P1`: Initial state covariance (m × m)
- `maxiter`: Maximum iterations
- `tol_ll`: Log-likelihood convergence tolerance
- `tol_param`: Parameter convergence tolerance
- `verbose`: Print progress
- `update_initial_state`: If true, update initial state (a1, P1) at each EM iteration
  using the smoothed state estimates (MARSS-style). Default: false.
"""
function _em_diagonal_ssm(Z::AbstractMatrix, T::AbstractMatrix, R::AbstractMatrix,
                          H_diag_params::Vector{Symbol}, Q_diag_params::Vector{Symbol},
                          H_diag_fixed::Vector{Tuple{Int,Float64}},
                          Q_diag_fixed::Vector{Tuple{Int,Float64}},
                          y::AbstractMatrix, a1_input::AbstractVector, P1_input::AbstractMatrix;
                          H_init::Vector{Float64}=ones(size(y, 1)),
                          Q_init::Vector{Float64}=ones(size(R, 2)),
                          maxiter::Int=100,
                          tol_ll::Real=1e-6,
                          tol_param::Real=1e-6,
                          verbose::Bool=false,
                          update_initial_state::Bool=false)

    p, n = size(y)      # p observations, n time points
    m = size(T, 1)      # m states
    r = size(R, 2)      # r shocks

    # Make mutable copies of initial state (will be updated if update_initial_state=true)
    a1 = copy(a1_input)
    P1 = copy(P1_input)

    # Current variance estimates (diagonal elements)
    H_diag = copy(H_init)
    Q_diag = copy(Q_init)

    # Apply fixed values
    for (idx, val) in H_diag_fixed
        H_diag[idx] = val
    end
    for (idx, val) in Q_diag_fixed
        Q_diag[idx] = val
    end

    # Track which indices are free
    H_free_idx = Int[]
    for (i, sym) in enumerate(H_diag_params)
        if sym !== :_fixed_
            push!(H_free_idx, i)
        end
    end
    # Actually, H_diag_params contains only free param names, need different approach
    # Let's track by position: all indices not in H_diag_fixed are free
    H_fixed_idx = Set(idx for (idx, _) in H_diag_fixed)
    Q_fixed_idx = Set(idx for (idx, _) in Q_diag_fixed)

    # Iteration history
    ll_history = Float64[]

    # EM iterations
    converged = false
    iter = 0
    ll_prev = -Inf

    for iter_i in 1:maxiter
        iter = iter_i

        # Build current KFParms
        H = Diagonal(H_diag)
        Q = Diagonal(Q_diag)
        kfp = KFParms(Z, Matrix(H), T, R, Matrix(Q))

        # E-step: Kalman filter and smoother
        filt = kalman_filter(kfp, y, a1, P1)
        smooth_result = kalman_smoother(Z, T, filt.at, filt.Pt, filt.vt, filt.Ft;
                                         compute_crosscov=true)

        alpha_hat = smooth_result.alpha       # m × n
        V_hat = smooth_result.V               # m × m × n
        P_crosslag = smooth_result.P_crosslag # m × m × (n-1)

        # Current log-likelihood
        ll = filt.loglik
        push!(ll_history, ll)

        if verbose
            println("EM iter $iter: loglik = $(round(ll, digits=4))")
            println("  H_diag = ", round.(H_diag, digits=4))
            println("  Q_diag = ", round.(Q_diag, digits=4))
        end

        # Check convergence
        if iter > 1
            ll_change = ll - ll_prev
            if ll_change < -1e-8
                @warn "Log-likelihood decreased at iteration $iter: $(round(ll_change, sigdigits=4))"
            end
            if abs(ll_change) < tol_ll
                converged = true
                if verbose
                    println("Converged: log-likelihood change < $tol_ll")
                end
                break
            end
        end
        ll_prev = ll

        # M-step: Update diagonal variances
        H_new, Q_new = _mstep_diagonal(Z, T, R, y, alpha_hat, V_hat, P_crosslag)

        # Apply updates only for free parameters, enforce positivity
        param_change = 0.0
        for i in 1:p
            if !(i in H_fixed_idx)
                H_new_i = max(H_new[i], 1e-10)
                param_change += abs(H_new_i - H_diag[i])
                H_diag[i] = H_new_i
            end
        end
        for i in 1:r
            if !(i in Q_fixed_idx)
                Q_new_i = max(Q_new[i], 1e-10)
                param_change += abs(Q_new_i - Q_diag[i])
                Q_diag[i] = Q_new_i
            end
        end

        # Update initial state (MARSS-style, optional)
        if update_initial_state
            a1_new, P1_new = _mstep_initial_state(alpha_hat, V_hat)
            # Ensure P1 stays positive definite
            P1_new = Matrix(project_psd(P1_new, 1e-8))
            param_change += sum(abs, a1_new - a1) + sum(abs, P1_new - P1)
            a1 = a1_new
            P1 = P1_new
        end

        # Check parameter convergence
        if param_change < tol_param
            converged = true
            if verbose
                println("Converged: parameter change < $tol_param")
            end
        end

        if converged
            break
        end
    end

    # Build final result
    # Collect all free parameter names and values
    all_param_names = Symbol[]
    all_param_values = Float64[]

    # H parameters (observation variances)
    h_param_idx = 1
    for i in 1:p
        if !(i in H_fixed_idx)
            push!(all_param_names, H_diag_params[h_param_idx])
            push!(all_param_values, H_diag[i])
            h_param_idx += 1
        end
    end

    # Q parameters (state variances)
    q_param_idx = 1
    for i in 1:r
        if !(i in Q_fixed_idx)
            push!(all_param_names, Q_diag_params[q_param_idx])
            push!(all_param_values, Q_diag[i])
            q_param_idx += 1
        end
    end

    theta_nt = _make_theta_nt(all_param_names, all_param_values)

    # Final smoother run
    H_final = Diagonal(H_diag)
    Q_final = Diagonal(Q_diag)
    kfp_final = KFParms(Z, Matrix(H_final), T, R, Matrix(Q_final))
    filt_final = kalman_filter(kfp_final, y, a1, P1)
    smooth_final = kalman_smoother(Z, T, filt_final.at, filt_final.Pt,
                                    filt_final.vt, filt_final.Ft; compute_crosscov=false)

    EMResult(
        theta_nt,
        all_param_values,
        filt_final.loglik,
        ll_history,
        converged,
        iter,
        smooth_final.alpha,
        smooth_final.V,
        vec(a1),
        Matrix(P1)
    )
end

"""
    _mstep_diagonal(Z, T, R, y, alpha_hat, V_hat, P_crosslag)

M-step for diagonal covariance model. Returns updated (H_diag, Q_diag).

Formulas for diagonal H (observation covariance):
    H_ii = (1/n) Σₜ [(yᵢₜ - Zᵢ α̂ₜ)² + Zᵢ V̂ₜ Zᵢ']

Formulas for diagonal Q (state covariance), with general T:
    Q_ii = (1/(n-1)) Σₜ₌₂ⁿ [E[(ηᵢₜ)² | Y]]

where ηₜ = R⁻¹(αₜ - T αₜ₋₁) for R invertible (or use pseudo-inverse).

For the case where R = I and T = I (random walk):
    Q_ii = (1/(n-1)) Σₜ₌₂ⁿ [(α̂ᵢₜ - α̂ᵢ,ₜ₋₁)² + V̂ᵢᵢₜ + V̂ᵢᵢ,ₜ₋₁ - 2P̂ₜ,ₜ₋₁,ᵢᵢ]
"""
function _mstep_diagonal(Z::AbstractMatrix, T::AbstractMatrix, R::AbstractMatrix,
                         y::AbstractMatrix, alpha_hat::AbstractMatrix,
                         V_hat::AbstractArray, P_crosslag::AbstractArray)
    p, n = size(y)
    m = size(alpha_hat, 1)
    r = size(R, 2)

    ET = eltype(alpha_hat)

    # ============================================
    # Update H (observation variances)
    # ============================================
    # H_ii = (1/n) Σₜ [(yᵢₜ - Zᵢ α̂ₜ)² + Zᵢ V̂ₜ Zᵢ']
    H_diag = zeros(ET, p)

    @inbounds for i in 1:p
        Z_i = view(Z, i, :)  # 1 × m row
        sum_i = zero(ET)
        for t in 1:n
            # Residual: y_it - Z_i * alpha_hat[:, t]
            residual = y[i, t]
            for k in 1:m
                residual -= Z_i[k] * alpha_hat[k, t]
            end
            # Variance contribution: Z_i * V_t * Z_i'
            var_contrib = zero(ET)
            for k1 in 1:m
                for k2 in 1:m
                    var_contrib += Z_i[k1] * V_hat[k1, k2, t] * Z_i[k2]
                end
            end
            sum_i += residual^2 + var_contrib
        end
        H_diag[i] = sum_i / n
    end

    # ============================================
    # Update Q (state variances)
    # ============================================
    # For R = I, T = I case (random walk):
    # Q_ii = (1/(n-1)) Σₜ₌₂ⁿ [(α̂ᵢₜ - α̂ᵢ,ₜ₋₁)² + V̂ᵢᵢₜ + V̂ᵢᵢ,ₜ₋₁ - 2P̂ₜ,ₜ₋₁,ᵢᵢ]
    #
    # General case with T ≠ I:
    # E[ηₜ ηₜ' | Y] = E[(αₜ - T αₜ₋₁)(αₜ - T αₜ₋₁)' | Y]
    #              = (α̂ₜ - T α̂ₜ₋₁)(α̂ₜ - T α̂ₜ₋₁)' + V̂ₜ + T V̂ₜ₋₁ T' - P̂ₜ,ₜ₋₁ T' - T P̂ₜ₋₁,ₜ
    #
    # For now, assume T = I (random walk) which is common
    # TODO: Generalize to arbitrary T

    Q_diag = zeros(ET, r)

    # Check if T is identity and R is identity
    # (simplifies the formula significantly)
    is_identity_T = (T ≈ I(m))
    is_identity_R = (R ≈ I(m)) && (r == m)

    if is_identity_T && is_identity_R
        # Simple random walk case
        @inbounds for i in 1:r
            sum_i = zero(ET)
            for t in 2:n
                state_diff = alpha_hat[i, t] - alpha_hat[i, t-1]
                cross_cov = P_crosslag[i, i, t-1]
                sum_i += state_diff^2 + V_hat[i, i, t] + V_hat[i, i, t-1] - 2*cross_cov
            end
            Q_diag[i] = sum_i / (n - 1)
        end
    else
        # General case: compute E[ηₜ ηₜ' | Y] for each t
        # ηₜ = αₜ - T αₜ₋₁ (assuming R = I for simplicity)
        # E[ηₜ ηₜ' | Y] = (α̂ₜ - T α̂ₜ₋₁)(α̂ₜ - T α̂ₜ₋₁)' + V̂ₜ + T V̂ₜ₋₁ T' - P̂ₜ,ₜ₋₁ T' - T P̂ₜ,ₜ₋₁'
        @inbounds for i in 1:r
            sum_i = zero(ET)
            for t in 2:n
                # α̂ₜ - T α̂ₜ₋₁ for component i
                eta_hat_i = alpha_hat[i, t]
                for k in 1:m
                    eta_hat_i -= T[i, k] * alpha_hat[k, t-1]
                end

                # V̂ₜ[i,i] + (T V̂ₜ₋₁ T')[i,i] - 2*(P̂ₜ,ₜ₋₁ T')[i,i]
                # = V̂ₜ[i,i] + Σⱼₖ T[i,j] V̂ₜ₋₁[j,k] T[i,k] - 2*Σⱼ P̂ₜ,ₜ₋₁[i,j] T[i,j]
                V_ii = V_hat[i, i, t]

                T_V_T_ii = zero(ET)
                for j in 1:m
                    for k in 1:m
                        T_V_T_ii += T[i, j] * V_hat[j, k, t-1] * T[i, k]
                    end
                end

                P_T_ii = zero(ET)
                for j in 1:m
                    P_T_ii += P_crosslag[i, j, t-1] * T[i, j]
                end

                sum_i += eta_hat_i^2 + V_ii + T_V_T_ii - 2*P_T_ii
            end
            Q_diag[i] = sum_i / (n - 1)
        end
    end

    return (H_diag, Q_diag)
end

# ============================================
# General EM for Full Parameter Estimation
# ============================================

"""
    _em_general_ssm(Z_init, T_init, R, H_init, Q_init, y, a1, P1;
                    Z_free, T_free, H_free, Q_free, kwargs...)

EM algorithm for general state-space models where Z, T, H, Q can all be estimated.

Model:
    yₜ = Z αₜ + εₜ,    εₜ ~ N(0, H)
    αₜ₊₁ = T αₜ + R ηₜ,  ηₜ ~ N(0, Q)

# Arguments
- `Z_init`: Initial observation matrix (p × m)
- `T_init`: Initial transition matrix (m × m)
- `R`: Fixed selection matrix (m × r)
- `H_init`: Initial observation covariance (p × p), diagonal
- `Q_init`: Initial state covariance (r × r), diagonal
- `y`: Observations (p × n)
- `a1`: Initial state mean (m-vector)
- `P1`: Initial state covariance (m × m)
- `Z_free`: BitMatrix indicating which Z elements are free (p × m)
- `T_free`: BitMatrix indicating which T elements are free (m × m)
- `H_free`: BitVector indicating which diagonal H elements are free (length p)
- `Q_free`: BitVector indicating which diagonal Q elements are free (length r)
- `update_initial_state`: If true, update initial state (a1, P1) at each EM iteration
  using the smoothed state estimates (MARSS-style). Default: false.
"""
function _em_general_ssm(Z_init::AbstractMatrix, T_init::AbstractMatrix, R::AbstractMatrix,
                          H_init::AbstractVector, Q_init::AbstractVector,
                          y::AbstractMatrix, a1_input::AbstractVector, P1_input::AbstractMatrix;
                          Z_free::AbstractMatrix{Bool}=trues(size(Z_init)),
                          T_free::AbstractMatrix{Bool}=trues(size(T_init)),
                          H_free::AbstractVector{Bool}=trues(length(H_init)),
                          Q_free::AbstractVector{Bool}=trues(length(Q_init)),
                          maxiter::Int=500,
                          tol_ll::Real=1e-6,
                          tol_param::Real=1e-6,
                          verbose::Bool=false,
                          update_initial_state::Bool=false)

    p, n = size(y)      # p observations, n time points
    m = size(T_init, 1) # m states
    r = size(R, 2)      # r shocks

    # Make mutable copies of initial state (will be updated if update_initial_state=true)
    a1 = copy(a1_input)
    P1 = copy(P1_input)

    # Current parameter estimates
    Z = copy(Z_init)
    T = copy(T_init)
    H_diag = copy(H_init)
    Q_diag = copy(Q_init)

    # Iteration history
    ll_history = Float64[]

    # EM iterations
    converged = false
    iter = 0
    ll_prev = -Inf

    for iter_i in 1:maxiter
        iter = iter_i

        # Build current KFParms
        H = Diagonal(H_diag)
        Q = Diagonal(Q_diag)
        kfp = KFParms(Z, Matrix(H), T, R, Matrix(Q))

        # E-step: Kalman filter and smoother
        filt = kalman_filter(kfp, y, a1, P1)
        smooth_result = kalman_smoother(Z, T, filt.at, filt.Pt, filt.vt, filt.Ft;
                                         compute_crosscov=true)

        alpha_hat = smooth_result.alpha       # m × n
        V_hat = smooth_result.V               # m × m × n
        P_crosslag = smooth_result.P_crosslag # m × m × (n-1)

        # Current log-likelihood
        ll = filt.loglik
        push!(ll_history, ll)

        if verbose
            println("EM iter $iter: loglik = $(round(ll, digits=4))")
        end

        # Check convergence
        if iter > 1
            ll_change = ll - ll_prev
            if ll_change < -1e-6
                @warn "Log-likelihood decreased at iteration $iter: $(round(ll_change, sigdigits=4))"
            end
            if abs(ll_change) < tol_ll
                converged = true
                if verbose
                    println("Converged: log-likelihood change < $tol_ll")
                end
                break
            end
        end
        ll_prev = ll

        # M-step: Update parameters sequentially with constraints applied immediately
        param_change = 0.0

        # 1. Update Z
        Z_new_unconstr = _mstep_Z(y, alpha_hat, V_hat)
        for i in 1:p, j in 1:m
            if Z_free[i, j]
                param_change += abs(Z_new_unconstr[i, j] - Z[i, j])
                Z[i, j] = Z_new_unconstr[i, j]
            end
        end
        if verbose
            ll_after_Z = kalman_loglik(KFParms(Z, Diagonal(H_diag), T, R, Diagonal(Q_diag)), y, a1, P1)
            println("    LL after Z update: ", ll_after_Z, " (change: ", ll_after_Z - ll, ")")
        end

        # 2. Update H (using updated Z)
        H_new = _mstep_H_diag(Z, y, alpha_hat, V_hat)
        for i in 1:p
            if H_free[i]
                H_new_i = max(H_new[i], 1e-10)
                param_change += abs(H_new_i - H_diag[i])
                H_diag[i] = H_new_i
            end
        end
        if verbose
            ll_after_H = kalman_loglik(KFParms(Z, Diagonal(H_diag), T, R, Diagonal(Q_diag)), y, a1, P1)
            println("    LL after H update: ", ll_after_H, " (change: ", ll_after_H - ll, ")")
        end

        # 3. Update T
        T_new_unconstr = _mstep_T(alpha_hat, V_hat, P_crosslag)
        for i in 1:m, j in 1:m
            if T_free[i, j]
                param_change += abs(T_new_unconstr[i, j] - T[i, j])
                T[i, j] = T_new_unconstr[i, j]
            end
        end
        if verbose
            ll_after_T = kalman_loglik(KFParms(Z, Diagonal(H_diag), T, R, Diagonal(Q_diag)), y, a1, P1)
            println("    LL after T update: ", ll_after_T, " (change: ", ll_after_T - ll, ")")
        end

        # 4. Update Q (using updated T)
        Q_new = _mstep_Q_diag(T, alpha_hat, V_hat, P_crosslag)
        for i in 1:r
            if Q_free[i]
                Q_new_i = max(Q_new[i], 1e-10)
                param_change += abs(Q_new_i - Q_diag[i])
                Q_diag[i] = Q_new_i
            end
        end
        if verbose
            ll_after_Q = kalman_loglik(KFParms(Z, Diagonal(H_diag), T, R, Diagonal(Q_diag)), y, a1, P1)
            println("    LL after Q update: ", ll_after_Q, " (change: ", ll_after_Q - ll, ")")
        end

        # 5. Update initial state (MARSS-style, optional)
        if update_initial_state
            a1_new, P1_new = _mstep_initial_state(alpha_hat, V_hat)
            # Ensure P1 stays positive definite
            P1_new = Matrix(project_psd(P1_new, 1e-8))
            param_change += sum(abs, a1_new - a1) + sum(abs, P1_new - P1)
            a1 = a1_new
            P1 = P1_new
            if verbose
                ll_after_init = kalman_loglik(KFParms(Z, Diagonal(H_diag), T, R, Diagonal(Q_diag)), y, a1, P1)
                println("    LL after initial state update: ", ll_after_init, " (change: ", ll_after_init - ll, ")")
            end
        end

        # Check parameter convergence
        if param_change < tol_param
            converged = true
            if verbose
                println("Converged: parameter change < $tol_param")
            end
        end

        if converged
            break
        end
    end

    # Final filter/smoother run
    H_final = Diagonal(H_diag)
    Q_final = Diagonal(Q_diag)
    kfp_final = KFParms(Z, Matrix(H_final), T, R, Matrix(Q_final))
    filt_final = kalman_filter(kfp_final, y, a1, P1)
    smooth_final = kalman_smoother(Z, T, filt_final.at, filt_final.Pt,
                                    filt_final.vt, filt_final.Ft; compute_crosscov=false)

    return (
        Z = Z,
        T = T,
        H_diag = H_diag,
        Q_diag = Q_diag,
        loglik = filt_final.loglik,
        loglik_history = ll_history,
        converged = converged,
        iterations = iter,
        smoothed_states = smooth_final.alpha,
        smoothed_cov = smooth_final.V,
        a1 = a1,
        P1 = P1
    )
end

# ============================================
# Positive-Definiteness Projection
# ============================================

"""
    project_psd(M::AbstractMatrix, ε::Real=1e-10) -> Symmetric

Project matrix onto positive semi-definite cone with minimum eigenvalue ε.

Uses spectral decomposition to clip negative eigenvalues to ε,
ensuring the result is positive definite.
"""
function project_psd(M::AbstractMatrix, ε::Real=1e-10)
    F = eigen(Symmetric(M))
    λ_clipped = max.(F.values, ε)
    return Symmetric(F.vectors * Diagonal(λ_clipped) * F.vectors')
end

# ============================================
# Granular M-Step Functions
# ============================================

function _mstep_Z(y::AbstractMatrix, alpha_hat::AbstractMatrix, V_hat::AbstractArray)
    p, n = size(y)
    m = size(alpha_hat, 1)
    
    sum_y_alpha = zeros(p, m)
    sum_alpha_alpha = zeros(m, m)
    
    # We could theoretically have different missingness per row of y, 
    # but the filter treats the whole column as missing if ANY is NaN.
    # To be general and robust, we check each column.
    
    @inbounds for t in 1:n
        y_t = view(y, :, t)
        if !ismissing_obs(y_t)
            for i in 1:p, j in 1:m
                sum_y_alpha[i, j] += y[i, t] * alpha_hat[j, t]
            end
            for i in 1:m, j in 1:m
                sum_alpha_alpha[i, j] += alpha_hat[i, t] * alpha_hat[j, t] + V_hat[i, j, t]
            end
        end
    end
    
    # Z_new = sum_y_alpha * inv(sum_alpha_alpha)
    return sum_y_alpha / sum_alpha_alpha
end

function _mstep_T(alpha_hat::AbstractMatrix, V_hat::AbstractArray, P_crosslag::AbstractArray)
    n = size(alpha_hat, 2)
    m = size(alpha_hat, 1)
    
    sum_alpha_alpha_lag = zeros(m, m)
    sum_alpha_alpha_prev = zeros(m, m)
    
    @inbounds for t in 2:n
        for i in 1:m, j in 1:m
            sum_alpha_alpha_lag[i, j] += alpha_hat[i, t] * alpha_hat[j, t-1] + P_crosslag[i, j, t-1]
            sum_alpha_alpha_prev[i, j] += alpha_hat[i, t-1] * alpha_hat[j, t-1] + V_hat[i, j, t-1]
        end
    end
    
    # T_new = sum_alpha_alpha_lag * inv(sum_alpha_alpha_prev)
    return sum_alpha_alpha_lag / sum_alpha_alpha_prev
end

"""
    _mstep_T_diag(alpha_hat, V_hat, P_crosslag)

M-step for diagonal T matrix.

For diagonal T, each element T_ii is updated independently:
    T_ii = Σₜ (α̂ᵢₜ α̂ᵢ,ₜ₋₁ + P̂ₜ,ₜ₋₁[i,i]) / Σₜ (α̂²ᵢ,ₜ₋₁ + V̂ᵢᵢ,ₜ₋₁)

This is the correct EM M-step when T is constrained to be diagonal.
"""
function _mstep_T_diag(alpha_hat::AbstractMatrix, V_hat::AbstractArray, P_crosslag::AbstractArray)
    m, n = size(alpha_hat)
    T_diag = zeros(m)

    @inbounds for i in 1:m
        num = 0.0
        den = 0.0
        for t in 2:n
            # Numerator: α̂ᵢₜ α̂ᵢ,ₜ₋₁ + P̂ₜ,ₜ₋₁[i,i]
            num += alpha_hat[i, t] * alpha_hat[i, t-1] + P_crosslag[i, i, t-1]
            # Denominator: α̂²ᵢ,ₜ₋₁ + V̂ᵢᵢ,ₜ₋₁
            den += alpha_hat[i, t-1]^2 + V_hat[i, i, t-1]
        end
        T_diag[i] = den > 0 ? num / den : 0.0
    end
    return T_diag
end

"""
    _mstep_initial_state(alpha_hat, V_hat)

M-step for initial state parameters using smoothed estimates.

Following MARSS (with tinitx=1, i.e., initial state at t=1):
    a1_new = α̂₁|T  (smoothed state mean at t=1)
    P1_new = V̂₁|T  (smoothed state covariance at t=1)

This provides an optional MARSS-style update for the initial state distribution.
When enabled, the initial state is treated as an additional parameter and updated
at each EM iteration using the smoothed state estimates.

# Arguments
- `alpha_hat`: Smoothed state means (m × n matrix)
- `V_hat`: Smoothed state covariances (m × m × n array)

# Returns
- `a1_new`: Updated initial state mean (m-vector)
- `P1_new`: Updated initial state covariance (m × m matrix)

# Notes
- Use this when: you want MARSS compatibility, have short time series where
  the initial state significantly affects the likelihood, or want to estimate
  the unconditional mean/variance of the state process.
- Keep initial state fixed when: using a diffuse prior is appropriate,
  numerical stability is a concern, or the time series is long enough that
  the initial state has negligible effect.

See also: [`_em_general_ssm_full_cov`](@ref), [`profile_em_ssm`](@ref)
"""
function _mstep_initial_state(alpha_hat::AbstractMatrix, V_hat::AbstractArray)
    a1_new = alpha_hat[:, 1]
    P1_new = V_hat[:, :, 1]
    return (a1_new, copy(P1_new))
end

function _mstep_H_diag(Z::AbstractMatrix, y::AbstractMatrix, alpha_hat::AbstractMatrix, V_hat::AbstractArray)
    p, n = size(y)
    m = size(alpha_hat, 1)
    H_diag = zeros(p)
    n_valid = 0

    @inbounds for i in 1:p
        sum_i = 0.0
        n_i = 0
        for t in 1:n
            if !isnan(y[i, t])
                n_i += 1
                # Residual: y_it - Z_i * alpha_hat[:, t]
                residual = y[i, t]
                for k in 1:m
                    residual -= Z[i, k] * alpha_hat[k, t]
                end
                # Variance contribution: Z_i * V_t * Z_i'
                var_contrib = 0.0
                for k1 in 1:m
                    for k2 in 1:m
                        var_contrib += Z[i, k1] * V_hat[k1, k2, t] * Z[i, k2]
                    end
                end
                sum_i += residual^2 + var_contrib
            end
        end
        H_diag[i] = n_i > 0 ? sum_i / n_i : 1.0
    end
    return H_diag
end

function _mstep_Q_diag(T::AbstractMatrix, alpha_hat::AbstractMatrix, V_hat::AbstractArray, P_crosslag::AbstractArray)
    m, n = size(alpha_hat)
    Q_diag = zeros(m)

    @inbounds for i in 1:m
        sum_i = 0.0
        for t in 2:n
            # α̂ₜ - Tᵢα̂ₜ₋₁ for component i
            eta_hat_i = alpha_hat[i, t]
            for k in 1:m
                eta_hat_i -= T[i, k] * alpha_hat[k, t-1]
            end

            # V̂ₜ[i,i]
            V_ii = V_hat[i, i, t]

            # (T V̂ₜ₋₁ T')ᵢᵢ = Σⱼₖ T[i,j] V̂ₜ₋₁[j,k] T[i,k]
            T_V_T_ii = 0.0
            for j in 1:m
                for k in 1:m
                    T_V_T_ii += T[i, j] * V_hat[j, k, t-1] * T[i, k]
                end
            end

            # (P̂ₜ,ₜ₋₁ T')ᵢᵢ = Σⱼ P̂ₜ,ₜ₋₁[i,j] T[i,j]
            P_T_ii = 0.0
            for j in 1:m
                P_T_ii += P_crosslag[i, j, t-1] * T[i, j]
            end

            sum_i += eta_hat_i^2 + V_ii + T_V_T_ii - 2*P_T_ii
        end
        Q_diag[i] = sum_i / (n - 1)
    end
    return Q_diag
end

function _mstep_H_full(Z::AbstractMatrix, y::AbstractMatrix, alpha_hat::AbstractMatrix, V_hat::AbstractArray)
    p, n = size(y)
    m = size(alpha_hat, 1)
    H_new = zeros(p, p)
    n_valid = 0
    
    @inbounds for t in 1:n
        if !ismissing_obs(view(y, :, t))
            n_valid += 1
            # Compute residual: y_t - Z * alpha_hat[:, t]
            residual = Vector{Float64}(undef, p)
            for i in 1:p
                residual[i] = y[i, t]
                for k in 1:m
                    residual[i] -= Z[i, k] * alpha_hat[k, t]
                end
            end
            # Outer product: residual * residual'
            for i in 1:p, j in 1:p
                H_new[i, j] += residual[i] * residual[j]
            end
            # Variance contribution: Z * V_hat[:, :, t] * Z'
            for i in 1:p, j in 1:p
                for k1 in 1:m, k2 in 1:m
                    H_new[i, j] += Z[i, k1] * V_hat[k1, k2, t] * Z[j, k2]
                end
            end
        end
    end
    if n_valid > 0
        H_new ./= n_valid
    end
    return (H_new + H_new') / 2
end

function _mstep_Q_full(T::AbstractMatrix, R::AbstractMatrix, alpha_hat::AbstractMatrix, V_hat::AbstractArray, P_crosslag::AbstractArray)
    m, n = size(alpha_hat)
    r = size(R, 2)
    Q_new = zeros(r, r)
    R_pinv = pinv(R)

    @inbounds for t in 2:n
        # State residual: α̂ₜ - T α̂ₜ₋₁
        state_resid = Vector{Float64}(undef, m)
        for i in 1:m
            state_resid[i] = alpha_hat[i, t]
            for k in 1:m
                state_resid[i] -= T[i, k] * alpha_hat[k, t-1]
            end
        end

        # η̂ₜ = R⁺ * state_resid
        eta_hat = R_pinv * state_resid

        # Outer product contribution: η̂ₜ η̂ₜ'
        for i in 1:r, j in 1:r
            Q_new[i, j] += eta_hat[i] * eta_hat[j]
        end

        # Variance contribution: R⁺ * V_contrib * R⁺'
        # V_contrib = V̂ₜ + T V̂ₜ₋₁ T' - P̂ₜ,ₜ₋₁ T' - T P̂'ₜ,ₜ₋₁
        V_contrib = zeros(m, m)
        for i in 1:m, j in 1:m
            V_contrib[i, j] = V_hat[i, j, t]
            # + T V̂ₜ₋₁ T'
            for k1 in 1:m, k2 in 1:m
                V_contrib[i, j] += T[i, k1] * V_hat[k1, k2, t-1] * T[j, k2]
            end
            # - P̂ₜ,ₜ₋₁ T'
            for k in 1:m
                V_contrib[i, j] -= P_crosslag[i, k, t-1] * T[j, k]
            end
            # - T P̂'ₜ,ₜ₋₁
            for k in 1:m
                V_contrib[i, j] -= T[i, k] * P_crosslag[j, k, t-1]
            end
        end

        # R⁺ * V_contrib * R⁺'
        RVR = R_pinv * V_contrib * R_pinv'
        for i in 1:r, j in 1:r
            Q_new[i, j] += RVR[i, j]
        end
    end
    Q_new ./= (n - 1)
    return (Q_new + Q_new') / 2
end

"""
    _mstep_general(Z, T, R, y, alpha_hat, V_hat, P_crosslag)

Deprecated: Use granular functions instead.
"""
function _mstep_general(Z::AbstractMatrix, T::AbstractMatrix, R::AbstractMatrix,
                        y::AbstractMatrix, alpha_hat::AbstractMatrix,
                        V_hat::AbstractArray, P_crosslag::AbstractArray)
    # This is legacy wrapper for any external calls, but internally we use granular
    Z_new = _mstep_Z(y, alpha_hat, V_hat)
    T_new = _mstep_T(alpha_hat, V_hat, P_crosslag)
    # Note: These use OLD Z and T, preserving legacy behavior of simultaneous update
    H_diag = _mstep_H_diag(Z, y, alpha_hat, V_hat)
    Q_diag = _mstep_Q_diag(T, alpha_hat, V_hat, P_crosslag)
    
    return (Z_new, T_new, H_diag, Q_diag)
end

"""
    _mstep_full_cov(Z, T, R, y, alpha_hat, V_hat, P_crosslag)

Deprecated: Use granular functions instead.
"""
function _mstep_full_cov(Z::AbstractMatrix, T::AbstractMatrix, R::AbstractMatrix,
                         y::AbstractMatrix, alpha_hat::AbstractMatrix,
                         V_hat::AbstractArray, P_crosslag::AbstractArray)
    # This is legacy wrapper for any external calls, but internally we use granular
    Z_new = _mstep_Z(y, alpha_hat, V_hat)
    T_new = _mstep_T(alpha_hat, V_hat, P_crosslag)
    # Note: These use OLD Z and T, preserving legacy behavior of simultaneous update
    H_new = _mstep_H_full(Z, y, alpha_hat, V_hat)
    Q_new = _mstep_Q_full(T, R, alpha_hat, V_hat, P_crosslag)
    
    return (Z_new, T_new, H_new, Q_new)
end

"""
    _em_general_ssm_full_cov(Z_init, T_init, R, H_init, Q_init, y, a1, P1; kwargs...)

EM algorithm for general state-space models with full covariance matrices H and Q.

Model:
    yₜ = Z αₜ + εₜ,    εₜ ~ N(0, H)
    αₜ₊₁ = T αₜ + R ηₜ,  ηₜ ~ N(0, Q)

# Arguments
- `Z_init`: Initial observation matrix (p × m)
- `T_init`: Initial transition matrix (m × m)
- `R`: Fixed selection matrix (m × r)
- `H_init`: Initial observation covariance (p × p) - full matrix
- `Q_init`: Initial state covariance (r × r) - full matrix
- `y`: Observations (p × n)
- `a1`: Initial state mean (m-vector)
- `P1`: Initial state covariance (m × m)
- `Z_free`: BitMatrix indicating which Z elements are free (p × m)
- `T_free`: BitMatrix indicating which T elements are free (m × m)
- `H_free`: BitMatrix indicating which H elements are free (p × p)
- `Q_free`: BitMatrix indicating which Q elements are free (r × r)
- `update_initial_state`: If true, update initial state (a1, P1) at each EM iteration
  using the smoothed state estimates (MARSS-style). Default: false.

# Returns
NamedTuple with: Z, T, H, Q, loglik, loglik_history, converged, iterations,
smoothed_states, smoothed_cov, a1, P1
"""
function _em_general_ssm_full_cov(Z_init::AbstractMatrix, T_init::AbstractMatrix, R::AbstractMatrix,
                                   H_init::AbstractMatrix, Q_init::AbstractMatrix,
                                   y::AbstractMatrix, a1_input::AbstractVector, P1_input::AbstractMatrix;
                                   Z_free::AbstractMatrix{Bool}=trues(size(Z_init)),
                                   T_free::AbstractMatrix{Bool}=trues(size(T_init)),
                                   H_free::AbstractMatrix{Bool}=trues(size(H_init)),
                                   Q_free::AbstractMatrix{Bool}=trues(size(Q_init)),
                                   maxiter::Int=500,
                                   tol_ll::Real=1e-6,
                                   tol_param::Real=1e-6,
                                   verbose::Bool=false,
                                   update_initial_state::Bool=false)

    p, n = size(y)      # p observations, n time points
    m = size(T_init, 1) # m states
    r = size(R, 2)      # r shocks

    # Current parameter estimates
    Z = copy(Z_init)
    T = copy(T_init)
    H = copy(H_init)
    Q = copy(Q_init)

    # Make mutable copies of initial state (will be updated if update_initial_state=true)
    a1 = copy(a1_input)
    P1 = copy(P1_input)

    # Iteration history
    ll_history = Float64[]

    # EM iterations
    converged = false
    iter = 0
    ll_prev = -Inf

    for iter_i in 1:maxiter
        iter = iter_i

        # Build current KFParms
        kfp = KFParms(Z, H, T, R, Q)

        # E-step: Kalman filter and smoother
        filt = kalman_filter(kfp, y, a1, P1)
        smooth_result = kalman_smoother(Z, T, filt.at, filt.Pt, filt.vt, filt.Ft;
                                         compute_crosscov=true)

        alpha_hat = smooth_result.alpha       # m × n
        V_hat = smooth_result.V               # m × m × n
        P_crosslag = smooth_result.P_crosslag # m × m × (n-1)

        # Current log-likelihood
        ll = filt.loglik
        push!(ll_history, ll)

        if verbose
            println("EM iter $iter: loglik = $(round(ll, digits=4))")
        end

        # Check convergence
        if iter > 1
            ll_change = ll - ll_prev
            if ll_change < -1e-6
                @warn "Log-likelihood decreased at iteration $iter: $(round(ll_change, sigdigits=4))"
            end
            if abs(ll_change) < tol_ll
                converged = true
                if verbose
                    println("Converged: log-likelihood change < $tol_ll")
                end
                break
            end
        end
        ll_prev = ll

        # M-step: Update parameters sequentially with constraints applied immediately
        param_change = 0.0

        # 1. Update Z
        Z_new_unconstr = _mstep_Z(y, alpha_hat, V_hat)
        for i in 1:p, j in 1:m
            if Z_free[i, j]
                param_change += abs(Z_new_unconstr[i, j] - Z[i, j])
                Z[i, j] = Z_new_unconstr[i, j]
            end
        end
        if verbose
            ll_after_Z = kalman_loglik(KFParms(Z, H, T, R, Q), y, a1, P1)
            println("    LL after Z update: ", ll_after_Z, " (change: ", ll_after_Z - ll, ")")
        end

        # 2. Update T
        # Check if T is diagonal-only (no off-diagonal elements free)
        T_is_diagonal = !any(T_free[i, j] for i in 1:m for j in 1:m if i != j)

        if T_is_diagonal
            # Use diagonal-specific M-step for proper EM guarantees
            T_diag_new = _mstep_T_diag(alpha_hat, V_hat, P_crosslag)
            for i in 1:m
                if T_free[i, i]
                    param_change += abs(T_diag_new[i] - T[i, i])
                    T[i, i] = T_diag_new[i]
                end
            end
        else
            # Full T case: use unconstrained M-step
            T_new_unconstr = _mstep_T(alpha_hat, V_hat, P_crosslag)
            for i in 1:m, j in 1:m
                if T_free[i, j]
                    param_change += abs(T_new_unconstr[i, j] - T[i, j])
                    T[i, j] = T_new_unconstr[i, j]
                end
            end
        end
        if verbose
            ll_after_T = kalman_loglik(KFParms(Z, H, T, R, Q), y, a1, P1)
            println("    LL after T update: ", ll_after_T, " (change: ", ll_after_T - ll, ")")
        end

        # 3. Update H (using updated Z)
        H_new = _mstep_H_full(Z, y, alpha_hat, V_hat)
        
        # Check if H is diagonal-only (no off-diagonal elements free)
        H_is_diagonal = !any(H_free[i, j] for i in 1:p for j in 1:p if i != j)

        if H_is_diagonal
            # Diagonal case: just update and ensure positivity
            for i in 1:p
                if H_free[i, i]
                    H_new_i = max(H_new[i, i], 1e-10)
                    param_change += abs(H_new_i - H[i, i])
                    H[i, i] = H_new_i
                end
            end
        else
            # Full covariance case: apply free mask and project to PSD
            for i in 1:p, j in 1:p
                if H_free[i, j]
                    param_change += abs(H_new[i, j] - H[i, j])
                    H[i, j] = H_new[i, j]
                    # Ensure symmetry
                    if i != j && H_free[j, i]
                        H[j, i] = H_new[i, j]
                    end
                end
            end
            # Project to PSD cone
            H = Matrix(project_psd(H))
        end
        if verbose
            ll_after_H = kalman_loglik(KFParms(Z, H, T, R, Q), y, a1, P1)
            println("    LL after H update: ", ll_after_H, " (change: ", ll_after_H - ll, ")")
        end

        # 4. Update Q (using updated T)
        Q_new = _mstep_Q_full(T, R, alpha_hat, V_hat, P_crosslag)
        
        # Check if Q is diagonal-only (no off-diagonal elements free)
        Q_is_diagonal = !any(Q_free[i, j] for i in 1:r for j in 1:r if i != j)

        if Q_is_diagonal
            # Diagonal case: just update and ensure positivity
            for i in 1:r
                if Q_free[i, i]
                    Q_new_i = max(Q_new[i, i], 1e-10)
                    param_change += abs(Q_new_i - Q[i, i])
                    Q[i, i] = Q_new_i
                end
            end
        else
            # Full covariance case: apply free mask and project to PSD
            for i in 1:r, j in 1:r
                if Q_free[i, j]
                    param_change += abs(Q_new[i, j] - Q[i, j])
                    Q[i, j] = Q_new[i, j]
                    # Ensure symmetry
                    if i != j && Q_free[j, i]
                        Q[j, i] = Q_new[i, j]
                    end
                end
            end
            # Project to PSD cone
            Q = Matrix(project_psd(Q))
        end
        if verbose
            ll_after_Q = kalman_loglik(KFParms(Z, H, T, R, Q), y, a1, P1)
            println("    LL after Q update: ", ll_after_Q, " (change: ", ll_after_Q - ll, ")")
        end

        # 5. Update initial state (MARSS-style, optional)
        if update_initial_state
            a1_new, P1_new = _mstep_initial_state(alpha_hat, V_hat)
            # Ensure P1 stays positive definite
            P1_new = Matrix(project_psd(P1_new, 1e-8))
            param_change += sum(abs, a1_new - a1) + sum(abs, P1_new - P1)
            a1 = a1_new
            P1 = P1_new
            if verbose
                ll_after_init = kalman_loglik(KFParms(Z, H, T, R, Q), y, a1, P1)
                println("    LL after initial state update: ", ll_after_init, " (change: ", ll_after_init - ll, ")")
            end
        end

        # Check parameter convergence
        if param_change < tol_param
            converged = true
            if verbose
                println("Converged: parameter change < $tol_param")
            end
        end

        if converged
            break
        end
    end

    # Final filter/smoother run
    kfp_final = KFParms(Z, H, T, R, Q)
    filt_final = kalman_filter(kfp_final, y, a1, P1)
    smooth_final = kalman_smoother(Z, T, filt_final.at, filt_final.Pt,
                                    filt_final.vt, filt_final.Ft; compute_crosscov=false)

    return (
        Z = Z,
        T = T,
        H = H,
        Q = Q,
        loglik = filt_final.loglik,
        loglik_history = ll_history,
        converged = converged,
        iterations = iter,
        smoothed_states = smooth_final.alpha,
        smoothed_cov = smooth_final.V,
        a1 = a1,
        P1 = P1
    )
end

# ============================================
# Profile EM for DNS and similar models
# ============================================

"""
    ProfileEMResult{T<:Real, NT<:NamedTuple}

Result from profile EM estimation where some parameters are optimized via
grid search (profile likelihood) while others use EM closed-form updates.

# Fields
- `λ_optimal::T`: Optimal value of profiled parameter
- `θ::NT`: All parameter estimates at optimum (NamedTuple)
- `loglik::T`: Log-likelihood at optimum
- `em_result`: Full EM result at best λ (NamedTuple)
- `λ_grid::Vector{T}`: Grid of λ values searched
- `loglik_profile::Vector{T}`: Profile log-likelihood at each grid point

# Usage
```julia
spec = dns_model(maturities; T_structure=:full, Q_structure=:full)
result = profile_em_ssm(spec, yields; λ_grid=0.01:0.01:0.2)

# Best λ
result.λ_optimal

# All parameters
result.θ  # NamedTuple with λ, T elements, H elements, Q elements

# Profile likelihood plot data
plot(result.λ_grid, result.loglik_profile)
```
"""
struct ProfileEMResult{T<:Real, NT<:NamedTuple}
    λ_optimal::T
    θ::NT
    loglik::T
    em_result::NamedTuple
    λ_grid::Vector{T}
    loglik_profile::Vector{T}
end

"""
    profile_em_ssm(spec::SSMSpec, y::AbstractMatrix; kwargs...) -> ProfileEMResult

Estimate a state-space model using profile EM: grid search over λ (or other
MatrixExpr parameters) with EM for remaining parameters at each grid point.

This is particularly useful for DNS models where λ enters the Z matrix non-linearly
and cannot be efficiently estimated via standard EM.

# Algorithm
1. Grid over λ values
2. For each λ: fix Z(λ), run EM to estimate T, H, Q
3. Return λ with highest profile log-likelihood

# Arguments
- `spec::SSMSpec`: Model specification with MatrixExpr for Z (e.g., from `dns_model()`)
- `y::AbstractMatrix`: Observations (p × n)

# Keyword Arguments
- `λ_grid`: Grid of λ values to search (default: 0.01:0.005:0.2)
- `λ_param::Symbol=:λ`: Name of the profiled parameter
- `verbose::Bool=false`: Print progress
- `maxiter::Int=500`: Max EM iterations per λ
- `tol_ll::Real=1e-6`: EM convergence tolerance
- `warm_start::Bool=true`: Use previous EM solution as starting point
- `update_initial_state::Bool=false`: If true, update initial state (a1, P1) at each
  EM iteration using smoothed state estimates (MARSS-style). Default: false.
- `tinitx::Int=0`: Initial state timing convention (MARSS-style):
  - `tinitx=0`: Initial state (a1, P1) is at t=0. P1 is computed as `T * V0 * T' + R * Q * R'`.
  - `tinitx=1`: Initial state (a1, P1) is at t=1. P1 = V0 directly (no transformation).
- `V0::Union{Real,AbstractMatrix}=100.0`: Initial state covariance. Can be a scalar
  (interpreted as `V0 * I`) or a matrix. Interpretation depends on `tinitx`.

# Returns
`ProfileEMResult` with optimal parameters and profile likelihood

# Example
```julia
maturities = [3, 6, 12, 24, 60, 120]
spec = dns_model(maturities; T_structure=:full, Q_structure=:full)

# Profile EM estimation
result = profile_em_ssm(spec, yields; λ_grid=0.02:0.005:0.15, verbose=true)

println("Optimal λ: ", result.λ_optimal)
println("Log-likelihood: ", result.loglik)

# Extract smoothed factors
model = StateSpaceModel(spec, result.θ, size(yields, 2))
kalman_filter!(model, yields)
kalman_smoother!(model)
smooth = smoothed_states(model)
```

# Notes
- Requires spec to have `:Z` in `matrix_exprs` as a `MatrixExpr`
- Works best with full T and Q structures (`:full`) for EM estimation
- Warm-starting significantly improves speed
"""
function profile_em_ssm(spec::SSMSpec, y::AbstractMatrix;
                         λ_grid=0.01:0.005:0.2,
                         λ_param::Symbol=:λ,
                         verbose::Bool=false,
                         maxiter::Int=500,
                         tol_ll::Real=1e-4,
                         warm_start::Bool=true,
                         update_initial_state::Bool=false,
                         tinitx::Int=0,
                         V0::Union{Real,AbstractMatrix}=100.0)

    # Validate: must have MatrixExpr for Z
    if !haskey(spec.matrix_exprs, :Z)
        throw(ArgumentError(
            "profile_em_ssm requires spec.matrix_exprs[:Z] to be a MatrixExpr. " *
            "Use dns_model() or manually add a MatrixExpr for Z."
        ))
    end

    Z_expr = spec.matrix_exprs[:Z]

    # Validate λ_param exists in Z_expr
    λ_idx = findfirst(p -> p.name == λ_param, Z_expr.params)
    if λ_idx === nothing
        throw(ArgumentError("Parameter :$λ_param not found in Z MatrixExpr"))
    end

    p, n = size(y)
    m = spec.n_states
    r = spec.n_shocks

    # Build initial T, H, Q, a1, P1 from spec
    T_init, T_free = _extract_matrix_for_em(spec.T, spec.params, m, m)
    H_init, _ = _extract_matrix_for_em(spec.H, spec.params, p, p)
    Q_init, _ = _extract_Q_for_em(spec, r, r)
    R = _build_fixed_matrix(spec.R, r, r)
    a1, P1 = _extract_initial_state(spec, m)

    # Initial state covariance based on tinitx convention
    V0_mat = V0 isa Real ? V0 * Matrix(1.0I, m, m) : Matrix{Float64}(V0)
    if tinitx == 0
        # tinitx=0: V0 is covariance at t=0, compute P1 at t=1
        # P1 = T * V0 * T' + R * Q * R'
        P1 = T_init * V0_mat * T_init' + R * Q_init * R'
    elseif tinitx == 1
        # tinitx=1: V0 is covariance at t=1, use directly
        P1 = V0_mat
    else
        throw(ArgumentError("tinitx must be 0 or 1, got $tinitx"))
    end
    # Ensure P1 is symmetric and PSD
    P1 = 0.5 * (P1 + P1')
    P1 = Matrix(project_psd(P1, 1e-10))

    # For EM with full covariance, always use full free masks for H and Q
    H_free = trues(p, p)
    Q_free = trues(r, r)

    # Grid search storage
    λ_vec = collect(Float64, λ_grid)
    n_grid = length(λ_vec)
    loglik_profile = Vector{Float64}(undef, n_grid)

    # Store best result
    best_loglik = -Inf
    best_idx = 1
    best_em_result = nothing

    # Current parameter estimates for warm-starting
    T_curr = copy(T_init)
    H_curr = copy(H_init)
    Q_curr = copy(Q_init)

    for (idx, λ_val) in enumerate(λ_vec)
        if verbose
            println("Profile EM: λ = $(round(λ_val, digits=4)) ($idx/$n_grid)")
        end

        # Build Z matrix at this λ
        θ_λ = Dict(λ_param => λ_val)
        Z = Z_expr.builder(θ_λ, Z_expr.data)

        # Run EM with fixed Z
        em_result = _em_general_ssm_full_cov(
            Float64.(Z), T_curr, Float64.(R),
            H_curr, Q_curr,
            y, a1, P1;
            Z_free=falses(p, m),  # Z is fixed
            T_free=T_free,
            H_free=H_free,
            Q_free=Q_free,
            maxiter=maxiter,
            tol_ll=tol_ll,
            tol_param=1e-8,
            verbose=verbose, # Pass verbose down to see intermediate LL steps
            update_initial_state=update_initial_state
        )

        loglik_profile[idx] = em_result.loglik

        if verbose
            println("  Final loglik = $(round(em_result.loglik, digits=4)), " *
                    "converged = $(em_result.converged) ($(em_result.iterations) iters)")
        end

        # Track best
        if em_result.loglik > best_loglik
            best_loglik = em_result.loglik
            best_idx = idx
            best_em_result = em_result
        end

        # Warm start: use this solution as next starting point
        if warm_start
            T_curr = copy(em_result.T)
            H_curr = copy(em_result.H)
            Q_curr = copy(em_result.Q)
            # Also update initial state if we're estimating it
            if update_initial_state
                a1 = copy(em_result.a1)
                P1 = copy(em_result.P1)
            end
        end
    end

    λ_optimal = λ_vec[best_idx]

    if verbose
        println("\nOptimal λ = $(round(λ_optimal, digits=4)) with loglik = $(round(best_loglik, digits=4))")
    end

    # Build final θ NamedTuple with all parameters
    θ = _build_profile_em_theta(spec, λ_optimal, best_em_result, λ_param)

    ProfileEMResult(
        λ_optimal,
        θ,
        best_loglik,
        best_em_result,
        λ_vec,
        loglik_profile
    )
end

"""
Extract matrix and free mask from SSMMatrixSpec for EM.
"""
function _extract_matrix_for_em(mat_spec::SSMMatrixSpec, params::Vector{SSMParameter{Float64}},
                                 nrow::Int, ncol::Int)
    mat = zeros(nrow, ncol)
    free = falses(nrow, ncol)

    # Build parameter name -> init value map
    param_init = Dict(p.name => p.init for p in params)

    for i in 1:nrow, j in 1:ncol
        elem = get(mat_spec.elements, (i, j), mat_spec.default)
        if elem isa FixedValue
            mat[i, j] = elem.value
            free[i, j] = false
        elseif elem isa ParameterRef
            mat[i, j] = get(param_init, elem.name, 0.0)
            free[i, j] = true
        else
            # Expression or other - treat as fixed
            mat[i, j] = 0.0
            free[i, j] = false
        end
    end

    (mat, free)
end

"""
Extract Q matrix for EM, handling CovMatrixExpr case.
"""
function _extract_Q_for_em(spec::SSMSpec, nrow::Int, ncol::Int)
    if haskey(spec.matrix_exprs, :Q)
        Q_expr = spec.matrix_exprs[:Q]
        if Q_expr isa CovMatrixExpr
            Q = zeros(nrow, ncol)
            # Use stored variances if available (avoids sqrt/square roundoff)
            if !isempty(Q_expr.var_init)
                for i in 1:min(nrow, length(Q_expr.var_init))
                    Q[i, i] = Q_expr.var_init[i]
                end
            else
                # Fallback: compute from σ params (may have roundoff)
                param_init = Dict(p.name => p.init for p in spec.params)
                for i in 1:nrow
                    σ_name = Q_expr.σ_param_names[i]
                    σ_val = get(param_init, σ_name, 1.0)
                    Q[i, i] = σ_val^2
                end
            end
            # All elements free for full covariance
            Q_free = trues(nrow, ncol)
            return (Q, Q_free)
        end
    end

    # Standard SSMMatrixSpec case
    _extract_matrix_for_em(spec.Q, spec.params, nrow, ncol)
end

"""
Build fixed matrix from SSMMatrixSpec.
"""
function _build_fixed_matrix(mat_spec::SSMMatrixSpec, nrow::Int, ncol::Int)
    mat = zeros(nrow, ncol)
    for i in 1:nrow, j in 1:ncol
        elem = get(mat_spec.elements, (i, j), mat_spec.default)
        if elem isa FixedValue
            mat[i, j] = elem.value
        end
    end
    mat
end

"""
Extract initial state mean and covariance from spec.
"""
function _extract_initial_state(spec::SSMSpec, m::Int)
    a1 = zeros(m)
    for i in 1:m
        elem = spec.a1[i]
        if elem isa FixedValue
            a1[i] = elem.value
        end
    end

    P1 = zeros(m, m)
    for i in 1:m, j in 1:m
        elem = get(spec.P1.elements, (i, j), spec.P1.default)
        if elem isa FixedValue
            P1[i, j] = elem.value
        end
    end

    (a1, P1)
end

"""
Build NamedTuple of all parameters from profile EM result.
"""
function _build_profile_em_theta(spec::SSMSpec, λ_optimal::Real,
                                  em_result::NamedTuple, λ_param::Symbol)
    names = Symbol[]
    values = Float64[]

    # Add λ first
    push!(names, λ_param)
    push!(values, λ_optimal)

    # Add T matrix elements (row-major for readability)
    m = size(em_result.T, 1)
    T_params = _find_matrix_params(spec.T, spec.params)
    for (name, i, j) in T_params
        push!(names, name)
        push!(values, em_result.T[i, j])
    end

    # Add H matrix elements
    p = size(em_result.H, 1)
    H_params = _find_matrix_params(spec.H, spec.params)
    for (name, i, j) in H_params
        push!(names, name)
        push!(values, em_result.H[i, j])
    end

    # Add Q parameters
    if haskey(spec.matrix_exprs, :Q)
        Q_expr = spec.matrix_exprs[:Q]
        if Q_expr isa CovMatrixExpr
            # Extract σ values and correlations from EM result
            r = Q_expr.n
            Q_em = em_result.Q

            # Compute σ values from diagonal
            for (idx, σ_name) in enumerate(Q_expr.σ_param_names)
                push!(names, σ_name)
                push!(values, sqrt(max(Q_em[idx, idx], 0.0)))
            end

            # Compute correlation parameters
            # We store the off-diagonal elements scaled by sqrt(var_i * var_j)
            for (idx, corr_name) in enumerate(Q_expr.corr_param_names)
                push!(names, corr_name)
                # Correlations are harder to extract; store 0 as placeholder
                # In practice, the full Q matrix is what matters
                push!(values, 0.0)
            end
        end
    else
        Q_params = _find_matrix_params(spec.Q, spec.params)
        for (name, i, j) in Q_params
            push!(names, name)
            push!(values, em_result.Q[i, j])
        end
    end

    NamedTuple{Tuple(names)}(Tuple(values))
end

"""
Find parameters in a matrix spec, returning (name, row, col) tuples.
"""
function _find_matrix_params(mat_spec::SSMMatrixSpec, params::Vector{SSMParameter{Float64}})
    param_names_set = Set(p.name for p in params)
    result = Tuple{Symbol,Int,Int}[]

    nrow, ncol = mat_spec.dims
    for i in 1:nrow, j in 1:ncol
        elem = get(mat_spec.elements, (i, j), mat_spec.default)
        if elem isa ParameterRef && elem.name in param_names_set
            push!(result, (elem.name, i, j))
        end
    end

    result
end
