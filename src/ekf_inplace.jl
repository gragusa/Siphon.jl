"""
    ekf_inplace.jl

In-place Extended Kalman Filter workspace and `ekf_filter!`.

Mirrors `KalmanWorkspace` / `kalman_filter!` in `inplace.jl` with two changes:

  1. The constant `Z` is replaced by a per-step Jacobian `Z_t = ∂h/∂α` written
     into `tmp_Zt`. `Z_t` is also stored in `Zt[:, :, t]` for downstream use
     (smoother, EM).
  2. The linear `Z * a` is replaced by `measurement!(tmp_yhat, model, a, t)`.

Jacobian dispatch:

  - `AnalyticJacobian()` — calls `measurement_jacobian!(tmp_Zt, model, a, t)`.
    No allocations beyond what the user method does.
  - `ADJacobian()` — uses a pre-allocated `ForwardDiff.JacobianConfig` stored
    in the workspace, plus a dual output buffer. The config is built once at
    construction time so the time loop is allocation-free.

The workspace, not the measurement object, owns `tmp_yhat`, `tmp_Zt`, and the
AD cache. This keeps measurement objects immutable and thread-safe.
"""

using LinearAlgebra
using ForwardDiff

# ============================================================================
# AD cache
# ============================================================================

"""
    EKFADCache{F, Cfg, YBuf}

Workspace-owned cache for `ForwardDiff`-based Jacobian evaluation. Holds the
in-place closure `f!`, a `JacobianConfig` keyed to that closure, and a real-
valued output buffer of length `p` that ForwardDiff requires to know the
output shape. Constructed once per workspace.

`ADJacobian()` workspace construction allocates this cache; subsequent
`ekf_filter!` calls reuse the cfg's internal dual buffers — the only per-step
allocations are those inside the user's `measurement!` body when it operates
on duals (e.g. a temporary state-vector product the user code computes).
"""
struct EKFADCache{F, Cfg, YBuf}
    f!::F                # closure: (out, x) -> measurement!(out, model, x, t_ref[])
    t_ref::Base.RefValue{Int}
    cfg::Cfg
    y_buf::YBuf          # real-valued output buffer (length p) for FD's in-place API
end

# Build an AD cache using ForwardDiff's in-place form. The cfg internally
# manages dual buffers for both the input and output sides; we just hand it a
# real-valued output buffer (`y_buf`) so it knows the output dimension.
function _build_ad_cache(measurement_model, p::Int, m::Int, ::Type{T}) where {T}
    t_ref = Ref(1)
    f! = let model = measurement_model, tref = t_ref
        (out, x) -> measurement!(out, model, x, tref[])
    end

    x_template = zeros(T, m)
    y_buf = Vector{T}(undef, p)
    cfg = ForwardDiff.JacobianConfig(f!, y_buf, x_template)

    return EKFADCache{typeof(f!), typeof(cfg), typeof(y_buf)}(f!, t_ref, cfg, y_buf)
end

# ============================================================================
# EKFWorkspace
# ============================================================================

"""
    EKFWorkspace{T<:Real, M, J, Cache}

In-place workspace for Extended Kalman Filter operations. Mirrors
`KalmanWorkspace` field-for-field except:

  - `Z::Matrix{T}` (constant in linear) is replaced by `Zt::Array{T,3}` storing
    per-step Jacobians.
  - Adds `yt::Matrix{T}` for predicted measurements.
  - Adds `measurement::M` (the `AbstractEKFMeasurement` object) and
    `jacobian_mode::J`.
  - Adds workspace-owned scratch `tmp_yhat`, `tmp_Zt` for the per-step
    measurement and Jacobian.
  - For `ADJacobian()`, holds an `EKFADCache` used by `ForwardDiff.jacobian!`.

Construct via `EKFWorkspace(p::EKFParms, a1, P1, n)` or by passing the system
matrices and measurement object directly.
"""
mutable struct EKFWorkspace{T <: Real, M, J <: AbstractEKFJacobianMode, Cache}
    # Dimensions
    obs_dim::Int      # p
    state_dim::Int    # m
    shock_dim::Int    # r
    n_times::Int      # n

    # Measurement model + jacobian mode
    measurement::M
    jacobian_mode::J

    # System matrices (owned copies)
    H::Matrix{T}      # p × p
    Tmat::Matrix{T}   # m × m
    R::Matrix{T}      # m × r
    Q::Matrix{T}      # r × r
    a1::Vector{T}     # m
    P1::Matrix{T}     # m × m

    # Precomputed
    RQR::Matrix{T}    # m × m

    # Filter storage
    at::Matrix{T}         # m × n
    Pt::Array{T, 3}        # m × m × n
    att::Matrix{T}        # m × n
    Ptt::Array{T, 3}       # m × m × n
    yt::Matrix{T}         # p × n   predicted measurement h(a_t, t)
    Zt::Array{T, 3}        # p × m × n  measurement Jacobians
    vt::Matrix{T}         # p × n
    Ft::Array{T, 3}        # p × p × n
    Ft_L::Array{T, 3}      # p × p × n  Cholesky factors (lower)
    Kt::Array{T, 3}        # m × p × n
    missing_mask::BitVector

    # Smoother storage (allocated for parity with KalmanWorkspace; populated
    # only after a smoother pass is implemented).
    αs::Matrix{T}
    Vs::Array{T, 3}
    Pcross::Array{T, 3}

    # Scratch
    tmp_mm1::Matrix{T}
    tmp_mm2::Matrix{T}
    tmp_mm3::Matrix{T}
    tmp_pp1::Matrix{T}
    tmp_pp2::Matrix{T}
    tmp_mp::Matrix{T}
    tmp_pm::Matrix{T}
    tmp_mr::Matrix{T}
    tmp_m1::Vector{T}
    tmp_m2::Vector{T}
    tmp_p1::Vector{T}
    tmp_p2::Vector{T}

    # EKF-specific scratch
    tmp_yhat::Vector{T}   # p     predicted measurement scratch
    tmp_Zt::Matrix{T}     # p × m measurement Jacobian scratch
    ad_cache::Cache       # AD config (or `nothing`)

    # Smoother scratch
    r_smooth::Vector{T}
    N_smooth::Matrix{T}
    L_smooth::Matrix{T}
    J_smooth::Matrix{T}

    # Scalars
    loglik::T
    n_obs_valid::Int
end

# ============================================================================
# Constructors
# ============================================================================

"""
    EKFWorkspace{T}(measurement, jacobian_mode, H, Tmat, R, Q, a1, P1, n)

Construct an `EKFWorkspace` from system matrices and a measurement model.

Dimensions are inferred from `H` (p), `Tmat` (m), and `Q` (r). For
`ADJacobian()`, builds a `ForwardDiff.JacobianConfig` once and stores it; for
`AnalyticJacobian()` the cache is `nothing`.
"""
function EKFWorkspace{T}(
        measurement_model,
        jacobian_mode::AbstractEKFJacobianMode,
        H::AbstractMatrix,
        Tmat::AbstractMatrix,
        R::AbstractMatrix,
        Q::AbstractMatrix,
        a1::AbstractVector,
        P1::AbstractMatrix,
        n::Int
) where {T <: Real}
    p = size(H, 1)
    m = size(Tmat, 1)
    r = size(Q, 1)
    @assert size(H) == (p, p)
    @assert size(Tmat) == (m, m)
    @assert size(R) == (m, r)
    @assert size(Q) == (r, r)
    @assert length(a1) == m
    @assert size(P1) == (m, m)

    # Verify the measurement protocol is consistent with jacobian_mode.
    _check_measurement_protocol(measurement_model, jacobian_mode, m, p, T)

    # Build AD cache once if requested.
    cache = if jacobian_mode isa ADJacobian
        _build_ad_cache(measurement_model, p, m, T)
    else
        nothing
    end

    ws = EKFWorkspace{T, typeof(measurement_model), typeof(jacobian_mode), typeof(cache)}(
        p, m, r, n,
        measurement_model, jacobian_mode,
        Matrix{T}(undef, p, p),     # H
        Matrix{T}(undef, m, m),     # Tmat
        Matrix{T}(undef, m, r),     # R
        Matrix{T}(undef, r, r),     # Q
        Vector{T}(undef, m),        # a1
        Matrix{T}(undef, m, m),     # P1
        Matrix{T}(undef, m, m),     # RQR
        Matrix{T}(undef, m, n),     # at
        Array{T, 3}(undef, m, m, n),# Pt
        Matrix{T}(undef, m, n),     # att
        Array{T, 3}(undef, m, m, n),# Ptt
        Matrix{T}(undef, p, n),     # yt
        Array{T, 3}(undef, p, m, n),# Zt
        Matrix{T}(undef, p, n),     # vt
        Array{T, 3}(undef, p, p, n),# Ft
        Array{T, 3}(undef, p, p, n),# Ft_L
        Array{T, 3}(undef, m, p, n),# Kt
        BitVector(undef, n),        # missing_mask
        Matrix{T}(undef, m, n),     # αs
        Array{T, 3}(undef, m, m, n),# Vs
        Array{T, 3}(undef, m, m, max(n - 1, 1)),# Pcross
        Matrix{T}(undef, m, m),     # tmp_mm1
        Matrix{T}(undef, m, m),     # tmp_mm2
        Matrix{T}(undef, m, m),     # tmp_mm3
        Matrix{T}(undef, p, p),     # tmp_pp1
        Matrix{T}(undef, p, p),     # tmp_pp2
        Matrix{T}(undef, m, p),     # tmp_mp
        Matrix{T}(undef, p, m),     # tmp_pm
        Matrix{T}(undef, m, r),     # tmp_mr
        Vector{T}(undef, m),        # tmp_m1
        Vector{T}(undef, m),        # tmp_m2
        Vector{T}(undef, p),        # tmp_p1
        Vector{T}(undef, p),        # tmp_p2
        Vector{T}(undef, p),        # tmp_yhat
        Matrix{T}(undef, p, m),     # tmp_Zt
        cache,                       # ad_cache
        Vector{T}(undef, m),        # r_smooth
        Matrix{T}(undef, m, m),     # N_smooth
        Matrix{T}(undef, m, m),     # L_smooth
        Matrix{T}(undef, m, m),     # J_smooth
        zero(T),                    # loglik
        0                           # n_obs_valid
    )

    set_params!(ws, H, Tmat, R, Q)
    set_initial!(ws, a1, P1)
    return ws
end

# Default T = Float64
function EKFWorkspace(
        measurement_model,
        jacobian_mode::AbstractEKFJacobianMode,
        H::AbstractMatrix,
        Tmat::AbstractMatrix,
        R::AbstractMatrix,
        Q::AbstractMatrix,
        a1::AbstractVector,
        P1::AbstractMatrix,
        n::Int
)
    EKFWorkspace{Float64}(measurement_model, jacobian_mode, H, Tmat, R, Q,
        a1, P1, n)
end

"""
    EKFWorkspace(p::EKFParms, a1, P1, n)

Construct from an `EKFParms` and initial state.
"""
function EKFWorkspace(p::EKFParms, a1::AbstractVector, P1::AbstractMatrix, n::Int)
    EKFWorkspace(p.measurement, p.jacobian_mode, p.H, p.T, p.R, p.Q, a1, P1, n)
end

# ============================================================================
# Protocol checks
# ============================================================================

function _check_measurement_protocol(
        model, ::AnalyticJacobian, m::Int, p::Int, ::Type{T}) where {T}
    a_probe = zeros(T, m)
    out_probe = zeros(T, p)
    Z_probe = zeros(T, p, m)
    if !hasmethod(measurement!, Tuple{
        typeof(out_probe), typeof(model), typeof(a_probe), Int})
        error("EKFWorkspace with AnalyticJacobian() requires `measurement!(out, model, a, t)` to be defined for $(typeof(model)).")
    end
    if !hasmethod(measurement_jacobian!, Tuple{
        typeof(Z_probe), typeof(model), typeof(a_probe), Int})
        error("EKFWorkspace with AnalyticJacobian() requires `measurement_jacobian!(Z, model, a, t)` to be defined for $(typeof(model)).")
    end
    return nothing
end

function _check_measurement_protocol(
        model, ::ADJacobian, m::Int, p::Int, ::Type{T}) where {T}
    a_probe = zeros(T, m)
    out_probe = zeros(T, p)
    if !hasmethod(measurement, Tuple{typeof(model), typeof(a_probe), Int})
        error("EKFWorkspace with ADJacobian() requires the pure functional `measurement(model, a, t)` to be defined for $(typeof(model)).")
    end
    # The mutating measurement! is also required so ekf_filter! can write into
    # tmp_yhat without allocating per step.
    if !hasmethod(measurement!, Tuple{
        typeof(out_probe), typeof(model), typeof(a_probe), Int})
        error("EKFWorkspace with ADJacobian() also requires `measurement!(out, model, a, t)` for the in-place residual computation.")
    end
    return nothing
end

function _check_measurement_protocol(
        model, ::FiniteDiffJacobian, m::Int, p::Int, ::Type{T}) where {T}
    error("FiniteDiffJacobian() is diagnostic only and not supported in EKFWorkspace.")
end

# ============================================================================
# Setters
# ============================================================================

function _update_RQR!(ws::EKFWorkspace{T}) where {T}
    mul!(ws.tmp_mr, ws.R, ws.Q)
    mul!(ws.RQR, ws.tmp_mr, ws.R')
    return nothing
end

"""
    set_params!(ws::EKFWorkspace, H, Tmat, R, Q)

Set the linear-system matrices. The measurement model is set at construction
time and not modified by this method (use `update_params!` to swap it).
"""
function set_params!(
        ws::EKFWorkspace,
        H::AbstractMatrix,
        Tmat::AbstractMatrix,
        R::AbstractMatrix,
        Q::AbstractMatrix
)
    copyto!(ws.H, H)
    copyto!(ws.Tmat, Tmat)
    copyto!(ws.R, R)
    copyto!(ws.Q, Q)
    _update_RQR!(ws)
    return ws
end

function set_initial!(ws::EKFWorkspace, a1::AbstractVector, P1::AbstractMatrix)
    copyto!(ws.a1, a1)
    copyto!(ws.P1, P1)
    return ws
end

"""
    update_params!(ws::EKFWorkspace; H=nothing, Tmat=nothing, R=nothing, Q=nothing)

Selective update of the linear-system matrices. Recomputes RQR if R or Q
change. The measurement model is held by reference; mutate or swap it via
direct assignment to `ws.measurement` if needed (and rebuild the AD cache if
the measurement structure changes).
"""
function update_params!(
        ws::EKFWorkspace;
        H::Union{Nothing, AbstractMatrix} = nothing,
        Tmat::Union{Nothing, AbstractMatrix} = nothing,
        R::Union{Nothing, AbstractMatrix} = nothing,
        Q::Union{Nothing, AbstractMatrix} = nothing
)
    H !== nothing && copyto!(ws.H, H)
    Tmat !== nothing && copyto!(ws.Tmat, Tmat)
    update_RQR = false
    if R !== nothing
        copyto!(ws.R, R)
        update_RQR = true
    end
    if Q !== nothing
        copyto!(ws.Q, Q)
        update_RQR = true
    end
    update_RQR && _update_RQR!(ws)
    return ws
end

# ============================================================================
# Per-step Jacobian dispatch (in-place)
# ============================================================================

@inline function _ekf_jacobian!(ws::EKFWorkspace, ::AnalyticJacobian, a, t::Int)
    measurement_jacobian!(ws.tmp_Zt, ws.measurement, a, t)
    return ws.tmp_Zt
end

@inline function _ekf_jacobian!(ws::EKFWorkspace, ::ADJacobian, a, t::Int)
    cache = ws.ad_cache::EKFADCache
    cache.t_ref[] = t
    # In-place form: cfg owns both input and output dual buffers, so the only
    # per-step allocations are whatever the user's measurement! body itself
    # does on Dual inputs (e.g. a temporary state-vector product).
    ForwardDiff.jacobian!(ws.tmp_Zt, cache.f!, cache.y_buf, a, cache.cfg)
    return ws.tmp_Zt
end

# ============================================================================
# In-place EKF
# ============================================================================

"""
    ekf_filter!(ws::EKFWorkspace, y::AbstractMatrix) -> loglik

Run the Extended Kalman Filter in place, storing per-step quantities in `ws`.

Behaves as `kalman_filter!` (in-place linear filter) with `Z` replaced by a
per-step Jacobian `tmp_Zt` and the measurement linearisation stored in
`ws.Zt[:, :, t]`.
"""
function ekf_filter!(ws::EKFWorkspace{T}, y::AbstractMatrix) where {T}
    p, m, n = ws.obs_dim, ws.state_dim, ws.n_times
    @assert size(y) == (p, n) "Observation matrix size mismatch"

    ws.loglik = zero(T)
    ws.n_obs_valid = 0

    a_curr = ws.tmp_m1
    P_curr = ws.tmp_mm1
    copyto!(a_curr, ws.a1)
    copyto!(P_curr, ws.P1)

    log2pi = log(T(2π))
    Tt = transpose(ws.Tmat)

    @inbounds for t in 1:n
        copyto!(view(ws.at, :, t), a_curr)
        copyto!(view(ws.Pt, :, :, t), P_curr)

        y_t = view(y, :, t)

        # Always evaluate ŷ and Z at the predicted state — even for missing,
        # so that yt and Zt are populated for diagnostics. (Skip if user wants
        # to save compute? The plan suggests `store_jacobians = false` as a
        # later optimisation; for now, store unconditionally.)
        measurement!(ws.tmp_yhat, ws.measurement, a_curr, t)
        _ekf_jacobian!(ws, ws.jacobian_mode, a_curr, t)
        copyto!(view(ws.yt, :, t), ws.tmp_yhat)
        copyto!(view(ws.Zt, :, :, t), ws.tmp_Zt)

        if _has_missing(y_t)
            ws.missing_mask[t] = true
            fill!(view(ws.vt, :, t), T(NaN))
            fill!(view(ws.Ft, :, :, t), T(NaN))
            fill!(view(ws.Ft_L, :, :, t), zero(T))
            fill!(view(ws.Kt, :, :, t), zero(T))
            copyto!(view(ws.att, :, t), a_curr)
            copyto!(view(ws.Ptt, :, :, t), P_curr)

            mul!(ws.tmp_m2, ws.Tmat, a_curr)
            copyto!(a_curr, ws.tmp_m2)
            mul!(ws.tmp_mm2, ws.Tmat, P_curr)
            mul!(ws.tmp_mm3, ws.tmp_mm2, Tt)
            @inbounds for idx in eachindex(P_curr)
                P_curr[idx] = ws.tmp_mm3[idx] + ws.RQR[idx]
            end
            continue
        end

        ws.missing_mask[t] = false
        ws.n_obs_valid += 1

        # Innovation: v = y - ŷ
        v_t = view(ws.vt, :, t)
        @inbounds for i in 1:p
            v_t[i] = y_t[i] - ws.tmp_yhat[i]
        end

        # F = Z * P * Z' + H
        # tmp_pm = Z * P (p × m)
        mul!(ws.tmp_pm, ws.tmp_Zt, P_curr)
        # tmp_pp1 = (Z * P) * Z' (p × p)
        Ztp = transpose(ws.tmp_Zt)
        mul!(ws.tmp_pp1, ws.tmp_pm, Ztp)
        Ft_view = view(ws.Ft, :, :, t)
        @inbounds for idx in eachindex(ws.tmp_pp1)
            ws.tmp_pp1[idx] += ws.H[idx]
            Ft_view[idx] = ws.tmp_pp1[idx]
        end

        cholF = cholesky!(Symmetric(ws.tmp_pp1, :L))
        L_lower = LowerTriangular(cholF.factors)

        # Store Cholesky lower triangle (zero upper for clean reuse)
        FtL_view = view(ws.Ft_L, :, :, t)
        @inbounds for j in 1:p
            for i in 1:(j - 1)
                FtL_view[i, j] = zero(T)
            end
            for i in j:p
                FtL_view[i, j] = L_lower[i, j]
            end
        end

        # log|F| = 2 Σ log L[i,i]
        logdetF = zero(T)
        @inbounds for i in 1:p
            logdetF += log(L_lower[i, i])
        end
        logdetF += logdetF

        # quad form via L⁻¹ v
        copyto!(ws.tmp_p2, v_t)
        ldiv!(L_lower, ws.tmp_p2)
        quad_form = zero(T)
        @inbounds for i in 1:p
            quad_form += ws.tmp_p2[i]^2
        end
        ws.loglik += -T(0.5) * (logdetF + quad_form)

        # M = P * Z' * F⁻¹  via two solves on Z*P, transposed.
        # tmp_pm currently holds Z * P.
        ldiv!(L_lower, ws.tmp_pm)
        ldiv!(transpose(L_lower), ws.tmp_pm)
        # M = (F⁻¹ Z P)' transposed into tmp_mp (m × p)
        transpose!(ws.tmp_mp, ws.tmp_pm)

        # K_t = T * M
        K_t = view(ws.Kt, :, :, t)
        mul!(K_t, ws.Tmat, ws.tmp_mp)

        # a_filt = a + M * v
        mul!(ws.tmp_m2, ws.tmp_mp, v_t)
        att_view = view(ws.att, :, t)
        @inbounds for i in 1:m
            att_view[i] = a_curr[i] + ws.tmp_m2[i]
        end

        # P_filt = P - M * (Z * P). Recompute Z*P (we trashed tmp_pm above).
        mul!(ws.tmp_pm, ws.tmp_Zt, P_curr)
        mul!(ws.tmp_mm2, ws.tmp_mp, ws.tmp_pm)
        Ptt_view = view(ws.Ptt, :, :, t)
        @inbounds for idx in eachindex(P_curr)
            Ptt_view[idx] = P_curr[idx] - ws.tmp_mm2[idx]
        end

        # Predict next: a = T * a_filt, P = T * P_filt * T' + RQR
        mul!(a_curr, ws.Tmat, att_view)
        mul!(ws.tmp_mm2, ws.Tmat, Ptt_view)
        mul!(P_curr, ws.tmp_mm2, Tt)
        @inbounds for idx in eachindex(P_curr)
            P_curr[idx] += ws.RQR[idx]
        end
    end

    ws.loglik += -p * ws.n_obs_valid * log2pi / 2
    return ws.loglik
end

# ============================================================================
# In-place EKF smoother (extended RTS)
# ============================================================================

"""
    ekf_smoother!(ws::EKFWorkspace; crosscov::Bool=true)

Run the extended RTS smoother in place using filter results stored in `ws`.
Must be called after `ekf_filter!(ws, y)`.

The recursion is identical to the linear smoother with the constant `Z`
replaced by the per-step linearisation `Z_t = ws.Zt[:, :, t]` produced during
the forward filter pass. This is the standard EKF smoother approximation:
smoothing is performed on the time-varying linear-Gaussian approximation
generated by the forward EKF.

# Arguments
- `ws::EKFWorkspace` — workspace populated by `ekf_filter!`.
- `crosscov` — also compute lag-1 cross-covariances `Cov[α_t, α_{t-1} | y_{1:n}]`
  used by EM. Default `true`.

# Stored
- `ws.αs` — smoothed state means.
- `ws.Vs` — smoothed state covariances.
- `ws.Pcross` — lag-1 cross-covariances (when `crosscov=true`).
"""
function ekf_smoother!(ws::EKFWorkspace{T}; crosscov::Bool = true) where {T}
    m, n = ws.state_dim, ws.n_times
    p = ws.obs_dim

    fill!(ws.r_smooth, zero(T))
    fill!(ws.N_smooth, zero(T))

    @inbounds for t in n:-1:1
        a_t = view(ws.at, :, t)
        P_t = view(ws.Pt, :, :, t)
        Z_t = view(ws.Zt, :, :, t)

        if ws.missing_mask[t]
            # Missing: r_{t-1} = T' r_t,  N_{t-1} = T' N_t T
            mul!(ws.tmp_m1, ws.Tmat', ws.r_smooth)
            copyto!(ws.r_smooth, ws.tmp_m1)

            mul!(ws.tmp_mm1, ws.Tmat', ws.N_smooth)
            mul!(ws.tmp_mm2, ws.tmp_mm1, ws.Tmat)
            copyto!(ws.N_smooth, ws.tmp_mm2)

            # α = a + P r
            mul!(ws.tmp_m1, P_t, ws.r_smooth)
            for i in 1:m
                ws.αs[i, t] = a_t[i] + ws.tmp_m1[i]
            end
            # V = P - P N P
            mul!(ws.tmp_mm1, ws.N_smooth, P_t)
            mul!(ws.tmp_mm2, P_t, ws.tmp_mm1)
            for j in 1:m, i in 1:m

                ws.Vs[i, j, t] = P_t[i, j] - ws.tmp_mm2[i, j]
            end
        else
            v_t = view(ws.vt, :, t)
            K_t = view(ws.Kt, :, :, t)
            L_lower = LowerTriangular(view(ws.Ft_L, :, :, t))

            # L_t = T - K_t Z_t  (uses time-varying Z_t)
            mul!(ws.L_smooth, K_t, Z_t)
            for j in 1:m, i in 1:m

                ws.L_smooth[i, j] = ws.Tmat[i, j] - ws.L_smooth[i, j]
            end

            # F^{-1} v
            copyto!(ws.tmp_p1, v_t)
            ldiv!(L_lower, ws.tmp_p1)
            ldiv!(L_lower', ws.tmp_p1)

            # F^{-1} Z_t
            copyto!(ws.tmp_pm, Z_t)
            ldiv!(L_lower, ws.tmp_pm)
            ldiv!(L_lower', ws.tmp_pm)

            # r_{t-1} = Z_t' F^{-1} v + L_t' r_t
            mul!(ws.tmp_m1, Z_t', ws.tmp_p1)
            mul!(ws.tmp_m2, ws.L_smooth', ws.r_smooth)
            for i in 1:m
                ws.r_smooth[i] = ws.tmp_m1[i] + ws.tmp_m2[i]
            end

            # N_{t-1} = Z_t' F^{-1} Z_t + L_t' N L_t
            mul!(ws.tmp_mm1, Z_t', ws.tmp_pm)
            mul!(ws.tmp_mm2, ws.L_smooth', ws.N_smooth)
            mul!(ws.tmp_mm3, ws.tmp_mm2, ws.L_smooth)
            for j in 1:m, i in 1:m

                ws.N_smooth[i, j] = ws.tmp_mm1[i, j] + ws.tmp_mm3[i, j]
            end

            # α = a + P r
            mul!(ws.tmp_m1, P_t, ws.r_smooth)
            for i in 1:m
                ws.αs[i, t] = a_t[i] + ws.tmp_m1[i]
            end
            # V = P - P N P
            mul!(ws.tmp_mm1, ws.N_smooth, P_t)
            mul!(ws.tmp_mm2, P_t, ws.tmp_mm1)
            for j in 1:m, i in 1:m

                ws.Vs[i, j, t] = P_t[i, j] - ws.tmp_mm2[i, j]
            end
        end

        # Lag-1 cross-covariance via Shumway–Stoffer:
        # P_{t, t-1 | n} = V_t * J_{t-1}'  with  J_{t-1} = P_{t-1|t-1} T' P_{t|t-1}^{-1}.
        if crosscov && t > 1
            V_t = view(ws.Vs, :, :, t)
            Ptt_tm1 = view(ws.Ptt, :, :, (t - 1))
            Pt_t = view(ws.Pt, :, :, t)

            # tmp_mm1 = Ptt_{t-1} * T'
            mul!(ws.tmp_mm1, Ptt_tm1, ws.Tmat')

            # Cholesky on a (symmetrised + lightly regularised) copy of Pt_t
            copyto!(ws.tmp_mm2, Pt_t)
            eps_reg = T(1e-10) * max(one(T), tr(ws.tmp_mm2) / m)
            for j in 1:m
                for i in 1:(j - 1)
                    avg = (ws.tmp_mm2[i, j] + ws.tmp_mm2[j, i]) / 2
                    ws.tmp_mm2[i, j] = avg
                    ws.tmp_mm2[j, i] = avg
                end
                ws.tmp_mm2[j, j] += eps_reg
            end
            chol_P = cholesky!(Symmetric(ws.tmp_mm2, :L))

            # J_smooth := J' so that V_t * J_smooth = P_{t, t-1 | n}
            for j in 1:m, i in 1:m

                ws.J_smooth[i, j] = ws.tmp_mm1[j, i]
            end
            L_P = LowerTriangular(chol_P.L)
            for j in 1:m
                col = view(ws.J_smooth, :, j)
                ldiv!(L_P, col)
                ldiv!(L_P', col)
            end
            Pcross_t = view(ws.Pcross, :, :, (t - 1))
            mul!(Pcross_t, V_t, ws.J_smooth)
        end
    end

    return nothing
end

"""
    ekf_filter_and_smooth!(ws::EKFWorkspace, y; crosscov=true) -> loglik

Run `ekf_filter!` followed by `ekf_smoother!`.
"""
function ekf_filter_and_smooth!(ws::EKFWorkspace, y::AbstractMatrix; crosscov::Bool = true)
    ll = ekf_filter!(ws, y)
    ekf_smoother!(ws; crosscov = crosscov)
    return ll
end

# ============================================================================
# Accessors (mirror KalmanWorkspace where possible)
# ============================================================================

predicted_states(ws::EKFWorkspace) = ws.at
variances_predicted_states(ws::EKFWorkspace) = ws.Pt
filtered_states(ws::EKFWorkspace) = ws.att
variances_filtered_states(ws::EKFWorkspace) = ws.Ptt
prediction_errors(ws::EKFWorkspace) = ws.vt
variances_prediction_errors(ws::EKFWorkspace) = ws.Ft
kalman_gains(ws::EKFWorkspace) = ws.Kt
loglikelihood(ws::EKFWorkspace) = ws.loglik
predicted_observations(ws::EKFWorkspace) = ws.yt
measurement_jacobians(ws::EKFWorkspace) = ws.Zt
smoothed_states(ws::EKFWorkspace) = ws.αs
variances_smoothed_states(ws::EKFWorkspace) = ws.Vs
crosslag_covs(ws::EKFWorkspace) = ws.Pcross
