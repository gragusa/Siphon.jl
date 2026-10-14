"""
    ekf_inplace.jl

In-place Extended Kalman Filter workspace and `kalman_filter!`.

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
`kalman_filter!` calls reuse the cfg's internal dual buffers — the only per-step
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
mutable struct EKFWorkspace{T <: Real, M, J <: AbstractEKFJacobianMode, Cache} <:
               AbstractFilterWorkspace
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

    # Smoother storage, filled by kalman_smoother!
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

    # Scalars
    loglik::T
    n_obs_valid::Int

    # Run status, as in KalmanWorkspace
    status::Int
    smoothed::Bool
    crosscov::Bool
end

# ============================================================================
# Constructors
# ============================================================================

"""
    EKFWorkspace(measurement, jacobian_mode, H, Tmat, R, Q, a1, P1, n)
    EKFWorkspace{T}(measurement, jacobian_mode, H, Tmat, R, Q, a1, P1, n)

Construct an `EKFWorkspace` from system matrices and a measurement model. The
element type is `T` when given, and otherwise the floating-point type that holds
`H`, `Tmat`, `R`, `Q`, `a1` and `P1`.

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
    _check_size("EKFWorkspace", "H", H, (p, p))
    _check_size("EKFWorkspace", "T", Tmat, (m, m))
    _check_size("EKFWorkspace", "R", R, (m, r))
    _check_size("EKFWorkspace", "Q", Q, (r, r))
    _check_size("EKFWorkspace", "a1", a1, (m,))
    _check_size("EKFWorkspace", "P1", P1, (m, m))

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
        zeros(T, p, p),     # H
        zeros(T, m, m),     # Tmat
        zeros(T, m, r),     # R
        zeros(T, r, r),     # Q
        zeros(T, m),        # a1
        zeros(T, m, m),     # P1
        zeros(T, m, m),     # RQR
        zeros(T, m, n),     # at
        zeros(T, m, m, n),# Pt
        zeros(T, m, n),     # att
        zeros(T, m, m, n),# Ptt
        zeros(T, p, n),     # yt
        zeros(T, p, m, n),# Zt
        zeros(T, p, n),     # vt
        zeros(T, p, p, n),# Ft
        zeros(T, p, p, n),# Ft_L
        zeros(T, m, p, n),# Kt
        falses(n),        # missing_mask
        zeros(T, m, n),     # αs
        zeros(T, m, m, n),# Vs
        zeros(T, m, m, max(n - 1, 1)),# Pcross
        zeros(T, m, m),     # tmp_mm1
        zeros(T, m, m),     # tmp_mm2
        zeros(T, m, m),     # tmp_mm3
        zeros(T, p, p),     # tmp_pp1
        zeros(T, p, p),     # tmp_pp2
        zeros(T, m, p),     # tmp_mp
        zeros(T, p, m),     # tmp_pm
        zeros(T, m, r),     # tmp_mr
        zeros(T, m),        # tmp_m1
        zeros(T, m),        # tmp_m2
        zeros(T, p),        # tmp_p1
        zeros(T, p),        # tmp_p2
        zeros(T, p),        # tmp_yhat
        zeros(T, p, m),     # tmp_Zt
        cache,                       # ad_cache
        zeros(T, m),        # r_smooth
        zeros(T, m, m),     # N_smooth
        zeros(T, m, m),     # L_smooth
        zero(T),                    # loglik
        0,                          # n_obs_valid
        _NOT_RUN,                   # status
        false,                      # smoothed
        false                       # crosscov
    )

    set_params!(ws, H, Tmat, R, Q)
    set_initial!(ws, a1, P1)
    return ws
end

# The element type holds every system array; the measurement model does not take part.
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
    EKFWorkspace{_float_eltype(H, Tmat, R, Q, a1, P1)}(measurement_model, jacobian_mode,
        H, Tmat, R, Q, a1, P1, n)
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
    # The mutating measurement! is also required so kalman_filter! can write into
    # tmp_yhat without allocating per step.
    if !hasmethod(measurement!, Tuple{
        typeof(out_probe), typeof(model), typeof(a_probe), Int})
        error("EKFWorkspace with ADJacobian() also requires `measurement!(out, model, a, t)` for the in-place residual computation.")
    end
    return nothing
end

function _check_measurement_protocol(
        model, ::FiniteDiffJacobian, m::Int, p::Int, ::Type{T}) where {T}
    a_probe = zeros(T, m)
    out_probe = zeros(T, p)
    if !hasmethod(measurement!, Tuple{
        typeof(out_probe), typeof(model), typeof(a_probe), Int})
        error("EKFWorkspace with FiniteDiffJacobian() requires `measurement!(out, model, a, t)` to be defined for $(typeof(model)).")
    end
    return nothing
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
    _check_params(ws, "set_params!"; H, Tmat, R, Q)
    copyto!(ws.H, H)
    copyto!(ws.Tmat, Tmat)
    copyto!(ws.R, R)
    copyto!(ws.Q, Q)
    _update_RQR!(ws)
    return ws
end

function set_initial!(ws::EKFWorkspace, a1::AbstractVector, P1::AbstractMatrix)
    _check_initial(ws, "set_initial!", a1, P1)
    copyto!(ws.a1, a1)
    copyto!(ws.P1, P1)
    return ws
end

"""
    update_params!(ws::EKFWorkspace; H=nothing, T=nothing, R=nothing, Q=nothing)

Selective update of the linear-system matrices. Recomputes RQR if R or Q
change. The measurement model is held by reference; mutate or swap it via
direct assignment to `ws.measurement` if needed (and rebuild the AD cache if
the measurement structure changes).
"""
function update_params!(
        ws::EKFWorkspace;
        H::Union{Nothing, AbstractMatrix} = nothing,
        T::Union{Nothing, AbstractMatrix} = nothing,
        R::Union{Nothing, AbstractMatrix} = nothing,
        Q::Union{Nothing, AbstractMatrix} = nothing
)
    _check_params(ws, "update_params!"; H, Tmat = T, R, Q)
    H !== nothing && copyto!(ws.H, H)
    T !== nothing && copyto!(ws.Tmat, T)
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

# Central differences with the step of the functional `_ekf_jacobian`. Uses `tmp_m2`
# for the shifted state and `tmp_p1`/`tmp_p2` for the two evaluations, which are free
# at this point of each filter step.
function _ekf_jacobian!(ws::EKFWorkspace{T}, ::FiniteDiffJacobian, a, t::Int) where {T}
    h_step = sqrt(eps(T))
    a_shift = ws.tmp_m2
    y_plus, y_minus = ws.tmp_p1, ws.tmp_p2
    copyto!(a_shift, a)
    for j in eachindex(a_shift)
        aj = a_shift[j]
        δ = max(h_step * abs(aj), h_step)
        a_shift[j] = aj + δ
        measurement!(y_plus, ws.measurement, a_shift, t)
        a_shift[j] = aj - δ
        measurement!(y_minus, ws.measurement, a_shift, t)
        a_shift[j] = aj
        for i in eachindex(y_plus)
            ws.tmp_Zt[i, j] = (y_plus[i] - y_minus[i]) / (2δ)
        end
    end
    return ws.tmp_Zt
end

# ============================================================================
# In-place EKF
# ============================================================================

"""
    kalman_filter!(ws::EKFWorkspace, y::AbstractMatrix) -> loglik

Run the Extended Kalman Filter in place, storing per-step quantities in `ws`.

Behaves as `kalman_filter!` (in-place linear filter) with `Z` replaced by a
per-step Jacobian `tmp_Zt` and the measurement linearisation stored in
`ws.Zt[:, :, t]`. Returns `-Inf` when an innovation covariance is not positive
definite; the workspace accessors then throw.
"""
function kalman_filter!(ws::EKFWorkspace{T}, y::AbstractMatrix) where {T}
    y = _nan_obs(y)
    p, m, n = ws.obs_dim, ws.state_dim, ws.n_times
    _check_obs(ws, "kalman_filter!", y)
    _require_no_partial_periods(y, "kalman_filter!")

    ws.loglik = zero(T)
    ws.n_obs_valid = 0
    ws.smoothed = false
    ws.crosscov = false

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

        cholF = cholesky!(Symmetric(ws.tmp_pp1, :L); check = false)
        if !issuccess(cholF)
            ws.status = t
            ws.loglik = T(-Inf)
            return ws.loglik
        end
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
    ws.status = 0
    return ws.loglik
end

# ============================================================================
# In-place EKF smoother (extended RTS)
# ============================================================================

"""
    kalman_smoother!(ws::EKFWorkspace; crosscov::Bool=false)

Run the extended RTS smoother in place using filter results stored in `ws`.
Must be called after `kalman_filter!(ws, y)`.

The recursion is identical to the linear smoother with the constant `Z`
replaced by the per-step linearisation `Z_t = ws.Zt[:, :, t]` produced during
the forward filter pass. This is the standard EKF smoother approximation:
smoothing is performed on the time-varying linear-Gaussian approximation
generated by the forward EKF.

# Arguments
- `ws::EKFWorkspace` — workspace populated by `kalman_filter!`.
- `crosscov` — also compute lag-1 cross-covariances `Cov[α_t, α_{t-1} | y_{1:n}]`
  used by EM. Default `false`.

# Stored
- `ws.αs` — smoothed state means.
- `ws.Vs` — smoothed state covariances.
- `ws.Pcross` — lag-1 cross-covariances (when `crosscov=true`).
"""
function kalman_smoother!(ws::EKFWorkspace{T}; crosscov::Bool = false) where {T}
    _require_output(ws, :kalman_smoother!)
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

        # Lag-1 cross-covariance (Durbin & Koopman 2012, Section 4.7):
        # P_{t, t-1 | n} = (I - P_t N_{t-1}) T P_{t-1|t-1}, with N_smooth = N_{t-1}.
        # No matrix is inverted, so a singular P_t is fine.
        if crosscov && t > 1
            Ptt_tm1 = view(ws.Ptt, :, :, (t - 1))
            mul!(ws.tmp_mm1, ws.Tmat, Ptt_tm1)
            mul!(ws.tmp_mm2, ws.N_smooth, ws.tmp_mm1)
            Pcross_t = view(ws.Pcross, :, :, (t - 1))
            copyto!(Pcross_t, ws.tmp_mm1)
            mul!(Pcross_t, P_t, ws.tmp_mm2, -one(T), one(T))
        end
    end

    ws.smoothed = true
    ws.crosscov = crosscov
    return nothing
end

# ============================================================================
# Accessors (mirror KalmanWorkspace where possible)
# ============================================================================

# The shared accessors are in accessors.jl.
predicted_observations(ws::EKFWorkspace) = _require_output(ws, :predicted_observations).yt
measurement_jacobians(ws::EKFWorkspace) = _require_output(ws, :measurement_jacobians).Zt
_observed_mask(ws::EKFWorkspace) = repeat(.!transpose(ws.missing_mask), ws.obs_dim)
observed_mask(ws::EKFWorkspace) = _observed_mask(_require_output(ws, :observed_mask))

function system_matrices(ws::EKFWorkspace)
    EKFParms(ws.measurement, ws.jacobian_mode, ws.H, ws.Tmat, ws.R, ws.Q)
end
function obs_matrix(::EKFWorkspace)
    throw(ArgumentError(
        "the extended Kalman filter has no observation matrix; " *
        "use measurement_jacobians for the linearisation at each period"))
end
