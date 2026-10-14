"""
    disturbance_smoother.jl

Disturbance smoother: the smoothed observation and state disturbances and their
conditional variances, Durbin & Koopman (2012), Section 4.5.
"""

"""
    DisturbanceSmootherResult{T<:Real}

Output of [`disturbance_smoother`](@ref): the smoothed disturbances of

    yₜ = Z αₜ + εₜ,       εₜ ~ N(0, H)
    αₜ₊₁ = T αₜ + R ηₜ,   ηₜ ~ N(0, Q)

with their conditional covariances and the auxiliary residuals.

# Fields
- `εs`: ε̂ₜ = E[εₜ | y₁:ₙ] (p × n)
- `Vε`: Var[εₜ | y₁:ₙ] (p × p × n)
- `ηs`: η̂ₜ = E[ηₜ | y₁:ₙ] (r × n)
- `Vη`: Var[ηₜ | y₁:ₙ] (r × r × n)
- `εs_std`, `ηs_std`: the auxiliary residuals, each element of ε̂ₜ and η̂ₜ divided by
  its standard deviation as an estimator, √diag(Var[ε̂ₜ]) and √diag(Var[η̂ₜ]), where
  Var[ε̂ₜ] = H − Var[εₜ | y₁:ₙ] and Var[η̂ₜ] = Q − Var[ηₜ | y₁:ₙ]. They are NaN where
  that variance is zero.

ηₜ moves the state from period t to t + 1, so the data carry no information on ηₙ:
η̂ₙ = 0 and Var[ηₙ | y₁:ₙ] = Q. In a period with no observation ε̂ₜ = 0 and
Var[εₜ | y₁:ₙ] = H. In a partly observed period, the disturbances of the missing series
are smoothed through their covariance with the observed ones.

# Accessors
`smoothed_obs_disturbances`, `smoothed_obs_disturbances_cov`,
`smoothed_state_disturbances`, `smoothed_state_disturbances_cov` and
`auxiliary_residuals`.

# Construction
- `disturbance_smoother(result)`: from the output of `kalman_filter`
- `disturbance_smoother(p, y, a1, P1)`: filter and smooth in one call
- `disturbance_smoother(ws)` or `disturbance_smoother!(out, ws)`: from a
  `KalmanWorkspace` after `kalman_filter!(ws, y)`
- `disturbance_smoother(model)`: from a filtered or fitted `StateSpaceModel`
- `DisturbanceSmootherResult{T}(p, r, n)`: zero-filled storage for
  `disturbance_smoother!`
"""
struct DisturbanceSmootherResult{T <: Real}
    εs::Matrix{T}
    Vε::Array{T, 3}
    ηs::Matrix{T}
    Vη::Array{T, 3}
    εs_std::Matrix{T}
    ηs_std::Matrix{T}
end

function DisturbanceSmootherResult{T}(p::Integer, r::Integer, n::Integer) where {T <: Real}
    return DisturbanceSmootherResult{T}(zeros(T, p, n), zeros(T, p, p, n),
        zeros(T, r, n), zeros(T, r, r, n), zeros(T, p, n), zeros(T, r, n))
end

"""
    smoothed_obs_disturbances(r::DisturbanceSmootherResult) -> Matrix

Return the smoothed observation disturbances ε̂ₜ = E[εₜ | y₁:ₙ] (p × n).
"""
smoothed_obs_disturbances(r::DisturbanceSmootherResult) = r.εs

"""
    smoothed_obs_disturbances_cov(r::DisturbanceSmootherResult) -> Array{T,3}

Return the conditional covariances Var[εₜ | y₁:ₙ] (p × p × n).
"""
smoothed_obs_disturbances_cov(r::DisturbanceSmootherResult) = r.Vε

"""
    smoothed_state_disturbances(r::DisturbanceSmootherResult) -> Matrix

Return the smoothed state disturbances η̂ₜ = E[ηₜ | y₁:ₙ] (r × n).
"""
smoothed_state_disturbances(r::DisturbanceSmootherResult) = r.ηs

"""
    smoothed_state_disturbances_cov(r::DisturbanceSmootherResult) -> Array{T,3}

Return the conditional covariances Var[ηₜ | y₁:ₙ] (r × r × n).
"""
smoothed_state_disturbances_cov(r::DisturbanceSmootherResult) = r.Vη

"""
    auxiliary_residuals(r::DisturbanceSmootherResult) -> (obs = Matrix, state = Matrix)

Return the standardized smoothed disturbances (Durbin & Koopman 2012, Section 2.12): `obs`
(p × n) divides each element of ε̂ₜ by its standard deviation √Var[ε̂ₜ], and `state`
(r × n) does the same for η̂ₜ. Values well beyond ±2 in `obs` point to outliers; in
`state`, to level shifts or other structural breaks. Elements whose standard deviation
is zero, such as missing series and the last period of `state`, are NaN.
"""
auxiliary_residuals(r::DisturbanceSmootherResult) = (obs = r.εs_std, state = r.ηs_std)

# x / √v, or NaN when the estimator has no variance.
_standardize(x, v) = v > 0 ? x / sqrt(v) : oftype(x / one(v), NaN)

# The backward recursion of Durbin & Koopman (2012), Section 4.5.3, on the arrays of a filter
# output. With rₜ, Nₜ the smoothing quantities after period t, Kₜ = T Pₜ Zₜ' Fₜ⁻¹ and
# Wₜ the rows observed in period t:
#     uₜ = Fₜ⁻¹ vₜ − Kₜ' rₜ,       Dₜ = Fₜ⁻¹ + Kₜ' Nₜ Kₜ,
#     ε̂ₜ = H Wₜ' uₜ,              Var[εₜ | y] = H − H Wₜ' Dₜ Wₜ H,
#     η̂ₜ = Q R' rₜ,               Var[ηₜ | y] = Q − Q R' Nₜ R Q,
#     rₜ₋₁ = Zₜ' uₜ + T' rₜ,       Nₜ₋₁ = Zₜ' Fₜ⁻¹ Zₜ + Lₜ' Nₜ Lₜ,  Lₜ = T − Kₜ Zₜ.
function _disturbance_smoother(Z::AbstractMatrix, H::AbstractMatrix, T::AbstractMatrix,
        R::AbstractMatrix, Q::AbstractMatrix, Pt::AbstractArray, vt::AbstractMatrix,
        Ft::AbstractArray; missing_mask::BitVector, observed_mask)
    p, m = size(Z)
    r = size(R, 2)
    n = size(vt, 2)
    ET = promote_type(eltype(Z), eltype(H), eltype(T), eltype(R), eltype(Q),
        eltype(Pt), eltype(vt), eltype(Ft))

    out = DisturbanceSmootherResult{ET}(p, r, n)
    QRt = Q * R'
    r_vec = zeros(ET, m)
    N_mat = zeros(ET, m, m)

    for t in n:-1:1
        # State disturbances from r_t and N_t.
        var_η̂ = QRt * N_mat * QRt'
        out.ηs[:, t] = QRt * r_vec
        out.Vη[:, :, t] = Q - var_η̂
        for i in 1:r
            out.ηs_std[i, t] = _standardize(out.ηs[i, t], var_η̂[i, i])
        end

        if missing_mask[t]
            out.Vε[:, :, t] = H
            out.εs_std[:, t] .= NaN
            r_vec = T' * r_vec
            N_mat = T' * N_mat * T
            continue
        end

        rows = _period_obs_rows(observed_mask, vt, t)
        pt = length(rows)
        Zr = _select_rows(Z, rows)
        HW = H[:, rows]
        F_inv = inv(view(Ft, 1:pt, 1:pt, t))
        K_t = T * view(Pt, :, :, t) * Zr' * F_inv
        L_t = T - K_t * Zr

        u_t = F_inv * view(vt, 1:pt, t) - K_t' * r_vec
        D_t = F_inv + K_t' * N_mat * K_t
        var_ε̂ = HW * D_t * HW'
        out.εs[:, t] = HW * u_t
        out.Vε[:, :, t] = H - var_ε̂
        for i in 1:p
            out.εs_std[i, t] = _standardize(out.εs[i, t], var_ε̂[i, i])
        end

        r_vec = Zr' * u_t + T' * r_vec
        N_mat = Zr' * F_inv * Zr + L_t' * N_mat * L_t
    end
    return out
end

"""
    disturbance_smoother(result::KalmanFilterResult) -> DisturbanceSmootherResult

Smooth the observation and state disturbances εₜ and ηₜ from the output of
[`kalman_filter`](@ref). AD-compatible. See [`DisturbanceSmootherResult`](@ref) for
the output.

The smoothed disturbances are the basis of outlier and structural-break diagnostics
through [`auxiliary_residuals`](@ref).

Throws an `ArgumentError` if the filter stopped on an innovation covariance that was not
positive definite.

# Example
```julia
filt = kalman_filter(p, y, a1, P1)
ds = disturbance_smoother(filt)
smoothed_obs_disturbances(ds)
auxiliary_residuals(ds).state      # look for |value| > 2
```
"""
function disturbance_smoother(result::KalmanFilterResult)
    _require_output(result, :disturbance_smoother)
    p = result.p
    return _disturbance_smoother(_as_matrix(p.Z), _as_matrix(p.H), _as_matrix(p.T),
        _as_matrix(p.R), _as_matrix(p.Q), result.Pt, result.vt, result.Ft;
        missing_mask = result.missing_mask, observed_mask = result.observed_mask)
end

"""
    disturbance_smoother(p::KFParms, y, a1, P1) -> DisturbanceSmootherResult

Run [`kalman_filter`](@ref) and then the disturbance smoother on its output.
"""
function disturbance_smoother(p::KFParms, y, a1, P1)
    disturbance_smoother(kalman_filter(p, y, a1, P1))
end

function disturbance_smoother(::DiffuseFilterResult)
    throw(ArgumentError(
        "disturbance_smoother: exact diffuse filter output is not supported. Smooth " *
        "the states with kalman_smoother, or use an approximate diffuse prior " *
        "(large P1) with kalman_filter."))
end

"""
    disturbance_smoother!(out::DisturbanceSmootherResult, ws::KalmanWorkspace) -> out

Fill `out` with the smoothed disturbances from the filter output in `ws`. This needs
`kalman_filter!(ws, y)` first, but not `kalman_smoother!`. It changes no output of
`ws`. `out` must have been built as `DisturbanceSmootherResult{T}(p, r, n)` with the
dimensions of `ws`. Each call allocates three small buffers (r × m and p × p) and
nothing per period.

    disturbance_smoother(ws::KalmanWorkspace) -> DisturbanceSmootherResult

The same, into newly allocated storage.

Throws an `ArgumentError` if the filter has not run or failed.
"""
function disturbance_smoother!(out::DisturbanceSmootherResult, ws::KalmanWorkspace{T}) where {T}
    _require_output(ws, :disturbance_smoother!)
    p, m, r, n = ws.obs_dim, ws.state_dim, ws.shock_dim, ws.n_times
    for (name, A, dims) in (("εs", out.εs, (p, n)), ("Vε", out.Vε, (p, p, n)),
        ("ηs", out.ηs, (r, n)), ("Vη", out.Vη, (r, r, n)),
        ("εs_std", out.εs_std, (p, n)), ("ηs_std", out.ηs_std, (r, n)))
        _check_size("disturbance_smoother!", "out.$name", A, dims)
    end

    H, Q = ws.H, ws.Q
    QRt = Q * transpose(ws.R)      # r × m
    QRt_N = similar(QRt)           # r × m
    HWD = similar(H)               # p × p, leading pt columns used
    r_vec, N_mat = ws.r_smooth, ws.N_smooth
    fill!(r_vec, zero(T))
    fill!(N_mat, zero(T))

    for t in n:-1:1
        # State disturbances: η̂ = Q R' r, Var = Q − Q R' N R Q.
        η_t = view(out.ηs, :, t)
        Vη_t = view(out.Vη, :, :, t)
        mul!(η_t, QRt, r_vec)
        mul!(QRt_N, QRt, N_mat)
        mul!(Vη_t, QRt_N, transpose(QRt))
        for i in 1:r
            out.ηs_std[i, t] = _standardize(η_t[i], Vη_t[i, i])
        end
        Vη_t .= Q .- Vη_t

        if ws.missing_mask[t]
            out.εs[:, t] .= zero(T)
            copyto!(view(out.Vε, :, :, t), H)
            out.εs_std[:, t] .= T(NaN)
            # r ← T' r, N ← T' N T
            mul!(ws.tmp_m1, transpose(ws.Tmat), r_vec)
            copyto!(r_vec, ws.tmp_m1)
            mul!(ws.tmp_mm1, transpose(ws.Tmat), N_mat)
            mul!(N_mat, ws.tmp_mm1, ws.Tmat)
            continue
        end

        pt = ws.n_observed[t]
        rows = view(ws.obs_rows, 1:pt, t)
        K_t = view(ws.Kt, :, 1:pt, t)

        # u = F⁻¹ v − K' r
        u_t = view(ws.tmp_p1, 1:pt)
        copyto!(u_t, view(ws.vt, 1:pt, t))
        _apply_Finv!(u_t, ws, t, pt, ws.tmp_p2)
        mul!(u_t, transpose(K_t), r_vec, -one(T), one(T))

        # D = F⁻¹ + K' N K
        D_t = view(ws.tmp_pp1, 1:pt, 1:pt)
        fill!(D_t, zero(T))
        for k in 1:pt
            D_t[k, k] = one(T)
        end
        _apply_Finv!(D_t, ws, t, pt, ws.tmp_pp2)
        KtN = view(ws.tmp_pm, 1:pt, :)
        mul!(KtN, transpose(K_t), N_mat)
        mul!(D_t, KtN, K_t, one(T), one(T))

        # ε̂ = H W' u and Var = H − (H W') D (W H), with H W' gathered into tmp_pp2.
        HW = view(ws.tmp_pp2, :, 1:pt)
        for (k, i) in pairs(rows), j in 1:p

            HW[j, k] = H[j, i]
        end
        mul!(view(out.εs, :, t), HW, u_t)
        HWD_t = view(HWD, :, 1:pt)
        mul!(HWD_t, HW, D_t)
        Vε_t = view(out.Vε, :, :, t)
        mul!(Vε_t, HWD_t, transpose(HW))
        for i in 1:p
            out.εs_std[i, t] = _standardize(out.εs[i, t], Vε_t[i, i])
        end
        Vε_t .= H .- Vε_t

        # N ← Z' F⁻¹ Z + L' N L with L = T − K Z. Z' F⁻¹ Z goes into tmp_mm3 first,
        # because F⁻¹ Z overwrites the gathered rows.
        Zr = view(ws.tmp_pm2, 1:pt, :)
        for j in 1:m, (k, i) in pairs(rows)

            Zr[k, j] = ws.Z[i, j]
        end
        # r ← Z' u + T' r, before N because both read the old r and N.
        mul!(ws.tmp_m1, transpose(ws.Tmat), r_vec)
        mul!(ws.tmp_m1, transpose(Zr), u_t, one(T), one(T))
        copyto!(r_vec, ws.tmp_m1)

        L = ws.L_smooth
        mul!(L, K_t, Zr)
        L .= ws.Tmat .- L
        mul!(ws.tmp_mm1, transpose(L), N_mat)
        mul!(ws.tmp_mm2, ws.tmp_mm1, L)
        FinvZ = view(ws.tmp_pm3, 1:pt, :)
        copyto!(FinvZ, Zr)
        _apply_Finv!(FinvZ, ws, t, pt, ws.tmp_pm)
        mul!(N_mat, transpose(Zr), FinvZ)
        N_mat .+= ws.tmp_mm2
    end
    return out
end

function disturbance_smoother(ws::KalmanWorkspace{T}) where {T}
    out = DisturbanceSmootherResult{T}(ws.obs_dim, ws.shock_dim, ws.n_times)
    return disturbance_smoother!(out, ws)
end

function _reject_diffuse_workspace(f::Symbol)
    throw(ArgumentError(
        "$f: an exact diffuse workspace is not supported. Use a KalmanWorkspace " *
        "with an approximate diffuse prior (large P1)."))
end
function disturbance_smoother!(::DisturbanceSmootherResult, ::DiffuseKalmanWorkspace)
    _reject_diffuse_workspace(:disturbance_smoother!)
end
function disturbance_smoother(::DiffuseKalmanWorkspace)
    _reject_diffuse_workspace(:disturbance_smoother)
end

"""
    disturbance_smoother(model::StateSpaceModel) -> DisturbanceSmootherResult

Smooth the disturbances of `model` at its current parameters, from the filter output it
stores. Needs `fit!` or `kalman_filter!(model, y)` first. EKF models and models with
`diffuse = :exact` are not supported.
"""
function disturbance_smoother(model::StateSpaceModel)
    _require_filter(model)
    model.spec isa EKFSpec && throw(ArgumentError(
        "disturbance_smoother: EKF models are not supported."))
    uses_exact_diffuse(model.spec) && throw(ArgumentError(
        "disturbance_smoother: models with diffuse = :exact are not supported."))
    p = build_kfparms(model.spec, _ssm_get_theta_namedtuple(model))
    return _disturbance_smoother(_as_matrix(p.Z), _as_matrix(p.H), _as_matrix(p.T),
        _as_matrix(p.R), _as_matrix(p.Q), model.Pt, model.vt, model.Ft;
        missing_mask = model.missing_mask, observed_mask = model.observed_mask)
end
