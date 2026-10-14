"""
    diffuse_smoother.jl

Exact diffuse smoothing (Durbin & Koopman 2012, Section 5.3) for the output of the exact
diffuse filters, shared by `kalman_smoother(::DiffuseFilterResult)` and
`kalman_smoother!(::DiffuseKalmanWorkspace)`.
"""

# Scratch storage for `_diffuse_smoother_backward!`, sized for `p` observables and
# `m` states.
struct _DiffuseSmootherScratch{T}
    Zs::Matrix{T}     # p × m: observed rows of Z, decorrelated
    Hs::Matrix{T}     # p × p: observed block of H
    C::Matrix{T}      # p × p: unit lower-triangular factor of the observed H
    w::Vector{T}      # p: decorrelated innovation
    h::Vector{T}      # p: decorrelated observation variances
    Minf::Matrix{T}   # m × p: Pinf zᵢ before row i
    Mstar::Matrix{T}  # m × p: Pstar zᵢ before row i
    Finf::Vector{T}   # p: zᵢ' Pinf zᵢ
    Fstar::Vector{T}  # p: zᵢ' Pstar zᵢ + hᵢ
    vrow::Vector{T}   # p: innovation of row i given rows 1, …, i - 1
    Pinf::Matrix{T}   # m × m
    Pstar::Matrix{T}  # m × m
    δ::Vector{T}      # m: a_{t,i} - a_t
    r0::Vector{T}     # m
    r1::Vector{T}     # m
    tmp_m::Vector{T}  # m
    N0::Matrix{T}     # m × m
    N1::Matrix{T}     # m × m
    N2::Matrix{T}     # m × m
    L0::Matrix{T}     # m × m
    L1::Matrix{T}     # m × m
    X::Matrix{T}      # m × m
    tmp1::Matrix{T}   # m × m
    tmp2::Matrix{T}   # m × m
end

function _DiffuseSmootherScratch{T}(p::Int, m::Int) where {T}
    mm() = zeros(T, m, m)
    return _DiffuseSmootherScratch{T}(
        zeros(T, p, m), zeros(T, p, p), zeros(T, p, p), zeros(T, p), zeros(T, p),
        zeros(T, m, p), zeros(T, m, p), zeros(T, p), zeros(T, p), zeros(T, p),
        mm(), mm(), zeros(T, m), zeros(T, m), zeros(T, m), zeros(T, m),
        mm(), mm(), mm(), mm(), mm(), mm(), mm(), mm())
end

# out += A' N B, through `tmp`.
function _add_sandwich!(out, A, N, B, tmp; α = true)
    mul!(tmp, N, B)
    mul!(out, transpose(A), tmp, α, true)
    return out
end

# The contents of `N` replaced by `T' N T`.
function _transition_cov!(N, Tm, tmp)
    mul!(tmp, N, Tm)
    mul!(N, transpose(Tm), tmp)
    return N
end

"""
    _diffuse_smoother_backward!(αs, Vs, sc, Z, H, Tm, at, Pt, Pinf_store, vt,
        observed_mask, d, r, N, tol, covariances)

Smooth periods `1, …, d` of an exact diffuse filter run, writing `αs[:, t]` and,
when `covariances` is true, `Vs[:, :, t]`.

On entry `r` and `N` hold the smoothing quantities `r_d` and `N_d` of the ordinary
backward recursion run over periods `n, …, d + 1` (zero when `d == n`); `N` is not
read when `covariances` is false. `at`, `Pt` and `Pinf_store` hold the predicted
state and the finite and diffuse parts of its covariance, and `vt` the innovations
of the observed rows in its leading entries.

With the initial covariance `Pstar + κ Pinf`, the smoothing quantities are expanded
as `r = r⁽⁰⁾ + κ⁻¹ r⁽¹⁾ + …` and `N = N⁽⁰⁾ + κ⁻¹ N⁽¹⁾ + κ⁻² N⁽²⁾ + …`, and as κ → ∞

    α̂ₜ = aₜ + Pstarₜ r⁽⁰⁾ + Pinfₜ r⁽¹⁾,
    Vₜ = Pstarₜ - Pstarₜ N⁽⁰⁾ Pstarₜ - Pinfₜ N⁽¹⁾ Pstarₜ - Pstarₜ N⁽¹⁾ Pinfₜ
         - Pinfₜ N⁽²⁾ Pinfₜ

(Durbin & Koopman 2012, Section 5.3). The observed rows of each period are processed one
at a time after the decorrelation of [`_decorrelate_rows!`](@ref), which handles
every step type of the filter (`diffuse_flags` 1, 2, 0 and -1) with one recursion
(Koopman & Durbin 2000). The row quantities are rebuilt from the stored predicted
covariances by repeating the filter's row-by-row update for the period. For a row
with `Finf = z' Pinf z > tol` the recursions are the coefficients of `κ⁰`, `κ⁻¹`
and `κ⁻²` in

    r ← z v / F + L' r,   N ← z z' / F + L' N L,
    F = κ Finf + Fstar,   L = I - K z' = L⁽⁰⁾ + κ⁻¹ L⁽¹⁾ + κ⁻² L⁽²⁾ + …,

with `K⁽⁰⁾ = Pinf z / Finf`, `K⁽¹⁾ = Pstar z / Finf - K⁽⁰⁾ Fstar / Finf`,
`L⁽⁰⁾ = I - K⁽⁰⁾ z'`, `L⁽¹⁾ = -K⁽¹⁾ z'` and `L⁽²⁾ = -(Fstar / Finf) L⁽¹⁾`. A row
with `Finf ≤ tol` has `Pinf z = 0` and the ordinary recursion, with `F = Fstar`,
applied to each coefficient.
"""
function _diffuse_smoother_backward!(αs, Vs, sc::_DiffuseSmootherScratch, Z, H, Tm, at,
        Pt, Pinf_store, vt, observed_mask, d, r, N, tol, covariances::Bool)
    ET = eltype(sc.r0)
    m = size(Tm, 1)
    p = size(Z, 1)
    (; r0, r1, N0, N1, N2, L0, L1, X, tmp1, tmp2, tmp_m) = sc

    copyto!(r0, r)
    fill!(r1, zero(ET))
    if covariances
        copyto!(N0, N)
        fill!(N1, zero(ET))
        fill!(N2, zero(ET))
    end

    for t in d:-1:1
        # From the start of period t + 1 to the end of period t: r ← T' r, N ← T' N T.
        mul!(tmp_m, transpose(Tm), r0)
        copyto!(r0, tmp_m)
        mul!(tmp_m, transpose(Tm), r1)
        copyto!(r1, tmp_m)
        if covariances
            _transition_cov!(N0, Tm, tmp1)
            _transition_cov!(N1, Tm, tmp1)
            _transition_cov!(N2, Tm, tmp1)
        end

        Pstar_t = view(Pt, :, :, t)
        Pinf_t = view(Pinf_store, :, :, t)

        # The observed rows, decorrelated.
        pt = 0
        for i in 1:p
            observed_mask[i, t] || continue
            pt += 1
            for j in 1:m
                sc.Zs[pt, j] = Z[i, j]
            end
            sc.w[pt] = vt[pt, t]
            k = 0
            for l in 1:p
                observed_mask[l, t] || continue
                k += 1
                sc.Hs[pt, k] = H[i, l]
            end
        end
        Zs = view(sc.Zs, 1:pt, :)
        _decorrelate_rows!(Zs, view(sc.w, 1:pt), view(sc.h, 1:pt),
            view(sc.Hs, 1:pt, 1:pt), view(sc.C, 1:pt, 1:pt))

        # Forward through the rows: the filter's row-by-row update, keeping the
        # quantities the backward pass needs.
        copyto!(sc.Pinf, Pinf_t)
        copyto!(sc.Pstar, Pstar_t)
        fill!(sc.δ, zero(ET))
        for i in 1:pt
            z = view(Zs, i, :)
            Minf = view(sc.Minf, :, i)
            Mstar = view(sc.Mstar, :, i)
            mul!(Minf, sc.Pinf, z)
            mul!(Mstar, sc.Pstar, z)
            Finf = dot(z, Minf)
            Fstar = dot(z, Mstar) + sc.h[i]
            v = sc.w[i] - dot(z, sc.δ)
            sc.Finf[i] = Finf
            sc.Fstar[i] = Fstar
            sc.vrow[i] = v
            if Finf > tol
                c = Fstar / Finf^2
                for k in 1:m, l in 1:m

                    sc.Pstar[l, k] += c * Minf[l] * Minf[k] -
                                      (Mstar[l] * Minf[k] + Minf[l] * Mstar[k]) / Finf
                    sc.Pinf[l, k] -= Minf[l] * Minf[k] / Finf
                end
                sc.δ .+= Minf .* (v / Finf)
            elseif Fstar > tol
                for k in 1:m, l in 1:m

                    sc.Pstar[l, k] -= Mstar[l] * Mstar[k] / Fstar
                end
                sc.δ .+= Mstar .* (v / Fstar)
            end
        end

        # Backward through the rows.
        for i in pt:-1:1
            z = view(Zs, i, :)
            Minf = view(sc.Minf, :, i)
            Mstar = view(sc.Mstar, :, i)
            Finf = sc.Finf[i]
            Fstar = sc.Fstar[i]
            v = sc.vrow[i]
            if Finf > tol
                # K⁽⁰⁾ = Minf / Finf, K⁽¹⁾ = (Mstar - K⁽⁰⁾ Fstar) / Finf
                # r⁽¹⁾ ← z v / Finf + L⁽⁰⁾' r⁽¹⁾ + L⁽¹⁾' r⁽⁰⁾,  r⁽⁰⁾ ← L⁽⁰⁾' r⁽⁰⁾
                k0r0 = dot(Minf, r0) / Finf
                k0r1 = dot(Minf, r1) / Finf
                k1r0 = (dot(Mstar, r0) - Fstar * dot(Minf, r0) / Finf) / Finf
                s1 = v / Finf - k0r1 - k1r0
                for l in 1:m
                    r1[l] += z[l] * s1
                    r0[l] -= z[l] * k0r0
                end
                covariances || continue

                for k in 1:m, l in 1:m

                    K0 = Minf[l] / Finf
                    K1 = (Mstar[l] - Minf[l] * Fstar / Finf) / Finf
                    L0[l, k] = (l == k) - K0 * z[k]
                    L1[l, k] = -K1 * z[k]
                end
                # X = L⁽¹⁾' N⁽⁰⁾ L⁽⁰⁾ + L⁽⁰⁾' N⁽⁰⁾ L⁽¹⁾
                fill!(X, zero(ET))
                _add_sandwich!(X, L1, N0, L0, tmp1)
                _add_sandwich!(X, L0, N0, L1, tmp1)
                # N⁽²⁾ ← -z z' Fstar / Finf² + L⁽⁰⁾' N⁽²⁾ L⁽⁰⁾ + L⁽⁰⁾' N⁽¹⁾ L⁽¹⁾
                #        + L⁽¹⁾' N⁽¹⁾ L⁽⁰⁾ + L⁽¹⁾' N⁽⁰⁾ L⁽¹⁾ - (Fstar / Finf) X
                # where the last term is L⁽²⁾' N⁽⁰⁾ L⁽⁰⁾ + L⁽⁰⁾' N⁽⁰⁾ L⁽²⁾.
                fill!(tmp2, zero(ET))
                _add_sandwich!(tmp2, L0, N2, L0, tmp1)
                _add_sandwich!(tmp2, L0, N1, L1, tmp1)
                _add_sandwich!(tmp2, L1, N1, L0, tmp1)
                _add_sandwich!(tmp2, L1, N0, L1, tmp1)
                for k in 1:m, l in 1:m

                    N2[l, k] = tmp2[l, k] - z[l] * z[k] * Fstar / Finf^2 -
                               X[l, k] * Fstar / Finf
                end
                # N⁽¹⁾ ← z z' / Finf + L⁽⁰⁾' N⁽¹⁾ L⁽⁰⁾ + X
                fill!(tmp2, zero(ET))
                _add_sandwich!(tmp2, L0, N1, L0, tmp1)
                for k in 1:m, l in 1:m

                    N1[l, k] = tmp2[l, k] + z[l] * z[k] / Finf + X[l, k]
                end
                # N⁽⁰⁾ ← L⁽⁰⁾' N⁽⁰⁾ L⁽⁰⁾
                fill!(tmp2, zero(ET))
                _add_sandwich!(tmp2, L0, N0, L0, tmp1)
                copyto!(N0, tmp2)
            elseif Fstar > tol
                # K = Mstar / Fstar, L = I - K z'
                # r⁽⁰⁾ ← z v / Fstar + L' r⁽⁰⁾,  r⁽¹⁾ ← L' r⁽¹⁾
                kr0 = dot(Mstar, r0) / Fstar
                kr1 = dot(Mstar, r1) / Fstar
                for l in 1:m
                    r0[l] += z[l] * (v / Fstar - kr0)
                    r1[l] -= z[l] * kr1
                end
                covariances || continue

                for k in 1:m, l in 1:m

                    L0[l, k] = (l == k) - Mstar[l] * z[k] / Fstar
                end
                # N⁽⁰⁾ ← z z' / Fstar + L' N⁽⁰⁾ L,  N⁽ʲ⁾ ← L' N⁽ʲ⁾ L for j = 1, 2
                for Nj in (N0, N1, N2)
                    fill!(tmp2, zero(ET))
                    _add_sandwich!(tmp2, L0, Nj, L0, tmp1)
                    copyto!(Nj, tmp2)
                end
                for k in 1:m, l in 1:m

                    N0[l, k] += z[l] * z[k] / Fstar
                end
            end
        end

        # α̂ₜ = aₜ + Pstarₜ r⁽⁰⁾ + Pinfₜ r⁽¹⁾
        α_t = view(αs, :, t)
        copyto!(α_t, view(at, :, t))
        mul!(α_t, Pstar_t, r0, true, true)
        mul!(α_t, Pinf_t, r1, true, true)

        covariances || continue
        # Vₜ = Pstar - Pstar N⁽⁰⁾ Pstar - Pinf N⁽¹⁾ Pstar - Pstar N⁽¹⁾ Pinf - Pinf N⁽²⁾ Pinf
        V_t = view(Vs, :, :, t)
        copyto!(V_t, Pstar_t)
        _add_sandwich!(V_t, Pstar_t, N0, Pstar_t, tmp1; α = -1)
        _add_sandwich!(V_t, Pinf_t, N1, Pstar_t, tmp1; α = -1)
        _add_sandwich!(V_t, Pstar_t, N1, Pinf_t, tmp1; α = -1)
        _add_sandwich!(V_t, Pinf_t, N2, Pinf_t, tmp1; α = -1)
    end
    return nothing
end

function _require_diffuse_ended(ended::Bool, f::Symbol)
    ended && return nothing
    throw(ArgumentError(
        "$f: the diffuse period does not end within the sample, so some smoothed " *
        "states have infinite variance"))
end

function _no_diffuse_crosscov(f::Symbol)
    throw(ArgumentError(
        "$f: cross-lag covariances are not available under exact diffuse " *
        "initialization. Use crosscov = false, or approximate diffuse " *
        "initialization (a large P1)."))
end

"""
    kalman_smoother(result::DiffuseFilterResult) -> SmootherResult

Run the exact diffuse smoother (Durbin & Koopman 2012, Section 5.3) on the output of the
exact diffuse filter `kalman_filter(p, y, a1, P1_star, P1_inf)`. The result is the
limit, as κ → ∞, of the ordinary smoother with initial covariance
`P1_star + κ P1_inf`.

The diffuse-period observations are processed one row at a time (Koopman & Durbin
2000), so every step type of the filter is handled, including periods where `Finf`
is singular and partially observed periods.

Throws an `ArgumentError` if the filter failed, if the diffuse period does not end
within the sample (some smoothed variances are then infinite), or if
`crosscov = true`: cross-lag covariances are not available under exact diffuse
initialization.

# Example
```julia
filt = kalman_filter(p, y, a1, P1_star, P1_inf)
sm = kalman_smoother(filt)
smoothed_states(sm)
```
"""
function kalman_smoother(result::DiffuseFilterResult; crosscov::Bool = false)
    _require_output(result, :kalman_smoother)
    crosscov && _no_diffuse_crosscov(:kalman_smoother)
    _require_diffuse_ended(result.diffuse_ended, :kalman_smoother)

    Z = _as_matrix(result.p.Z)
    H = _as_matrix(result.p.H)
    Tm = _as_matrix(result.p.T)
    at, Pt, vt, Ft = result.at, result.Pt, result.vt, result.Ft
    m, n = size(at)
    p = size(vt, 1)
    d = result.d
    ET = promote_type(eltype(Z), eltype(H), eltype(Tm), eltype(at), eltype(Pt))

    αs = Matrix{ET}(undef, m, n)
    Vs = Array{ET}(undef, m, m, n)

    # Ordinary recursion after the diffuse period.
    r = zeros(ET, m)
    N = zeros(ET, m, m)
    for t in n:-1:(d + 1)
        (α_t, V_t, r, N) = _rts_step(Z, Tm, at, Pt, vt, Ft, t, r, N;
            missing_mask = result.missing_mask, observed_mask = result.observed_mask)
        αs[:, t] = α_t
        Vs[:, :, t] = V_t
    end

    sc = _DiffuseSmootherScratch{ET}(p, m)
    _diffuse_smoother_backward!(αs, Vs, sc, Z, H, Tm, at, Pt, result.Pinf_store, vt,
        result.observed_mask, d, r, N, result.tol, true)
    return SmootherResult(αs, Vs, nothing)
end
