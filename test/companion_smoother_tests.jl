# Structured fast paths: a declared block-companion transition and a
# smoothed-mean-only backward pass. Both are reorganizations of the general
# recursions, so the general path is the reference and agreement is exact.

using Test
using Siphon
using LinearAlgebra
using Random
using Statistics

"""
    _companion_setup(; nser, nlag, n, seed)

A stable companion-form VAR written as a state-space model: state
`αₜ = (xₜ', xₜ₋₁', …)'` of dimension `nser·nlag`, `R = (I; 0)` so `R Q R'` has
rank `nser`, and `Z = (I 0)` observing the current period only. `H = 0` makes the
measurement noiseless, as it is for a latent-variable VAR.

Returns the primitives together with one simulated dataset whose missing pattern
mixes a quarterly series, a ragged edge, and one period with nothing observed.
"""
function _companion_setup(; nser::Int = 3, nlag::Int = 4, n::Int = 24,
        seed::Int = 20260918)
    rng = Xoshiro(seed)
    m = nser * nlag

    Tm = zeros(m, m)
    for l in 1:nlag
        Tm[1:nser, ((l - 1) * nser + 1):(l * nser)] = 0.4^l .* randn(rng, nser, nser) ./
                                                      sqrt(nser)
    end
    if nlag > 1
        Tm[(nser + 1):end, 1:(nser * (nlag - 1))] = I(nser * (nlag - 1))
    end

    R = zeros(m, nser)
    R[1:nser, :] = I(nser)
    Xq = randn(rng, nser, nser)
    Q = Matrix(Symmetric(Xq * Xq' / nser + 0.3I))
    Z = zeros(nser, m)
    Z[:, 1:nser] = I(nser)
    H = zeros(nser, nser)
    a1 = zeros(m)
    P1 = Matrix(0.5I, m, m)

    Lq = cholesky(Q).L
    y = zeros(nser, n)
    a = a1 + cholesky(Symmetric(P1)).L * randn(rng, m)
    for t in 1:n
        y[:, t] = Z * a
        a = Tm * a
        a[1:nser] += Lq * randn(rng, nser)
    end

    for t in 1:n                      # series 1 observed only every third period
        t % 3 == 0 || (y[1, t] = NaN)
    end
    y[:, n] .= NaN                    # a period with nothing observed
    y[nser, n - 1] = NaN              # a ragged edge one period deep

    return (; Z, H, Tm, R, Q, a1, P1, y, nser, nlag, m, n)
end

"""
    _companion_workspace(s; block)

A workspace carrying `s`'s parameters, with the companion structure declared when
`block` is nonzero.
"""
function _companion_workspace(s; block::Int = 0)
    ws = KalmanWorkspace(s.nser, s.m, s.nser, s.n)
    set_params!(ws, s.Z, s.H, s.Tm, s.R, s.Q)
    set_initial!(ws, s.a1, s.P1)
    block == 0 || set_companion_structure!(ws, block)
    return ws
end

@testset "companion structure — declaration is verified against Tmat" begin
    s = _companion_setup()
    ws = _companion_workspace(s)
    @test companion_block(ws) == 0

    set_companion_structure!(ws, s.nser)
    @test companion_block(ws) == s.nser

    set_companion_structure!(ws, 0)
    @test companion_block(ws) == 0
end

@testset "companion structure — a non-companion transition is rejected" begin
    s = _companion_setup()
    ws = _companion_workspace(s)

    # A dense transition of the same size: the C-BVAR case, where the lags are
    # quarters apart and there is no shift block at all.
    dense = randn(Xoshiro(5), s.m, s.m)
    dense ./= 2 * maximum(abs, eigvals(dense))
    update_params!(ws; Tmat = dense)
    @test_throws "not block-companion" set_companion_structure!(ws, s.nser)
    @test companion_block(ws) == 0
end

@testset "companion structure — block width must divide the state dimension" begin
    s = _companion_setup()
    ws = _companion_workspace(s)
    @test_throws "not a multiple" set_companion_structure!(ws, 5)
    @test_throws ArgumentError set_companion_structure!(ws, -1)
end

@testset "companion structure — a stale declaration is caught at filter time" begin
    # A declaration made before a parameter update must not survive a transition
    # that no longer has the shift structure: filtering with the wrong recursion
    # would give a plausible, silently wrong answer.
    s = _companion_setup()
    ws = _companion_workspace(s; block = s.nser)

    dense = randn(Xoshiro(6), s.m, s.m)
    dense ./= 2 * maximum(abs, eigvals(dense))
    update_params!(ws; Tmat = dense)
    @test_throws "declares a companion transition" kalman_filter!(ws, s.y)
end

@testset "companion structure — redrawing the leading block keeps the declaration" begin
    # A Gibbs sampler replaces only the VAR coefficients, which live in the
    # leading block row; the shift below is part of the companion form itself.
    s = _companion_setup()
    ws = _companion_workspace(s; block = s.nser)

    redrawn = copy(s.Tm)
    redrawn[1:s.nser, :] = 0.1 .* randn(Xoshiro(7), s.nser, s.m)
    update_params!(ws; Tmat = redrawn)
    @test kalman_filter!(ws, s.y) isa Real
    @test companion_block(ws) == s.nser
end

@testset "companion structure — filter results match the general path exactly" begin
    s = _companion_setup()
    wg = _companion_workspace(s)
    wc = _companion_workspace(s; block = s.nser)

    llg = kalman_filter!(wg, s.y)
    llc = kalman_filter!(wc, s.y)

    # The structured form skips multiplications by the exact zeros and ones of
    # the shift block; what remains is the same arithmetic in the same order.
    @test llc == llg
    @test wc.at == wg.at
    @test wc.Pt == wg.Pt
    @test wc.att == wg.att
    @test wc.Ptt == wg.Ptt
end

@testset "companion structure — smoother results match the general path exactly" begin
    s = _companion_setup()
    wg = _companion_workspace(s)
    wc = _companion_workspace(s; block = s.nser)

    filter_and_smooth!(wg, s.y)
    filter_and_smooth!(wc, s.y)

    @test wc.αs == wg.αs
    @test wc.Vs == wg.Vs
    @test wc.Pcross == wg.Pcross
end

@testset "companion structure — a single-lag model is the general path" begin
    # With nlag == 1 there is no shift block, so the declaration is vacuous and
    # the structured branch must still reproduce the general one.
    s = _companion_setup(; nser = 3, nlag = 1, n = 16)
    wg = _companion_workspace(s)
    wc = _companion_workspace(s; block = s.nser)

    @test kalman_filter!(wc, s.y) == kalman_filter!(wg, s.y)
    @test wc.Pt == wg.Pt
end

@testset "mean-only smoother — reproduces the smoothed means" begin
    # Agreement here is to rounding, not bitwise: evaluating `Lₜ' rₜ` as
    # `T' rₜ - Zₜ'(Kₜ' rₜ)` associates the products differently from forming `Lₜ`
    # first, which costs about one ulp per period and accumulates over the
    # backward pass. A sampler driven by these means is chaotic in them, so its
    # trajectory changes although its target distribution does not.
    s = _companion_setup()
    wg = _companion_workspace(s)
    wm = _companion_workspace(s)

    filter_and_smooth!(wg, s.y; crosscov = false)
    filter_and_smooth!(wm, s.y; crosscov = false, covariances = false)

    scale = max(1, maximum(abs, wg.αs))
    @test maximum(abs, wm.αs - wg.αs) < 1e-11 * scale
end

@testset "mean-only smoother — also on the companion path" begin
    s = _companion_setup()
    wg = _companion_workspace(s; block = s.nser)
    wm = _companion_workspace(s; block = s.nser)

    filter_and_smooth!(wg, s.y; crosscov = false)
    filter_and_smooth!(wm, s.y; crosscov = false, covariances = false)

    scale = max(1, maximum(abs, wg.αs))
    @test maximum(abs, wm.αs - wg.αs) < 1e-11 * scale
end

@testset "mean-only smoother — matches a reference built from the primitives" begin
    # The two implementations share the filter, so agreeing with each other is
    # weaker evidence than agreeing with the posterior mean computed from the
    # model primitives alone. On a model this small the whole path can be
    # conditioned in one dense Gaussian update.
    s = _companion_setup(; nser = 2, nlag = 2, n = 12)
    m, n, p = s.m, s.n, s.nser

    Tpow = [Matrix{Float64}(I, m, m)]
    for _ in 1:n
        push!(Tpow, s.Tm * Tpow[end])
    end
    mu = vcat((Tpow[t] * s.a1 for t in 1:n)...)
    Sig = zeros(m * n, m * n)
    RQR = s.R * s.Q * s.R'
    for t in 1:n, u in 1:n

        blk = Tpow[t] * s.P1 * Tpow[u]'
        for k in 1:(min(t, u) - 1)
            blk += Tpow[t - k] * RQR * Tpow[u - k]'
        end
        Sig[((t - 1) * m + 1):(t * m), ((u - 1) * m + 1):(u * m)] = blk
    end
    Sig = Matrix(Symmetric((Sig + Sig') / 2))

    obs = [(t, i) for t in 1:n for i in 1:p if !isnan(s.y[i, t])]
    q = length(obs)
    Zb = zeros(q, m * n)
    yv = zeros(q)
    for (k, (t, i)) in pairs(obs)
        Zb[k, ((t - 1) * m + 1):(t * m)] = s.Z[i, :]
        yv[k] = s.y[i, t]
    end
    # H = 0, so the conditioning is exact and the update needs a pseudo-inverse.
    post = mu + Sig * Zb' * (pinv(Zb * Sig * Zb') * (yv - Zb * mu))

    ws = _companion_workspace(s; block = p)
    filter_and_smooth!(ws, s.y; crosscov = false, covariances = false)

    @test maximum(abs, vec(ws.αs) - post) < 1e-6 * max(1, maximum(abs, post))
end

@testset "mean-only smoother — crosscov requires the covariances" begin
    s = _companion_setup()
    ws = _companion_workspace(s)
    kalman_filter!(ws, s.y)
    @test_throws "requires covariances=true" kalman_smoother!(
        ws; crosscov = true, covariances = false)
end

@testset "mean-only smoother — a fully missing dataset gives the prior path" begin
    # With nothing observed the smoothed mean is the prior mean at every period,
    # which exercises the branch that never touches an observation.
    s = _companion_setup()
    ws = _companion_workspace(s)
    ynan = fill(NaN, s.nser, s.n)

    filter_and_smooth!(ws, ynan; crosscov = false, covariances = false)
    expected = copy(s.a1)
    for t in 1:s.n
        @test ws.αs[:, t] ≈ expected atol = 1e-12
        expected = s.Tm * expected
    end
end

@testset "simulation smoother — the structured path draws the same numbers" begin
    # Same seed, same censoring: the draw must be the one the general path would
    # have produced, not merely one with the same distribution.
    s = _companion_setup()
    wg = _companion_workspace(s)
    wc = _companion_workspace(s; block = s.nser)

    dg = simulation_smoother(wg, s.y; rng = Xoshiro(11), n_draws = 3)
    dc = simulation_smoother(wc, s.y; rng = Xoshiro(11), n_draws = 3)

    @test maximum(abs, dg - dc) < 1e-12 * max(1, maximum(abs, dg))
end

@testset "simulation smoother — draw covariance survives the fast paths" begin
    # A simulation smoother wrong only in its variance still reproduces the
    # smoothed mean, so the mean alone cannot certify these paths. The draw
    # covariance is compared across paths at a tolerance well inside the Monte
    # Carlo error of the number of draws used.
    s = _companion_setup(; nser = 2, nlag = 3, n = 10)
    wg = _companion_workspace(s)
    wc = _companion_workspace(s; block = s.nser)

    nd = 4000
    dg = simulation_smoother(wg, s.y; rng = Xoshiro(13), n_draws = nd)
    dc = simulation_smoother(wc, s.y; rng = Xoshiro(13), n_draws = nd)
    fg = reshape(dg, s.m * s.n, nd)
    fc = reshape(dc, s.m * s.n, nd)

    @test maximum(abs, cov(fg, dims = 2) - cov(fc, dims = 2)) < 1e-10
    @test maximum(abs, vec(mean(fg, dims = 2)) - vec(mean(fc, dims = 2))) < 1e-10
end
