# The rank-revealing measurement update. A state space driven by fewer shocks
# than states and observed without measurement error has a singular innovation
# covariance, which the Cholesky path cannot factor. Filtering on the range of
# `Fₜ` handles it, and an observation whose innovation leaves that range is
# rejected rather than projected onto it.

using Test
using Siphon
using LinearAlgebra
using Random

"""
    _fullrank_setup(; seed)

A model with a positive definite `Fₜ` in every period, partially observed. The
rank-revealing path must reproduce the Cholesky path on it exactly.
"""
function _fullrank_setup(; seed::Int = 20260920)
    rng = Xoshiro(seed)
    p, m, r, n = 3, 4, 4, 24

    Z = randn(rng, p, m)
    Xh = randn(rng, p, p)
    H = Matrix(Symmetric(Xh * Xh' / p + 0.5I))
    Tm = randn(rng, m, m) ./ (2 * sqrt(m))
    R = Matrix{Float64}(I, m, r)
    Xq = randn(rng, r, r)
    Q = Matrix(Symmetric(Xq * Xq' / r + 0.4I))
    a1 = zeros(m)
    P1 = Matrix(0.8I, m, m)

    Lq = cholesky(Q).L
    Lh = cholesky(H).L
    y = zeros(p, n)
    a = a1 + cholesky(Symmetric(P1)).L * randn(rng, m)
    for t in 1:n
        y[:, t] = Z * a + Lh * randn(rng, p)
        a = Tm * a + R * (Lq * randn(rng, r))
    end

    y[1, 5] = NaN                     # a single hole
    y[:, 11] .= NaN                   # a period with nothing observed
    y[2, n] = NaN                     # a ragged edge

    return (; Z, H, Tm, R, Q, a1, P1, y, p, m, r, n)
end

"""
    _singular_setup(; nser, nlag, n, seed)

A companion-form VAR with `R Q R'` of rank `nser` out of `nser · nlag` states,
`H = 0`, and `2 · nser` observed rows covering the current block and the first
lag. The lagged rows repeat the previous period's current block, so after the
first period only `nser` of the `2 · nser` observed directions carry any
variance and `Fₜ` is singular. The data are simulated from these primitives, so
every observation is on-support.
"""
function _singular_setup(; nser::Int = 2, nlag::Int = 3, n::Int = 10,
        seed::Int = 20260921)
    rng = Xoshiro(seed)
    m = nser * nlag
    p = 2 * nser

    Tm = zeros(m, m)
    for l in 1:nlag
        Tm[1:nser, ((l - 1) * nser + 1):(l * nser)] = 0.35^l .*
                                                      randn(rng, nser, nser) ./ sqrt(nser)
    end
    Tm[(nser + 1):end, 1:(nser * (nlag - 1))] = I(nser * (nlag - 1))

    R = zeros(m, nser)
    R[1:nser, :] = I(nser)
    Xq = randn(rng, nser, nser)
    Q = Matrix(Symmetric(Xq * Xq' / nser + 0.5I))
    Z = zeros(p, m)
    Z[1:p, 1:p] = I(p)
    H = zeros(p, p)
    a1 = zeros(m)
    P1 = Matrix(0.6I, m, m)

    Lq = cholesky(Q).L
    y = zeros(p, n)
    a = a1 + cholesky(Symmetric(P1)).L * randn(rng, m)
    for t in 1:n
        y[:, t] = Z * a
        a = Tm * a
        a[1:nser] += Lq * randn(rng, nser)
    end

    y[1, 4] = NaN                     # a hole, so pₜ varies across periods
    y[:, 7] .= NaN

    return (; Z, H, Tm, R, Q, a1, P1, y, nser, nlag, m, n, p)
end

"""
    _dense_posterior(s)

The posterior mean of the stacked state path under a single dense Gaussian
conditioning built from the model primitives alone. With `H = 0` the conditioning
is exact, so it needs a pseudo-inverse.
"""
function _dense_posterior(s)
    m, n, p = s.m, s.n, s.p

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
    return mu + Sig * Zb' * (pinv(Zb * Sig * Zb') * (yv - Zb * mu))
end

function _workspace(s; rank_revealing::Bool = false)
    ws = KalmanWorkspace(s.Z, s.H, s.Tm, s.R, s.Q, s.a1, s.P1, s.n)
    rank_revealing && set_rank_revealing!(ws, true)
    ws
end

@testset "rank-revealing declaration — accessors and validation" begin
    s = _fullrank_setup()
    ws = _workspace(s)
    @test rank_revealing(ws) == false
    @test innovation_ranks(ws) === ws.Ft_rank
    @test length(innovation_ranks(ws)) == s.n

    set_rank_revealing!(ws, true)
    @test rank_revealing(ws) == true
    set_rank_revealing!(ws, false)
    @test rank_revealing(ws) == false

    @test_throws "rank_rtol must be finite" set_rank_revealing!(ws, true; rank_rtol = NaN)
    @test_throws "must be finite" set_rank_revealing!(ws, true; support_rtol = Inf)
    @test_throws ArgumentError set_rank_revealing!(ws, true; rank_rtol = -1.0)
end

@testset "rank-revealing — reproduces the Cholesky path on a full-rank model" begin
    # On a positive definite Fₜ the eigenfactorization spans the same range, so
    # the two paths differ only by the rounding of two different factorizations.
    s = _fullrank_setup()
    wc = _workspace(s)
    wr = _workspace(s; rank_revealing = true)

    llc = filter_and_smooth!(wc, s.y)
    llr = filter_and_smooth!(wr, s.y)

    @test llr ≈ llc atol = 1e-10
    @test maximum(abs, wr.at - wc.at) < 1e-10
    @test maximum(abs, wr.att - wc.att) < 1e-10
    @test maximum(abs, wr.Pt - wc.Pt) < 1e-10
    @test maximum(abs, wr.Ptt - wc.Ptt) < 1e-10
    @test maximum(abs, wr.αs - wc.αs) < 1e-10
    @test maximum(abs, wr.Vs - wc.Vs) < 1e-10

    # Every Fₜ is full rank here, so the rank is the count of observed rows.
    @test innovation_ranks(wr) == wc.n_observed
end

@testset "rank-revealing — singular Q and H = 0 match a dense conditioning" begin
    # The smoother and the dense update share no code, so agreement is evidence
    # about the recursion rather than about a shared implementation.
    s = _singular_setup()
    post = _dense_posterior(s)

    ws = _workspace(s; rank_revealing = true)
    filter_and_smooth!(ws, s.y; crosscov = false, covariances = false)
    @test maximum(abs, vec(ws.αs) - post) < 1e-6 * max(1, maximum(abs, post))

    # The covariance path runs the same factors through a different recursion.
    wv = _workspace(s; rank_revealing = true)
    filter_and_smooth!(wv, s.y)
    @test maximum(abs, vec(wv.αs) - post) < 1e-6 * max(1, maximum(abs, post))
    @test all(isfinite, wv.Vs)
    for t in 1:s.n
        V = wv.Vs[:, :, t]
        @test maximum(abs, V - V') < 1e-8
        @test minimum(eigvals(Symmetric((V + V') / 2))) > -1e-8
    end

    # The Cholesky path cannot factor this Fₜ at all.
    wcx = _workspace(s)
    @test_throws PosDefException kalman_filter!(wcx, s.y)
end

@testset "rank-revealing — the simulation smoother runs on a singular model" begin
    s = _singular_setup()
    ws = _workspace(s; rank_revealing = true)
    kalman_filter!(ws, s.y)

    draws = simulation_smoother(ws, s.y; rng = Xoshiro(31), n_draws = 4)
    @test size(draws) == (s.m, s.n, 4)
    @test all(isfinite, draws)

    # Every draw reproduces the observed rows exactly: with H = 0 those are not
    # free, whatever the shocks did.
    for d in 1:4, t in 1:s.n, i in 1:s.p
        isnan(s.y[i, t]) && continue
        @test s.Z[i, :]' * draws[:, t, d] ≈ s.y[i, t] atol = 1e-8
    end
end

@testset "rank-revealing — an off-support observation is rejected" begin
    # A lagged row is a copy of the previous period's current block, so moving
    # the current block of one period alone contradicts the next period's lagged
    # row: the innovation there leaves the range of F.
    s = _singular_setup()
    ws = _workspace(s; rank_revealing = true)
    kalman_filter!(ws, s.y)
    @test any(t -> !ws.missing_mask[t] &&
                   innovation_ranks(ws)[t] < ws.n_observed[t], 1:s.n)

    yoff = copy(s.y)
    yoff[1, 2] += 1.0

    wb = _workspace(s; rank_revealing = true)
    @test_throws InnovationSupportError kalman_filter!(wb, yoff)
    @test_throws "at period 3" kalman_filter!(wb, yoff)
    @test_throws "incompatible with the model" kalman_filter!(wb, yoff)
end

@testset "rank-revealing — a known start admits only its own observation" begin
    # With P1 = 0 and H = 0 the first period is deterministic: F₁ = 0, and the
    # only admissible observation is Z a₁ itself.
    s = _singular_setup()
    rng = Xoshiro(41)
    a1 = randn(rng, s.m)
    P1 = zeros(s.m, s.m)

    # The whole path is resimulated from this start, so the later periods stay
    # consistent with the first one and only the start is under test.
    Lq = cholesky(s.Q).L
    yk = zeros(s.p, s.n)
    a = copy(a1)
    for t in 1:s.n
        yk[:, t] = s.Z * a
        a = s.Tm * a
        a[1:s.nser] += Lq * randn(rng, s.nser)
    end

    ws = KalmanWorkspace(s.Z, s.H, s.Tm, s.R, s.Q, a1, P1, s.n)
    set_rank_revealing!(ws, true)
    @test kalman_filter!(ws, yk) isa Real
    @test innovation_ranks(ws)[1] == 0
    @test ws.att[:, 1] ≈ a1

    yk[1, 1] += 1.0
    wb = KalmanWorkspace(s.Z, s.H, s.Tm, s.R, s.Q, a1, P1, s.n)
    set_rank_revealing!(wb, true)
    @test_throws "at period 1" kalman_filter!(wb, yk)
end

@testset "rank-revealing — one shock reaching two rows through a lag" begin
    # Investment (state 1) responds to lagged credit (state 2); only credit
    # carries a shock. The two-step forecast-error covariance is then full rank
    # although the one-step innovation covariance has rank one, so both rows can
    # be observed two periods out. One row observed one period out cannot: at
    # that horizon the investment row has no variance at all.
    Tm = [0.0 0.5; 0.0 0.0]
    R = reshape([0.0, 1.0], 2, 1)
    Q = fill(1.0, 1, 1)
    Z = Matrix(1.0I, 2, 2)
    H = zeros(2, 2)
    a1 = zeros(2)
    P1 = zeros(2, 2)
    n = 3

    rng = Xoshiro(20260922)
    path = zeros(2, n)
    a = copy(a1)
    for t in 1:n
        path[:, t] = a
        a = Tm * a + R * randn(rng, 1)
    end

    y = fill(NaN, 2, n)
    y[:, 1] = Z * path[:, 1]          # the known start, observed exactly
    y[:, 3] = Z * path[:, 3]          # both rows, on-support two periods out

    ws = KalmanWorkspace(Z, H, Tm, R, Q, a1, P1, n)
    set_rank_revealing!(ws, true)
    @test filter_and_smooth!(ws, y) isa Real
    @test innovation_ranks(ws) == [0, 0, 2]
    @test ws.Ft[:, :, 3] ≈ [0.25 0.0; 0.0 1.0]
    @test maximum(abs, ws.αs - path) < 1e-10

    # P_{2|1} = R Q R' = diag(0, 1): the investment row carries no variance one
    # period out, so any nonzero innovation on it is off-support.
    ybad = fill(NaN, 2, n)
    ybad[:, 1] = Z * path[:, 1]
    ybad[1, 2] = path[1, 2] + 1.0

    wb = KalmanWorkspace(Z, H, Tm, R, Q, a1, P1, n)
    set_rank_revealing!(wb, true)
    @test_throws "at period 2" kalman_filter!(wb, ybad)
end

@testset "rank-revealing — the diffuse filter rejects the declaration" begin
    s = _fullrank_setup()
    ws = DiffuseKalmanWorkspace(s.p, s.m, s.r, s.n)
    set_params!(ws, s.Z, s.H, s.Tm, s.R, s.Q)
    set_initial_diffuse!(ws, s.a1, zeros(s.m, s.m), Matrix(1.0I, s.m, s.m))

    set_rank_revealing!(ws.base, true)
    yfull = replace(s.y, NaN => 0.0)
    @test_throws ArgumentError kalman_filter!(ws, yfull)
    @test_throws "KalmanWorkspace only" kalman_filter!(ws, yfull)
end
