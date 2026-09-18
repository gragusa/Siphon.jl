# Simulation smoother: draws must come from the conditional distribution of the
# state path, matching an analytic posterior in covariance as well as in mean.

using Test
using Siphon
using LinearAlgebra
using Random
using Statistics

"""
    _joint_posterior(Z, H, T, R, Q, a1, P1, y)

The exact posterior of the stacked state path, obtained by writing the whole
model as one joint Gaussian over `(α₁:ₙ, y₁:ₙ)` and conditioning on the observed
entries of `y`. Returns `(mean, covariance)` of length/size `m·n`.

Builds the stacked prior directly from the primitives and shares no code with
the filter, so agreement is evidence rather than tautology. Only usable on tiny
models — the covariance is `m·n × m·n` and dense.
"""
function _joint_posterior(Z, H, Tm, R, Q, a1, P1, y)
    p, m = size(Z)
    n = size(y, 2)

    # Prior over the stacked path: αₜ = T^{t-1} α₁ + Σ_{u<t} T^{t-1-u} R η_u.
    Tpow = [Matrix{Float64}(I, m, m) for _ in 0:n]
    for k in 1:n
        Tpow[k + 1] = Tm * Tpow[k]
    end
    mu = zeros(m * n)
    Sig = zeros(m * n, m * n)
    for t in 1:n
        mu[((t - 1) * m + 1):(t * m)] = Tpow[t] * a1
    end
    for t in 1:n, s in 1:n

        blk = Tpow[t] * P1 * Tpow[s]'
        for u in 1:(min(t, s) - 1)
            blk += Tpow[t - u] * R * Q * R' * Tpow[s - u]'
        end
        Sig[((t - 1) * m + 1):(t * m), ((s - 1) * m + 1):(s * m)] = blk
    end
    Sig = Matrix(Symmetric((Sig + Sig') / 2))

    # Condition on the observed entries only.
    obs = [(t, i) for t in 1:n for i in 1:p if !isnan(y[i, t])]
    q = length(obs)
    Zb = zeros(q, m * n)
    Hb = zeros(q, q)
    yv = zeros(q)
    for (k, (t, i)) in pairs(obs)
        Zb[k, ((t - 1) * m + 1):(t * m)] = Z[i, :]
        yv[k] = y[i, t]
        for (l, (t2, j)) in pairs(obs)
            if t2 == t
                Hb[k, l] = H[i, j]
            end
        end
    end
    K = (Sig * Zb') / Symmetric(Zb * Sig * Zb' + Hb)
    return mu + K * (yv - Zb * mu), Matrix(Symmetric(Sig - K * Zb * Sig))
end

"""
    _simsmooth_setup(; p=3, m=2, n=8, seed=20260918)

A small stable model with nonsingular `H`, `Q` and `P1`, together with one
dataset simulated from it.
"""
function _simsmooth_setup(; p::Int = 3, m::Int = 2, n::Int = 8, seed::Int = 20260918)
    rng = Xoshiro(seed)
    Z = randn(rng, p, m)
    Xh = randn(rng, p, p)
    H = Matrix(Symmetric(Xh * Xh' + 0.5I))
    Tm = randn(rng, m, m)
    Tm ./= 2 * maximum(abs, eigvals(Tm))
    R = Matrix{Float64}(I, m, m)
    Xq = randn(rng, m, m)
    Q = Matrix(Symmetric(Xq * Xq' + 0.3I))
    a1 = randn(rng, m)
    Xp = randn(rng, m, m)
    P1 = Matrix(Symmetric(Xp * Xp' + 0.4I))

    y = zeros(p, n)
    a = a1 + cholesky(P1).L * randn(rng, m)
    for t in 1:n
        y[:, t] = Z * a + cholesky(H).L * randn(rng, p)
        a = Tm * a + R * (cholesky(Q).L * randn(rng, size(Q, 1)))
    end

    return (; Z, H, Tm, R, Q, a1, P1, y, p, m, n)
end

"""
    _draw_moments(ws, y, n_draws, seed)

Mean and covariance of `n_draws` simulation-smoother draws, with the state path
flattened to a single `m·n` vector so the covariance covers across-period
dependence as well as within-period.
"""
function _draw_moments(ws, y, n_draws::Int, seed::Int)
    D = simulation_smoother(ws, y; rng = Xoshiro(seed), n_draws = n_draws)
    flat = reshape(D, size(D, 1) * size(D, 2), n_draws)
    return vec(mean(flat, dims = 2)), cov(flat, dims = 2)
end

@testset "psd_factor — factors a rank-deficient matrix" begin
    X = randn(Xoshiro(1), 5, 3)
    S = Matrix(Symmetric(X * X'))          # rank 3 of 5
    f = Siphon.psd_factor(S)
    @test rank(S) == 3
    @test f.L * f.L' ≈ S atol = 1e-12
    @test size(f) == (5, 5)
end

@testset "psd_factor — accepts an exactly zero matrix" begin
    # H = 0 is a legitimate noiseless measurement equation, not an error.
    f = Siphon.psd_factor(zeros(3, 3))
    @test iszero(f.L)
    @test f.L * f.L' ≈ zeros(3, 3)
end

@testset "psd_factor — rejects a negative eigenvalue" begin
    # A negative variance is a modeling error; clamping it would hide the cause.
    @test_throws "not positive semi-definite" Siphon.psd_factor([1.0 0.0; 0.0 -0.5])
    @test_throws "smallest eigenvalue" Siphon.psd_factor([1.0 0.0; 0.0 -0.5])
end

@testset "psd_factor — a draw has the requested covariance" begin
    X = randn(Xoshiro(2), 4, 2)
    S = Matrix(Symmetric(X * X'))
    f = Siphon.psd_factor(S)
    rng = Xoshiro(3)
    out = zeros(4)
    scratch = zeros(4)
    draws = Matrix{Float64}(undef, 4, 40_000)
    for k in axes(draws, 2)
        draws[:, k] = Siphon.psd_rand!(out, f, rng, scratch)
    end
    @test maximum(abs, cov(draws, dims = 2) - S) < 0.08 * max(1, maximum(abs, S))
    @test maximum(abs, vec(mean(draws, dims = 2))) < 0.05
end

@testset "simulation smoother — draws match the analytic posterior" begin
    s = _simsmooth_setup()
    mu_ref, Sig_ref = _joint_posterior(s.Z, s.H, s.Tm, s.R, s.Q, s.a1, s.P1, s.y)
    ws = KalmanWorkspace(s.Z, s.H, s.Tm, s.R, s.Q, s.a1, s.P1, s.n)
    mu, Sig = _draw_moments(ws, s.y, 40_000, 7)

    scale = maximum(abs, Sig_ref)
    # The covariance is the part that a plausible-looking but wrong smoother
    # gets wrong while still reproducing the mean, so it is tested too.
    @test maximum(abs, mu - mu_ref) < 0.03 * sqrt(scale)
    @test maximum(abs, Sig - Sig_ref) < 0.05 * scale
end

@testset "simulation smoother — Monte Carlo error falls at the root-N rate" begin
    # Separates sampling noise, which shrinks with the number of draws, from a
    # systematic bias, which would not. Each replication uses its own seed: a
    # shared one would make the larger run reuse the smaller run's draws and
    # inherit its particular deviation.
    s = _simsmooth_setup()
    mu_ref, _ = _joint_posterior(s.Z, s.H, s.Tm, s.R, s.Q, s.a1, s.P1, s.y)
    ws = KalmanWorkspace(s.Z, s.H, s.Tm, s.R, s.Q, s.a1, s.P1, s.n)

    mean_err(n_draws, seeds) = mean(
        maximum(abs, first(_draw_moments(ws, s.y, n_draws, sd)) - mu_ref)
    for sd in seeds)

    err_small = mean_err(2_500, 101:108)
    err_large = mean_err(40_000, 201:208)
    @test err_large < err_small / 2
end

@testset "simulation smoother — draws average to the smoothed mean" begin
    s = _simsmooth_setup()
    ws = KalmanWorkspace(s.Z, s.H, s.Tm, s.R, s.Q, s.a1, s.P1, s.n)
    filter_and_smooth!(ws, s.y; crosscov = false)
    α_hat = copy(smoothed_states(ws))

    mu, _ = _draw_moments(ws, s.y, 40_000, 5)
    @test maximum(abs, mu - vec(α_hat)) < 0.03 * sqrt(maximum(abs, α_hat) + 1)
end

@testset "simulation smoother — partially observed periods" begin
    # Mixed-frequency censoring with a ragged edge: row 1 is seen every third
    # period, and the last period loses row 2.
    s = _simsmooth_setup()
    y = copy(s.y)
    for t in 1:(s.n)
        if t % 3 != 0
            y[1, t] = NaN
        end
    end
    y[2, s.n] = NaN

    mu_ref, Sig_ref = _joint_posterior(s.Z, s.H, s.Tm, s.R, s.Q, s.a1, s.P1, y)
    ws = KalmanWorkspace(s.Z, s.H, s.Tm, s.R, s.Q, s.a1, s.P1, s.n)
    mu, Sig = _draw_moments(ws, y, 40_000, 3)

    scale = maximum(abs, Sig_ref)
    @test maximum(abs, mu - mu_ref) < 0.03 * sqrt(scale)
    @test maximum(abs, Sig - Sig_ref) < 0.05 * scale
end

@testset "simulation smoother — singular Q and a noiseless measurement" begin
    # A companion-form VAR: the shock enters only the first lag block, so R Q R'
    # is rank 2 of 4, and the observation equation is an exact selection.
    # `cholesky` rejects both matrices; the draw must still be correct.
    m = 4
    Tm = [0.5 0.1 0.1 0.0
          -0.2 0.4 0.05 0.1
          1.0 0.0 0.0 0.0
          0.0 1.0 0.0 0.0]
    R = [Matrix{Float64}(I, 2, 2); zeros(2, 2)]
    Q = [0.6 0.2; 0.2 0.5]
    Z = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]
    H = zeros(2, 2)
    a1 = zeros(m)
    P1 = Matrix(0.9I, m, m)
    n = 10

    @test rank(R * Q * R') == 2
    @test iszero(H)

    rng = Xoshiro(5)
    y = zeros(2, n)
    a = a1 + cholesky(Symmetric(P1)).L * randn(rng, m)
    for t in 1:n
        y[:, t] = Z * a
        a = Tm * a + R * (cholesky(Q).L * randn(rng, 2))
    end

    mu_ref, Sig_ref = _joint_posterior(Z, H, Tm, R, Q, a1, P1, y)
    ws = KalmanWorkspace(Z, H, Tm, R, Q, a1, P1, n)
    mu, Sig = _draw_moments(ws, y, 40_000, 13)

    scale = maximum(abs, Sig_ref)
    @test maximum(abs, mu - mu_ref) < 0.03 * sqrt(scale)
    @test maximum(abs, Sig - Sig_ref) < 0.05 * scale
end

@testset "simulation smoother — the observed rows are reproduced exactly" begin
    # With H = 0 the observation pins a linear combination of the state, so
    # every draw must satisfy it. This is what fails when the synthetic data is
    # not censored to the real missing pattern.
    Tm = [0.6 0.0; 0.0 0.3]
    R = Matrix{Float64}(I, 2, 2)
    Q = [0.4 0.1; 0.1 0.5]
    Z = [1.0 0.0]
    H = zeros(1, 1)
    a1 = zeros(2)
    P1 = Matrix(1.0I, 2, 2)
    n = 6
    y = reshape([0.3, -0.1, 0.8, NaN, 0.2, -0.4], 1, n)

    ws = KalmanWorkspace(Z, H, Tm, R, Q, a1, P1, n)
    draws = simulation_smoother(ws, y; rng = Xoshiro(21), n_draws = 25)
    for d in 1:25, t in 1:n

        if !isnan(y[1, t])
            @test draws[1, t, d] ≈ y[1, t] atol = 1e-8
        end
    end
    # The unobserved period is genuinely random rather than pinned.
    @test std(draws[1, 4, :]) > 1e-3
end

@testset "simulation smoother — an explicit rng makes a draw reproducible" begin
    s = _simsmooth_setup()
    ws = KalmanWorkspace(s.Z, s.H, s.Tm, s.R, s.Q, s.a1, s.P1, s.n)
    d1 = simulation_smoother(ws, s.y; rng = Xoshiro(4), n_draws = 3)
    d2 = simulation_smoother(ws, s.y; rng = Xoshiro(4), n_draws = 3)
    d3 = simulation_smoother(ws, s.y; rng = Xoshiro(5), n_draws = 3)
    @test d1 == d2
    @test d1 != d3
end

@testset "simulation smoother — shape and argument checks" begin
    s = _simsmooth_setup()
    ws = KalmanWorkspace(s.Z, s.H, s.Tm, s.R, s.Q, s.a1, s.P1, s.n)

    @test size(simulation_smoother(ws, s.y; rng = Xoshiro(1))) == (s.m, s.n)
    @test size(simulation_smoother(ws, s.y; rng = Xoshiro(1), n_draws = 4)) ==
          (s.m, s.n, 4)
    @test_throws "n_draws must be at least 1" simulation_smoother(
        ws, s.y; n_draws = 0)

    sws = SimulationSmootherWorkspace(ws)
    out = zeros(s.m, s.n)
    @test_throws DimensionMismatch simulation_smoother!(
        out, sws, ws, s.y[:, 1:(s.n - 1)])
    @test_throws DimensionMismatch simulation_smoother!(
        zeros(s.m, s.n - 1), sws, ws, s.y)
end

@testset "simulation smoother — refresh_factors! picks up a new Q" begin
    # The factors are cached, so a model change that bypasses them would keep
    # simulating from the old Q while the filter used the new one.
    s = _simsmooth_setup()
    ws = KalmanWorkspace(s.Z, s.H, s.Tm, s.R, s.Q, s.a1, s.P1, s.n)
    sws = SimulationSmootherWorkspace(ws)
    Q2 = 9.0 * s.Q
    Siphon.update_Q!(ws, Q2)
    refresh_factors!(sws, ws)
    @test sws.Q_factor.L * sws.Q_factor.L' ≈ Q2 atol = 1e-10
end
