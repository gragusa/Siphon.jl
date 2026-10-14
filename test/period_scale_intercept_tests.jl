# Transition intercept and per-period disturbance scales on a KalmanWorkspace:
#     αₜ₊₁ = c + T αₜ + R ηₜ,   ηₜ ~ N(0, sₜ² Q).
# Every result is checked against dense Gaussian conditioning of the stacked
# state path, which shares no code with the filter.

using Test
using Siphon
using LinearAlgebra
using Random
using Statistics
using TOML

_ps_block(t, m) = ((t - 1) * m + 1):(t * m)

# Prior mean and covariance of the stacked path (α₁, …, αₙ):
#     αₜ = T^{t-1} α₁ + Σ_{u<t} T^{t-1-u} (c + R ηᵤ),   Var(ηᵤ) = sᵤ² Q.
function _ps_prior(Tm, R, Q, c, s, a1, P1, n)
    m = size(Tm, 1)
    Tpow = [Matrix{Float64}(I, m, m)]
    for _ in 1:n
        push!(Tpow, Tm * Tpow[end])
    end
    RQR = R * Q * R'
    mu = zeros(m * n)
    Sig = zeros(m * n, m * n)
    for t in 1:n
        μ = Tpow[t] * a1
        for u in 1:(t - 1)
            μ += Tpow[t - u] * c
        end
        mu[_ps_block(t, m)] = μ
    end
    for t in 1:n, t2 in 1:n

        b = Tpow[t] * P1 * Tpow[t2]'
        for u in 1:(min(t, t2) - 1)
            b += s[u]^2 * Tpow[t - u] * RQR * Tpow[t2 - u]'
        end
        Sig[_ps_block(t, m), _ps_block(t2, m)] = b
    end
    return mu, Matrix(Symmetric((Sig + Sig') / 2))
end

# Posterior mean and covariance of the stacked path given the observed entries of
# `y`, and the log density of those entries.
function _ps_posterior(Z, H, Tm, R, Q, c, s, a1, P1, y)
    p, m = size(Z)
    n = size(y, 2)
    mu, Sig = _ps_prior(Tm, R, Q, c, s, a1, P1, n)
    obs = [(t, i) for t in 1:n for i in 1:p if !isnan(y[i, t])]
    q = length(obs)
    Zb = zeros(q, m * n)
    Hb = zeros(q, q)
    yv = zeros(q)
    for (k, (t, i)) in pairs(obs)
        Zb[k, _ps_block(t, m)] = Z[i, :]
        yv[k] = y[i, t]
        for (l, (t2, j)) in pairs(obs)
            t2 == t && (Hb[k, l] = H[i, j])
        end
    end
    S = Symmetric(Zb * Sig * Zb' + Hb)
    e = yv - Zb * mu
    loglik = -(q * log(2π) + logdet(S) + dot(e, S \ e)) / 2
    K = (Sig * Zb') / S
    return (mean = mu + K * e, cov = Matrix(Symmetric(Sig - K * Zb * Sig)), loglik)
end

# A VAR(2) in companion form (block width 2, m = 4) with an intercept, scales that
# rise and decay, and a mixed observation pattern: row 3 averages the first series
# over two periods and is seen every third period, period 5 is entirely missing,
# and the last period loses row 1.
function _ps_setup(; H = [0.3 0.05 0.0; 0.05 0.2 0.0; 0.0 0.0 0.1], seed = 20261014)
    rng = Xoshiro(seed)
    A1 = [0.5 0.1; 0.2 0.3]
    A2 = [0.1 0.0; 0.05 0.2]
    Tm = [A1 A2; I zeros(2, 2)]
    R = [Matrix(1.0I, 2, 2); zeros(2, 2)]
    Q = [1.0 0.3; 0.3 0.5]
    Z = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.5 0.0 0.5 0.0]
    c = [0.3, -0.2, 0.0, 0.0]
    a1 = [0.1, -0.1, 0.2, 0.0]
    P1 = Matrix(Diagonal([0.8, 0.6, 0.7, 0.9]))
    n = 10
    s = [1.0, 1.0, 1.0, 4.0, 3.0, 2.0, 1.5, 1.2, 1.1, 1.7]

    p = size(Z, 1)
    y = zeros(p, n)
    α = a1 + cholesky(P1).L * randn(rng, 4)
    LH = cholesky(Symmetric(H + 1e-300I)).L
    for t in 1:n
        y[:, t] = Z * α + (iszero(H) ? zeros(p) : LH * randn(rng, p))
        α = c + Tm * α + s[t] * (R * (cholesky(Q).L * randn(rng, 2)))
    end
    for t in 1:n
        t % 3 == 0 || (y[3, t] = NaN)
    end
    y[:, 5] .= NaN
    y[1, n] = NaN
    return (; Z, H, Tm, R, Q, c, a1, P1, s, y, n)
end

function _ps_workspace(d; block = 0, Z = d.Z, H = d.H)
    ws = KalmanWorkspace(Z, H, d.Tm, d.R, d.Q, d.a1, d.P1, d.n)
    set_companion_structure!(ws, block)
    Siphon.set_transition_intercept!(ws, d.c)
    Siphon.set_period_scales!(ws, d.s)
    return ws
end

@testset "period scales and intercept — filter and smoother match dense conditioning" begin
    d = _ps_setup()
    ref = _ps_posterior(d.Z, d.H, d.Tm, d.R, d.Q, d.c, d.s, d.a1, d.P1, d.y)
    m = 4
    for block in (0, 2)
        ws = _ps_workspace(d; block)
        @test kalman_filter!(ws, d.y) ≈ ref.loglik rtol = 1e-10

        kalman_smoother!(ws)
        @test vec(smoothed_states(ws)) ≈ ref.mean atol = 1e-10
        V = smoothed_states_cov(ws)
        for t in 1:(d.n)
            @test V[:, :, t] ≈ ref.cov[_ps_block(t, m), _ps_block(t, m)] atol = 1e-10
        end

        kalman_smoother!(ws; covariances = false)
        @test vec(smoothed_states(ws)) ≈ ref.mean atol = 1e-10
    end

    # Without the intercept and scales the result differs, so the test above
    # exercises both.
    ws = KalmanWorkspace(d.Z, d.H, d.Tm, d.R, d.Q, d.a1, d.P1, d.n)
    @test !isapprox(kalman_filter!(ws, d.y), ref.loglik; rtol = 1e-3)
end

@testset "period scales and intercept — cross-lag covariances" begin
    d = _ps_setup()
    ref = _ps_posterior(d.Z, d.H, d.Tm, d.R, d.Q, d.c, d.s, d.a1, d.P1, d.y)
    m = 4
    ws = _ps_workspace(d; block = 2)
    kalman_filter!(ws, d.y)
    kalman_smoother!(ws; crosscov = true)
    Pc = crosslag_covs(ws)
    for t in 1:(d.n - 1)
        @test Pc[:, :, t] ≈ ref.cov[_ps_block(t + 1, m), _ps_block(t, m)] atol = 1e-10
    end
end

@testset "period scales and intercept — rank-revealing update" begin
    # Rows 1 and 2 observed twice with H = 0, so every observed Fₜ is singular.
    # The duplicated system has the posterior of the system without duplicates.
    d = _ps_setup(; H = zeros(3, 3))
    Z2 = [d.Z[1:2, :]; d.Z[1:2, :]]
    y2 = [d.y[1:2, :]; d.y[1:2, :]]
    ref = _ps_posterior(d.Z[1:2, :], zeros(2, 2), d.Tm, d.R, d.Q, d.c, d.s, d.a1,
        d.P1, d.y[1:2, :])
    ws = _ps_workspace(d; block = 2, Z = Z2, H = zeros(4, 4))
    set_rank_revealing!(ws, true)
    kalman_filter!(ws, y2)
    @test maximum(innovation_ranks(ws)) == 2
    kalman_smoother!(ws)
    @test vec(smoothed_states(ws)) ≈ ref.mean atol = 1e-9
    V = smoothed_states_cov(ws)
    for t in 1:(d.n)
        @test V[:, :, t] ≈ ref.cov[_ps_block(t, 4), _ps_block(t, 4)] atol = 1e-9
    end
end

@testset "period scales and intercept — disturbance smoother" begin
    # With R'R = I, ηₜ = R'(αₜ₊₁ − c − T αₜ) is a linear map of the stacked path.
    d = _ps_setup()
    ref = _ps_posterior(d.Z, d.H, d.Tm, d.R, d.Q, d.c, d.s, d.a1, d.P1, d.y)
    m = 4
    ws = _ps_workspace(d; block = 2)
    kalman_filter!(ws, d.y)
    ds = disturbance_smoother(ws)
    ηs = smoothed_state_disturbances(ds)
    Vη = smoothed_state_disturbances_cov(ds)
    for t in 1:(d.n - 1)
        G = zeros(2, m * d.n)
        G[:, _ps_block(t + 1, m)] = d.R'
        G[:, _ps_block(t, m)] = -d.R' * d.Tm
        @test ηs[:, t] ≈ G * ref.mean - d.R' * d.c atol = 1e-10
        @test Vη[:, :, t] ≈ G * ref.cov * G' atol = 1e-10
    end
    # The last disturbance is not informed by the data.
    @test ηs[:, d.n] == zeros(2)
    @test Vη[:, :, d.n] ≈ d.s[d.n]^2 * d.Q
end

@testset "period scales and intercept — simulation smoother draws" begin
    d = _ps_setup()
    ref = _ps_posterior(d.Z, d.H, d.Tm, d.R, d.Q, d.c, d.s, d.a1, d.P1, d.y)
    ws = _ps_workspace(d; block = 2)
    D = simulation_smoother(ws, d.y, 40_000; rng = Xoshiro(11))
    flat = reshape(D, size(D, 1) * size(D, 2), size(D, 3))
    mu, Sig = vec(mean(flat, dims = 2)), cov(flat, dims = 2)
    scale = maximum(abs, ref.cov)
    @test maximum(abs, mu - ref.mean) < 0.03 * sqrt(scale)
    @test maximum(abs, Sig - ref.cov) < 0.05 * scale

    # The cached factors are those of the unscaled Q, so new scales take effect
    # without refresh_factors!.
    sws = Siphon.SimulationSmootherWorkspace(ws)
    s_new = ones(d.n)
    Siphon.set_period_scales!(ws, s_new)
    ref1 = _ps_posterior(d.Z, d.H, d.Tm, d.R, d.Q, d.c, s_new, d.a1, d.P1, d.y)
    out = zeros(4, d.n)
    draws = zeros(4 * d.n, 20_000)
    rng = Xoshiro(12)
    for k in axes(draws, 2)
        draws[:, k] = vec(Siphon.simulation_smoother!(out, sws, ws, d.y; rng))
    end
    scale1 = maximum(abs, ref1.cov)
    @test maximum(abs, cov(draws, dims = 2) - ref1.cov) < 0.05 * scale1
end

@testset "period scales — ecbindex smoother fixture" begin
    # Reference values from the Python ecbindex package (H = 0, a quarterly flow
    # row, a ragged edge). Its scales[t] scales the disturbance entering period t,
    # which in Siphon's timing is s[t - 1].
    fx = TOML.parsefile(joinpath(@__DIR__, "ecbindex_smoother.toml"))
    rows(A) = reduce(vcat, permutedims.(Vector{Float64}.(A)))
    sy = fx["system"]
    Tm, R, Q, Z = rows(sy["T"]), rows(sy["R"]), rows(sy["Q"]), rows(sy["Z"])
    a1, P1 = Vector{Float64}(sy["a1"]), rows(sy["P1"])
    y = rows(sy["y"])
    p, n = size(y)
    s = [Float64.(sy["scales"])[2:end]; 1.0]

    ws = KalmanWorkspace(Z, zeros(p, p), Tm, R, Q, a1, P1, n)
    set_companion_structure!(ws, size(Q, 1))
    Siphon.set_period_scales!(ws, s)
    ll = kalman_filter!(ws, y)
    kalman_smoother!(ws)

    sm = fx["smoother"]
    @test ll ≈ sm["loglik"] rtol = 1e-10
    @test smoothed_states(ws) ≈ rows(sm["smoothed"]) atol = 1e-10
    @test smoothed_states(ws) ≈ rows(sm["dense_mean"]) atol = 1e-10
    V = smoothed_states_cov(ws)
    Vdiag = reduce(hcat, [diag(V[:, :, t]) for t in 1:n])
    @test Vdiag ≈ rows(sm["dense_var"]) atol = 1e-10
end

@testset "period scales and intercept — defaults and argument checks" begin
    d = _ps_setup()
    ws0 = KalmanWorkspace(d.Z, d.H, d.Tm, d.R, d.Q, d.a1, d.P1, d.n)
    @test Siphon.transition_intercept(ws0) == zeros(4)
    @test Siphon.period_scales(ws0) == ones(d.n)
    ll0 = kalman_filter!(ws0, d.y)

    # Explicit defaults give the plain filter bit for bit.
    ws1 = KalmanWorkspace(d.Z, d.H, d.Tm, d.R, d.Q, d.a1, d.P1, d.n)
    Siphon.set_transition_intercept!(ws1, zeros(4))
    Siphon.set_period_scales!(ws1, ones(d.n))
    @test kalman_filter!(ws1, d.y) == ll0
    @test filtered_states(ws1) == filtered_states(ws0)
    @test loglikelihood(KalmanFilterResult(ws1)) == ll0

    @test_throws DimensionMismatch Siphon.set_period_scales!(ws1, ones(d.n - 1))
    @test_throws DimensionMismatch Siphon.set_transition_intercept!(ws1, zeros(3))
    @test_throws "finite and non-negative" Siphon.set_period_scales!(ws1,
        [-1.0; ones(d.n - 1)])
    @test_throws "finite and non-negative" Siphon.set_period_scales!(ws1,
        [NaN; ones(d.n - 1)])
    @test_throws "must be finite" Siphon.set_transition_intercept!(ws1, [Inf, 0, 0, 0])
end

@testset "period scales and intercept — unsupported consumers throw" begin
    d = _ps_setup()
    ws = _ps_workspace(d)
    kalman_filter!(ws, d.y)
    @test_throws "KalmanFilterResult does not support a transition intercept and period scales" KalmanFilterResult(ws)

    Siphon.set_transition_intercept!(ws, zeros(4))
    @test_throws "does not support period scales" KalmanFilterResult(ws)
    Siphon.set_period_scales!(ws, ones(d.n))
    Siphon.set_transition_intercept!(ws, d.c)
    @test_throws "does not support a transition intercept." KalmanFilterResult(ws)

    # EM on fully observed data with an intercept.
    yfull = d.Z * randn(Xoshiro(3), 4, d.n)
    em_ws = Siphon.EMWorkspace(ws)
    @test_throws "em_estimate! does not support a transition intercept" Siphon.em_estimate!(
        ws, em_ws, yfull; maxiter = 2, warn = false)
    kalman_filter!(ws, yfull)
    kalman_smoother!(ws; crosscov = true)
    @test_throws "compute_sufficient_stats! does not support" Siphon.compute_sufficient_stats!(
        em_ws, ws, yfull)

    dws = DiffuseKalmanWorkspace(d.Z, d.H, d.Tm, d.R, d.Q, d.a1, zeros(4, 4),
        Matrix(1.0I, 4, 4), d.n)
    Siphon.set_period_scales!(dws.base, d.s)
    @test_throws "the exact diffuse filter does not support period scales" kalman_filter!(
        dws, d.y)
end
