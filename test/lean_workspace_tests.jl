# A KalmanWorkspace built with `covariances = false` stores no per-period state
# covariances. Its smoothed means come from the fast state smoother, which shares
# only the rₜ recursion with the full smoother, so they are checked against dense
# Gaussian conditioning and against a full workspace on the same model.

using Test
using Siphon
using LinearAlgebra
using Random
using Statistics

# The dense-conditioning oracle `_ps_posterior` and the model `_ps_setup`.
isdefined(@__MODULE__, :_ps_posterior) || include("period_scale_intercept_tests.jl")

function _lean_pair(d; block = 0, Z = d.Z, H = d.H)
    make(covariances) = begin
        ws = KalmanWorkspace(Z, H, d.Tm, d.R, d.Q, d.a1, d.P1, d.n; covariances)
        set_companion_structure!(ws, block)
        Siphon.set_transition_intercept!(ws, d.c)
        Siphon.set_period_scales!(ws, d.s)
        ws
    end
    return make(true), make(false)
end

@testset "lean workspace — storage" begin
    d = _ps_setup()
    full, lean = _lean_pair(d)
    m, n = 4, d.n
    @test size(lean.Pt) == (m, m, 1)
    @test size(lean.Ptt) == (m, m, 1)
    @test size(lean.Vs) == (m, m, 0)
    @test size(lean.Pcross) == (m, m, 0)
    @test size(full.Pt) == (m, m, n)
    @test Base.summarysize(lean) < Base.summarysize(full)

    @test repr(KalmanWorkspace(3, 4, 2, 10; covariances = false)) ==
          "KalmanWorkspace{Float64}(p = 3, m = 4, r = 2, n = 10, covariances = false, not run)"
    @test KalmanWorkspace{Float32}(3, 4, 2, 10; covariances = false).Pt isa
          Array{Float32, 3}
    kf = KFParms(d.Z, d.H, d.Tm, d.R, d.Q)
    @test size(KalmanWorkspace(kf, d.a1, d.P1, n; covariances = false).Pt, 3) == 1
end

@testset "lean workspace — smoothed means match dense conditioning" begin
    d = _ps_setup()
    ref = _ps_posterior(d.Z, d.H, d.Tm, d.R, d.Q, d.c, d.s, d.a1, d.P1, d.y)
    for block in (0, 2)
        full, lean = _lean_pair(d; block)
        @test kalman_filter!(lean, d.y) ≈ ref.loglik rtol = 1e-10
        @test kalman_filter!(full, d.y) == loglikelihood(lean)

        # The filter output that the lean workspace keeps is the same.
        @test predicted_states(lean) == predicted_states(full)
        @test filtered_states(lean) == filtered_states(full)
        @test lean.vt[.!isnan.(lean.vt)] == full.vt[.!isnan.(full.vt)]
        @test lean.Kt == full.Kt

        kalman_smoother!(lean)
        @test vec(smoothed_states(lean)) ≈ ref.mean atol = 1e-10
        kalman_smoother!(full; covariances = false)
        @test smoothed_states(lean) ≈ smoothed_states(full) atol = 1e-12
    end
end

@testset "lean workspace — missing first and last periods, no intercept or scales" begin
    d = _ps_setup()
    y = copy(d.y)
    y[:, 1] .= NaN
    y[:, d.n] .= NaN
    s1 = ones(d.n)
    ref = _ps_posterior(d.Z, d.H, d.Tm, d.R, d.Q, zeros(4), s1, d.a1, d.P1, y)
    lean = KalmanWorkspace(d.Z, d.H, d.Tm, d.R, d.Q, d.a1, d.P1, d.n; covariances = false)
    @test kalman_filter_and_smooth!(lean, y) ≈ ref.loglik rtol = 1e-10
    @test vec(smoothed_states(lean)) ≈ ref.mean atol = 1e-10
end

@testset "lean workspace — rank-revealing update" begin
    # Rows 1 and 2 observed twice with H = 0, so every observed Fₜ is singular.
    d = _ps_setup(; H = zeros(3, 3))
    Z2 = [d.Z[1:2, :]; d.Z[1:2, :]]
    y2 = [d.y[1:2, :]; d.y[1:2, :]]
    ref = _ps_posterior(d.Z[1:2, :], zeros(2, 2), d.Tm, d.R, d.Q, d.c, d.s, d.a1,
        d.P1, d.y[1:2, :])
    _, lean = _lean_pair(d; block = 2, Z = Z2, H = zeros(4, 4))
    set_rank_revealing!(lean, true)
    kalman_filter_and_smooth!(lean, y2)
    @test maximum(innovation_ranks(lean)) == 2
    @test vec(smoothed_states(lean)) ≈ ref.mean atol = 1e-9
end

@testset "lean workspace — disturbance and simulation smoothers" begin
    d = _ps_setup()
    full, lean = _lean_pair(d; block = 2)
    kalman_filter!(full, d.y)
    kalman_filter!(lean, d.y)
    df, dl = disturbance_smoother(full), disturbance_smoother(lean)
    @test smoothed_state_disturbances(dl) == smoothed_state_disturbances(df)
    @test smoothed_state_disturbances_cov(dl) == smoothed_state_disturbances_cov(df)

    # The same random stream gives the same draws; the two smoothed means differ
    # only by rounding.
    Df = simulation_smoother(full, d.y, 50; rng = Xoshiro(5))
    Dl = simulation_smoother(lean, d.y, 50; rng = Xoshiro(5))
    @test Dl ≈ Df atol = 1e-10

    sws = Siphon.SimulationSmootherWorkspace(lean)
    out = zeros(4, d.n)
    Siphon.simulation_smoother!(out, sws, lean, d.y; rng = Xoshiro(5))
    @test out ≈ simulation_smoother(full, d.y; rng = Xoshiro(5)) atol = 1e-10
end

@testset "lean workspace — no allocation in the filter and smoother" begin
    d = _ps_setup()
    full, lean = _lean_pair(d; block = 2)
    run!(ws) = (kalman_filter!(ws, d.y); kalman_smoother!(ws; covariances = false))
    run!(full)
    run!(lean)
    @test (@allocated run!(lean)) <= (@allocated run!(full))
    @test (@allocated kalman_smoother!(lean)) == 0
end

@testset "lean workspace — covariance consumers throw" begin
    d = _ps_setup()
    lean = KalmanWorkspace(d.Z, d.H, d.Tm, d.R, d.Q, d.a1, d.P1, d.n; covariances = false)
    yfull = d.Z * randn(Xoshiro(3), 4, d.n)
    kalman_filter_and_smooth!(lean, yfull)
    msg = "does not store per-period state covariances"
    @test_throws msg predicted_states_cov(lean)
    @test_throws msg filtered_states_cov(lean)
    @test_throws msg smoothed_states_cov(lean)
    @test_throws msg crosslag_covs(lean)
    @test_throws msg SmootherResult(lean)
    @test_throws "KalmanFilterResult needs the per-period state covariances" KalmanFilterResult(lean)
    @test_throws "kalman_smoother! with covariances = true needs the per-period state covariances" kalman_smoother!(
        lean; covariances = true)
    @test_throws "crosscov=true requires covariances=true" kalman_smoother!(
        lean; crosscov = true)

    em_ws = Siphon.EMWorkspace(lean)
    @test_throws "em_estimate! needs the per-period state covariances" Siphon.em_estimate!(
        lean, em_ws, yfull; maxiter = 2, warn = false)
    @test_throws "compute_sufficient_stats! needs the per-period state covariances" Siphon.compute_sufficient_stats!(
        em_ws, lean, yfull)
end
