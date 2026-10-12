# One smoother result type, one cross-covariance keyword, and conversions between the
# functional results and the in-place workspaces.

using Siphon
using Test
using LinearAlgebra
using Random
using ForwardDiff

@testset "Smoother interface: functional and in-place agree" begin
    rng = MersenneTwister(5)
    p_obs, m, n = 2, 2, 30
    Z = [1.0 0.0; 0.5 1.0]
    H = [0.4 0.1; 0.1 0.7]
    Tm = [0.8 0.1; 0.0 0.6]
    R = Matrix(1.0I, m, m)
    Q = [0.3 0.05; 0.05 0.2]
    a1 = [0.1, -0.2]
    P1 = [2.0 0.3; 0.3 1.5]
    y = randn(rng, p_obs, n)
    y[1, 7] = NaN
    y[:, 12] .= NaN
    kf = KFParms(Z, H, Tm, R, Q)

    filt = kalman_filter(kf, y, a1, P1)
    sm = kalman_smoother(filt)
    ws = KalmanWorkspace(Z, H, Tm, R, Q, a1, P1, n)
    ll = kalman_filter_and_smooth!(ws, y)

    # Neither path computes the cross-lag covariances by default
    @test sm.Pcross === nothing
    @test_throws "crosscov = true" crosslag_covs(sm)
    @test_throws "crosscov = true" crosslag_covs(ws)

    @test ll ≈ loglikelihood(filt)
    @test smoothed_states(ws) ≈ smoothed_states(sm)
    @test smoothed_states_cov(ws) ≈ smoothed_states_cov(sm)

    kalman_filter_and_smooth!(ws, y; crosscov = true)
    sm_c = kalman_smoother(filt; crosscov = true)
    @test crosslag_covs(ws) ≈ crosslag_covs(sm_c)

    # The combined functional call returns both results
    filt2, sm2 = kalman_filter_and_smooth(kf, y, a1, P1)
    @test filt2 isa KalmanFilterResult
    @test loglikelihood(filt2) == loglikelihood(filt)
    @test smoothed_states(sm2) == smoothed_states(sm)
end

@testset "KalmanFilterResult(ws) copies the filter output" begin
    rng = MersenneTwister(6)
    n = 25
    Z, H, Tm, R, Q = [1.0;;], [2.0;;], [0.9;;], [1.0;;], [0.5;;]
    a1, P1 = [0.0], [5.0;;]
    y = randn(rng, 1, n)
    y[1, [4, 5]] .= NaN

    ws = KalmanWorkspace(Z, H, Tm, R, Q, a1, P1, n)
    @test_throws "kalman_filter!(ws, y)" KalmanFilterResult(ws)

    kalman_filter!(ws, y)
    r = KalmanFilterResult(ws)
    ref = kalman_filter(KFParms(Z, H, Tm, R, Q), y, a1, P1)
    @test r isa KalmanFilterResult
    @test loglikelihood(r) ≈ loglikelihood(ref)
    @test filtered_states(r) ≈ filtered_states(ref)
    @test predicted_states_cov(r) ≈ predicted_states_cov(ref)
    @test isapprox(prediction_errors(r), prediction_errors(ref); nans = true)
    @test Siphon.missing_mask(r) == Siphon.missing_mask(ref)
    @test system_matrices(r).T == Tm

    # The result feeds the functional smoother
    kalman_smoother!(ws)
    @test smoothed_states(kalman_smoother(r)) ≈ smoothed_states(ws)

    # Later runs on the workspace leave the result unchanged
    att = copy(filtered_states(r))
    update_params!(ws; T = [0.5;;])
    kalman_filter!(ws, randn(rng, 1, n))
    @test filtered_states(r) == att
    @test transition_matrix(r) == Tm

    # A failed run converts to a failed result
    set_params!(ws, Z, [-10.0;;], Tm, R, Q)
    kalman_filter!(ws, y)
    rf = KalmanFilterResult(ws)
    @test loglikelihood(rf) == -Inf
    @test_throws "stopped in period" filtered_states(rf)
    @test_throws "stopped in period" kalman_smoother(rf)
end

@testset "SmootherResult(ws) and SmootherResult(model) copy" begin
    rng = MersenneTwister(7)
    n = 20
    ws = KalmanWorkspace([1.0;;], [2.0;;], [0.9;;], [1.0;;], [0.5;;], [0.0], [5.0;;], n)
    y = randn(rng, 1, n)

    kalman_filter!(ws, y)
    @test_throws "kalman_smoother!(ws)" SmootherResult(ws)

    kalman_smoother!(ws)
    sr = SmootherResult(ws)
    @test sr.Pcross === nothing
    αs = copy(smoothed_states(ws))
    @test smoothed_states(sr) == αs
    @test smoothed_states(sr) !== smoothed_states(ws)

    kalman_filter_and_smooth!(ws, randn(rng, 1, n); crosscov = true)
    @test smoothed_states(sr) == αs
    @test crosslag_covs(SmootherResult(ws)) == crosslag_covs(ws)

    spec = local_level()
    model = StateSpaceModel(spec, (var_obs = 2.0, var_level = 0.5), n)
    kalman_filter!(model, y)
    srm = SmootherResult(model)
    @test smoothed_states(srm) == smoothed_states(model)
    @test smoothed_states(srm) !== smoothed_states(model)
end

@testset "kalman_smoother on an exact diffuse result" begin
    y = randn(MersenneTwister(8), 1, 15)
    p = KFParms([1.0;;], [1.0;;], [1.0;;], [1.0;;], [0.5;;])
    r = kalman_filter(p, y, [0.0], [0.0;;], [1.0;;])
    @test r isa DiffuseFilterResult
    @test_throws "exact diffuse" kalman_smoother(r)
end

@testset "kalman_filter_and_smooth is AD-compatible" begin
    y = randn(MersenneTwister(9), 1, 20)
    function f(θ)
        p = KFParms([1.0;;], [θ[1];;], [1.0;;], [1.0;;], [θ[2];;])
        filt, sm = kalman_filter_and_smooth(p, y, [0.0], [10.0;;]; crosscov = true)
        return loglikelihood(filt) + sum(smoothed_states(sm)) + sum(crosslag_covs(sm))
    end
    g = ForwardDiff.gradient(f, [1.0, 0.5])
    @test all(isfinite, g)
end
