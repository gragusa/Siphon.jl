# The failure model: objective functions return -Inf, failed or unrun outputs refuse to
# return storage, shape errors are DimensionMismatch, and non-convergence warns.

using Siphon
using Test
using LinearAlgebra
using Random
using StaticArrays

struct FailTestLinearMeasurement{ZT <: AbstractMatrix} <: Siphon.AbstractEKFMeasurement
    Z::ZT
end
Siphon.measurement(m::FailTestLinearMeasurement, a, t) = m.Z * a
Siphon.measurement!(out, m::FailTestLinearMeasurement, a, t) = (mul!(out, m.Z, a); out)
Siphon.measurement_jacobian(m::FailTestLinearMeasurement, a, t) = m.Z
Siphon.measurement_jacobian!(Z, m::FailTestLinearMeasurement, a, t) = (copyto!(Z, m.Z); Z)

# A local level with a negative observation variance. With P1 = 0 and Q = 1 the
# predicted variance is t - 1, so after two missing periods F = 2 - 2.5 < 0 and the
# filter fails in period 3.
const fail_Z = [1.0;;]
const fail_H = [-2.5;;]
const fail_T = [1.0;;]
const fail_R = [1.0;;]
const fail_Q = [1.0;;]
const fail_a1 = [0.0]
const fail_P1 = [0.0;;]
fail_y(n = 10) = (y = randn(MersenneTwister(1), 1, n); y[1, 1:2] .= NaN; y)

const fail_state_accessors = (predicted_states, predicted_states_cov, filtered_states,
    filtered_states_cov, prediction_errors, prediction_errors_cov, kalman_gains,
    Siphon.missing_mask)

@testset "Failed functional filters" begin
    y = fail_y()
    p = KFParms(fail_Z, fail_H, fail_T, fail_R, fail_Q)

    r = kalman_filter(p, y, fail_a1, fail_P1)
    @test loglikelihood(r) == -Inf
    @test kalman_loglik(p, y, fail_a1, fail_P1) == -Inf
    for f in fail_state_accessors
        @test_throws "stopped in period 3" f(r)
    end
    @test_throws "stopped in period 3" Siphon.observed_mask(r)
    # Storage past the failure holds NaN, not uninitialised memory.
    @test all(isnan, r.at[:, 4:end])
    @test all(isnan, r.Ptt[:, :, 3:end])
    @test r.at[:, 1:3] == [0.0 0.0 0.0]
    @test system_matrices(r) === p

    # Nothing downstream runs on a failed result.
    @test_throws "stopped in period 3" kalman_smoother(r)
    @test_throws "stopped in period 3" kalman_smoother(p, y, fail_a1, fail_P1)
    @test_throws "stopped in period 3" predict(p, y, fail_a1, fail_P1)
    @test_throws "stopped in period 3" forecast(p, y, fail_a1, fail_P1, 2)

    # The static path, including its single-observation branch.
    sm(A) = SMatrix{size(A, 1), size(A, 2)}(A)
    ps = KFParms(sm(fail_Z), sm(fail_H), sm(fail_T), sm(fail_R), sm(fail_Q))
    rs = kalman_filter(ps, y, SVector{1}(fail_a1), SMatrix{1, 1}(fail_P1))
    @test loglikelihood(rs) == -Inf
    @test_throws "stopped in period 3" filtered_states(rs)
    @test kalman_loglik(ps, y, SVector{1}(fail_a1), SMatrix{1, 1}(fail_P1)) == -Inf

    # Several observables: the Cholesky factorisation of F fails.
    p2 = KFParms([1.0; 1.0;;], [-2.5 0.0; 0.0 1.0], fail_T, fail_R, fail_Q)
    y2 = vcat(y, y)
    r2 = kalman_filter(p2, y2, fail_a1, fail_P1)
    @test loglikelihood(r2) == -Inf
    @test_throws "stopped in period 3" prediction_errors(r2)

    # The scalar filter returns a failed result instead of a DomainError.
    psc = KFParms(1.0, -2.5, 1.0, 1.0, 1.0)
    rsc = kalman_filter(psc, vec(y), 0.0, 0.0)
    @test loglikelihood(rsc) == -Inf
    @test_throws "stopped in period 3" filtered_states(rsc)
    @test_throws "stopped in period 3" kalman_smoother(rsc)
    @test kalman_loglik(psc, vec(y), 0.0, 0.0) == -Inf

    # A successful result is unaffected.
    ok = kalman_filter(KFParms(fail_Z, [1.0;;], fail_T, fail_R, fail_Q), y, fail_a1, [1.0;;])
    @test isfinite(loglikelihood(ok))
    @test size(filtered_states(ok)) == (1, 10)
end

@testset "Failed exact diffuse filters" begin
    # Exact diffuse local level with H < 0: the diffuse step leaves Pstar = -1.5, so
    # F = Pstar + H < 0 in period 2.
    y = randn(MersenneTwister(2), 1, 8)
    p = KFParms(fail_Z, fail_H, fail_T, fail_R, fail_Q)
    r = kalman_filter(p, y, [0.0], [0.0;;], [1.0;;])
    @test r isa DiffuseFilterResult
    @test loglikelihood(r) == -Inf
    @test_throws "stopped in period 2" filtered_states(r)
    @test_throws "stopped in period 2" diffuse_flags(r)
    @test all(isnan, r.at[:, 3:end])
    @test kalman_loglik(p, y, [0.0], [0.0;;], [1.0;;]) == -Inf

    ws = DiffuseKalmanWorkspace(fail_Z, fail_H, fail_T, fail_R, fail_Q, [0.0], [0.0;;],
        [1.0;;], 8)
    @test_throws "kalman_filter!(ws, y)" filtered_states(ws)
    @test kalman_filter!(ws, y) == -Inf
    @test loglikelihood(ws) == -Inf
    @test_throws "stopped in period 2" filtered_states(ws)
    @test_throws "stopped in period 2" diffuse_flags(ws)
end

@testset "Workspaces before a run and after a failure" begin
    y = fail_y()
    ws = KalmanWorkspace(fail_Z, fail_H, fail_T, fail_R, fail_Q, fail_a1, fail_P1, 10)

    # Before any run: no storage, no log-likelihood.
    for f in fail_state_accessors
        @test_throws "Call kalman_filter!(ws, y) first" f(ws)
    end
    @test_throws "Call kalman_filter!(ws, y) first" loglikelihood(ws)
    @test_throws "Call kalman_filter!(ws, y) first" smoothed_states(ws)
    @test_throws "Call kalman_filter!(ws, y) first" kalman_smoother!(ws)
    @test_throws "Call kalman_filter!(ws, y) first" Siphon.observed_rows(ws, 1)
    # The system and the initial state are set, so they are available.
    @test obs_cov(ws) == fail_H
    @test initial_state(ws) == fail_a1

    # A failed run returns -Inf instead of throwing a PosDefException.
    @test kalman_filter!(ws, y) == -Inf
    @test loglikelihood(ws) == -Inf
    @test_throws "stopped in period 3" filtered_states(ws)
    @test_throws "stopped in period 3" kalman_smoother!(ws)
    @test kalman_filter_and_smooth!(ws, y) == -Inf
    @test_throws "stopped in period 3" smoothed_states(ws)

    # The same workspace recovers once the parameters are valid.
    set_params!(ws, fail_Z, [1.0;;], fail_T, fail_R, fail_Q)
    ll = kalman_filter!(ws, y)
    @test isfinite(ll)
    @test loglikelihood(ws) == ll
    @test size(filtered_states(ws)) == (1, 10)
    @test_throws "Call kalman_smoother!(ws) first" smoothed_states(ws)
    kalman_smoother!(ws)
    @test size(smoothed_states(ws)) == (1, 10)
    @test_throws "crosscov = true" crosslag_covs(ws)
    kalman_smoother!(ws; crosscov = true)
    @test size(crosslag_covs(ws), 3) == 9
    # A new filter run invalidates the smoother output.
    kalman_filter!(ws, y)
    @test_throws "Call kalman_smoother!(ws) first" smoothed_states_cov(ws)

    # A workspace built from dimensions alone holds zeros, not uninitialised memory.
    ws0 = KalmanWorkspace(2, 3, 1, 5)
    @test iszero(obs_matrix(ws0))
    @test iszero(initial_state_cov(ws0))
end

@testset "EKF failures" begin
    y = fail_y()
    meas = FailTestLinearMeasurement(fail_Z)
    p = EKFParms(meas, fail_H, fail_T, fail_R, fail_Q)
    r = kalman_filter(p, y, fail_a1, fail_P1)
    @test loglikelihood(r) == -Inf
    @test kalman_loglik(p, y, fail_a1, fail_P1) == -Inf
    @test_throws "stopped in period 3" filtered_states(r)
    @test_throws "stopped in period 3" predicted_observations(r)

    ws = EKFWorkspace(p, fail_a1, fail_P1, 10)
    @test_throws "Call kalman_filter!(ws, y) first" filtered_states(ws)
    @test kalman_filter!(ws, y) == -Inf
    @test_throws "stopped in period 3" filtered_states(ws)
    @test_throws "stopped in period 3" kalman_smoother!(ws)
    @test kalman_filter_and_smooth!(ws, y) == -Inf

    update_params!(ws; H = [1.0;;])
    @test isfinite(kalman_filter!(ws, y))
    @test_throws "Call kalman_smoother!(ws) first" smoothed_states(ws)
    kalman_smoother!(ws)
    @test size(smoothed_states(ws)) == (1, 10)

    @test_throws DimensionMismatch EKFWorkspace(meas, ADJacobian(), [1.0;;], fail_T,
        [1.0 0.0], fail_Q, fail_a1, fail_P1, 10)
    @test_throws DimensionMismatch kalman_filter!(ws, randn(2, 10))
    @test_throws DimensionMismatch set_initial!(ws, [0.0, 0.0], fail_P1)
end

@testset "Shape errors are DimensionMismatch" begin
    @test_throws "KFParms: H must be 1 × 1, got 2 × 2" KFParms(fail_Z, zeros(2, 2),
        fail_T, fail_R, fail_Q)
    @test_throws DimensionMismatch KFParms(fail_Z, fail_H, zeros(2, 2), fail_R, fail_Q)
    @test_throws DimensionMismatch KFParms(fail_Z, fail_H, fail_T, zeros(1, 2), fail_Q)
    @test_throws DimensionMismatch KalmanWorkspace(fail_Z, zeros(2, 2), fail_T, fail_R,
        fail_Q, fail_a1, fail_P1, 10)

    ws = KalmanWorkspace(2, 1, 1, 10)
    @test_throws "set_params!: Z must be 2 × 1, got 1 × 1" set_params!(ws, fail_Z,
        zeros(2, 2), fail_T, fail_R, fail_Q)
    @test_throws DimensionMismatch set_initial!(ws, [0.0, 0.0], fail_P1)
    @test_throws DimensionMismatch update_params!(ws; H = zeros(1, 1))
    @test_throws DimensionMismatch Siphon.set_Q!(ws, zeros(2, 2))
    @test_throws "kalman_filter!: y must be 2 × 10, got 1 × 10" kalman_filter!(ws,
        randn(1, 10))
    @test_throws DimensionMismatch kalman_filter!(ws, randn(2, 9))

    dws = DiffuseKalmanWorkspace(1, 1, 1, 10)
    @test_throws DimensionMismatch Siphon.set_initial_diffuse!(dws, [0.0], [1.0;;],
        zeros(2, 2))

    @test_throws DimensionMismatch diag_free([:a, :b]; init = [1.0, 2.0, 3.0])
end

@testset "Models: known parameters and failed filters" begin
    n = 30
    y = randn(MersenneTwister(3), 1, n)
    spec = local_level()

    model = StateSpaceModel(spec, (var_obs = 1.0, var_level = 0.5), n)
    @test isfitted(model)
    @test_throws "kalman_filter!(model, y)" loglikelihood(model)
    ll = kalman_filter!(model, y)
    @test loglikelihood(model) == ll
    fc = forecast(model, 3)
    fc_ref = forecast(spec, (var_obs = 1.0, var_level = 0.5), y, 3; static = :off)
    @test fc.yhat ≈ fc_ref.yhat && fc.F ≈ fc_ref.F
    @test forecast_periods(fc) == (n + 1):(n + 3)

    @test_throws DimensionMismatch kalman_filter!(model, randn(2, n))
    @test_throws DimensionMismatch kalman_loglik(model, randn(1, n - 1))
    @test_throws DimensionMismatch fit!(MLE(), StateSpaceModel(spec, n), randn(1, n - 1))
    @test_throws DimensionMismatch fit!(EM(), StateSpaceModel(spec, n), randn(2, n))

    # The default approximate diffuse P1 = 1e7 passes period 1 and leaves P₂ = -1.5.
    bad = StateSpaceModel(spec, (var_obs = -2.5, var_level = 1.0), n)
    @test kalman_filter!(bad, y) == -Inf
    @test loglikelihood(bad) == -Inf
    @test_throws "stopped in period 2" filtered_states(bad)
    @test_throws "stopped in period 2" smoothed_states(bad)
    @test_throws "stopped in period 2" forecast(bad, 2)

    @test_throws "diffuse must be true, false or :exact, got :foo" local_level(
        diffuse = :foo)
    @test_throws ArgumentError arma(1, 0; diffuse = :approx)
end

@testset "Non-convergence warns" begin
    rng = MersenneTwister(4)
    n = 60
    y = cumsum(randn(rng, 1, n); dims = 2) .+ randn(rng, 1, n)
    spec = local_level()

    @test_logs (:warn, r"EM did not converge in 2 iterations") fit!(EM(),
        StateSpaceModel(spec, n), y; maxiter = 2)
    @test_logs fit!(EM(), StateSpaceModel(spec, n), y; maxiter = 2, warn = false)
    m = fit!(EM(), StateSpaceModel(spec, n), y; maxiter = 2, warn = false)
    @test !isconverged(m)

    @test_logs (:warn, r"did not converge") optimize_ssm(spec, y; maxiters = 1)
    @test_logs optimize_ssm(spec, y; maxiters = 1, warn = false)
    @test_logs (:warn, r"did not converge") fit!(MLE(), StateSpaceModel(spec, n), y;
        maxiters = 1)

    # EM whose starting values make the filter fail stops at once and says why.
    kf_ws = KalmanWorkspace(fail_Z, fail_H, fail_T, fail_R, fail_Q, fail_a1, [1.0;;], n)
    em_ws = Siphon.EMWorkspace(kf_ws)
    res = @test_logs (:warn, r"EM stopped at iteration 1: the filter stopped in period 1") Siphon.em_estimate!(
        kf_ws, em_ws, y)
    @test !res.converged
    @test res.loglik == -Inf
    @test res.iterations == 1

    dfm_y = randn(rng, 4, 80)
    @test_logs (:warn, r"EM did not converge") match_mode=:any fit!(EM(),
        DynamicFactorModel(4, 1, 80), dfm_y; maxiter = 2)
end
