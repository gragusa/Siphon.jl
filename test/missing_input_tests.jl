# Observations with `missing` entries, and innovations reported in series positions.

using Siphon
using Test
using LinearAlgebra
using Random
using StaticArrays

@testset "missing entries are accepted wherever NaN is" begin
    rng = MersenneTwister(21)
    p_obs, m, n = 2, 2, 30
    Z = [1.0 0.0; 0.5 1.0]
    H = [0.4 0.1; 0.1 0.7]
    Tm = [0.8 0.1; 0.0 0.6]
    R = Matrix(1.0I, m, m)
    Q = [0.3 0.05; 0.05 0.2]
    a1 = zeros(m)
    P1 = Matrix(1.0I, m, m)
    y = randn(rng, p_obs, n)
    y[1, 4] = NaN
    y[:, 9] .= NaN
    ym = nan_to_missing(y)
    @test ym isa Matrix{Union{Missing, Float64}}
    p = KFParms(Z, H, Tm, R, Q)

    ll = kalman_loglik(p, y, a1, P1)
    @test kalman_loglik(p, ym, a1, P1) == ll
    @test isequal(filtered_states(kalman_filter(p, ym, a1, P1)),
        filtered_states(kalman_filter(p, y, a1, P1)))
    ps = KFParms_static(Z, H, Tm, R, Q)
    @test kalman_loglik(ps, ym, SVector{2}(a1), SMatrix{2, 2}(P1)) ≈ ll
    @test kalman_loglik(p, ym, a1, zeros(m, m), Matrix(1.0I, m, m)) ==
          kalman_loglik(p, y, a1, zeros(m, m), Matrix(1.0I, m, m))
    @test isequal(smoothed_states(kalman_smoother(p, ym, a1, P1)),
        smoothed_states(kalman_smoother(p, y, a1, P1)))
    @test isequal(predict(p, ym, a1, P1), predict(p, y, a1, P1))
    @test forecast(p, ym, a1, P1, 3).yhat == forecast(p, y, a1, P1, 3).yhat

    ws = KalmanWorkspace(Z, H, Tm, R, Q, a1, P1, n)
    @test kalman_filter!(ws, ym) == kalman_filter!(ws, y)
    @test kalman_filter_and_smooth!(ws, ym) ≈ ll
    @test simulation_smoother(ws, ym; rng = MersenneTwister(1)) ==
          simulation_smoother(ws, y; rng = MersenneTwister(1))

    # The scalar path takes a vector.
    yv = vec(y[1, :])
    ps = KFParms(1.0, 1.0, 0.9, 1.0, 0.5)
    @test kalman_loglik(ps, nan_to_missing(yv), 0.0, 1.0) == kalman_loglik(ps, yv, 0.0, 1.0)
end

@testset "missing entries in the model and estimation layer" begin
    rng = MersenneTwister(22)
    n = 60
    y = cumsum(randn(rng, 1, n); dims = 2) .+ randn(rng, 1, n)
    y[1, [5, 17, 40]] .= NaN
    ym = nan_to_missing(y)
    spec = local_level()
    θ = (var_obs = 1.0, var_level = 0.5)

    @test ssm_loglik(spec, θ, ym) == ssm_loglik(spec, θ, y)
    @test isequal(predict(spec, θ, ym), predict(spec, θ, y))
    θ0 = (var_obs = 0.8, var_level = 0.4)
    @test optimize_ssm(spec, ym; θ0).loglik ≈ optimize_ssm(spec, y; θ0).loglik

    mle = fit!(MLE(), StateSpaceModel(spec, n), ym; θ0)
    @test loglikelihood(mle) ≈ loglikelihood(fit!(MLE(), StateSpaceModel(spec, n), y; θ0))
    em = fit!(EM(), StateSpaceModel(spec, n), ym; maxiter = 20, warn = false)
    @test loglikelihood(em) ≈
          loglikelihood(fit!(EM(), StateSpaceModel(spec, n), y; maxiter = 20, warn = false))

    model = StateSpaceModel(spec, θ, n)
    @test kalman_filter!(model, ym) == kalman_filter!(model, y)
    @test kalman_loglik(model, ym) == kalman_loglik(model, y)
end

@testset "innovations are reported in series positions" begin
    rng = MersenneTwister(23)
    p_obs, m, n = 3, 2, 12
    Z = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    H = Matrix(Diagonal([0.5, 0.8, 0.3]))
    Tm = [0.7 0.1; 0.0 0.5]
    R = Matrix(1.0I, m, m)
    Q = [0.4 0.0; 0.0 0.3]
    a1 = zeros(m)
    P1 = Matrix(1.0I, m, m)
    p = KFParms(Z, H, Tm, R, Q)
    y = randn(rng, p_obs, n)
    y[2, 3] = NaN             # rows 1 and 3 observed
    y[[1, 3], 6] .= NaN       # only row 2 observed
    y[:, 8] .= NaN            # nothing observed

    r = kalman_filter(p, y, a1, P1)
    v = prediction_errors(r)
    F = prediction_errors_cov(r)
    K = kalman_gains(r)
    at, Pt = predicted_states(r), predicted_states_cov(r)

    for t in 1:n
        obs = findall(!isnan, y[:, t])
        miss = findall(isnan, y[:, t])
        Ft = Z * Pt[:, :, t] * Z' + H
        @test v[obs, t] ≈ y[obs, t] - Z[obs, :] * at[:, t]
        @test all(isnan, v[miss, t])
        @test F[obs, obs, t] ≈ Ft[obs, obs]
        @test all(isnan, F[miss, :, t]) && all(isnan, F[:, miss, t])
        if !isempty(obs)
            Kt = Tm * Pt[:, :, t] * Z[obs, :]' / Ft[obs, obs]
            @test K[:, obs, t] ≈ Kt
        end
        @test iszero(K[:, miss, t])
    end

    # The accessors return copies; the stored layout is unchanged.
    v[1, 1] = 99.0
    @test prediction_errors(r)[1, 1] != 99.0

    # Workspaces, diffuse filters and models agree with the functional result.
    ws = KalmanWorkspace(Z, H, Tm, R, Q, a1, P1, n)
    kalman_filter!(ws, y)
    @test isapprox(prediction_errors(ws), prediction_errors(r); nans = true)
    @test prediction_errors_cov(ws)[:, :, [1, 3, 6]] ≈ F[:, :, [1, 3, 6]] nans = true
    @test kalman_gains(ws) ≈ K

    spec = custom_ssm(Z = Z, H = H, T = Tm, R = R, Q = Q, a1 = a1, P1 = P1)
    model = StateSpaceModel(spec, NamedTuple(), n)
    kalman_filter!(model, y)
    @test isapprox(prediction_errors(model), prediction_errors(r); nans = true)
    @test kalman_gains(model) ≈ K

    rd = kalman_filter(p, y, a1, zeros(m, m), Matrix(1.0I, m, m))
    vd = prediction_errors(rd)
    @test isnan(vd[2, 3]) && !isnan(vd[3, 3])
    @test all(isnan, vd[:, 8])
end

struct MissingTestLinearMeasurement{ZT <: AbstractMatrix} <: Siphon.AbstractEKFMeasurement
    Z::ZT
end
Siphon.measurement(m::MissingTestLinearMeasurement, a, t) = m.Z * a
Siphon.measurement!(out, m::MissingTestLinearMeasurement, a, t) = (mul!(out, m.Z, a); out)
Siphon.measurement_jacobian(m::MissingTestLinearMeasurement, a, t) = m.Z
function Siphon.measurement_jacobian!(Z, m::MissingTestLinearMeasurement, a, t)
    (copyto!(Z, m.Z); Z)
end

@testset "every filter output has the missing-data vocabulary" begin
    y = randn(MersenneTwister(24), 2, 10)
    y[:, 4] .= NaN
    ym = nan_to_missing(y)
    mask = trues(2, 10)
    mask[:, 4] .= false
    Z = [1.0 0.0; 0.0 1.0]
    p = EKFParms(MissingTestLinearMeasurement(Z), Matrix(1.0I, 2, 2), 0.5 * Z, Z, Z)
    a1, P1 = zeros(2), Matrix(1.0I, 2, 2)

    r = kalman_filter(p, ym, a1, P1)
    @test kalman_loglik(p, ym, a1, P1) == kalman_loglik(p, y, a1, P1) == loglikelihood(r)
    @test Siphon.observed_mask(r) == mask
    @test Siphon.missing_mask(r) == .!mask[1, :]
    ws = EKFWorkspace(p, a1, P1, 10)
    @test kalman_filter!(ws, ym) ≈ loglikelihood(r)
    @test Siphon.observed_mask(ws) == mask
    @test isapprox(prediction_errors(ws), prediction_errors(r); nans = true)

    rs = kalman_filter(KFParms(1.0, 1.0, 0.5, 1.0, 1.0), nan_to_missing(y[1, :]), 0.0, 1.0)
    @test Siphon.observed_mask(rs) == mask[1:1, :]
end
