"""
Tests for exported functions that previously had no direct coverage:
prediction and forecasting, missing-data utilities, static-array helpers,
in-place parameter updates and the StateSpaceModel status/covariance accessors.
"""

using Test
using Siphon
using LinearAlgebra
using Random
using Statistics
using StaticArrays
using CSV
using DataFrames

let nile = CSV.read(joinpath(@__DIR__, "Nile.csv"), DataFrame; header = false)
    global y_cov = reshape(Float64.(nile[!, 1]), 1, :)
end

const p_cov = KFParms([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.1;;])
const a1_cov = [0.0]
const P1_cov = [1e7;;]

@testset "API coverage" begin

@testset "predict (KFParms)" begin
    yhat = predict(p_cov, y_cov, a1_cov, P1_cov)
    filt = kalman_filter(p_cov, y_cov, a1_cov, P1_cov)
    n = size(y_cov, 2)
    @test size(yhat) == (1, n)
    @test yhat ≈ p_cov.Z * filt.at
    @test predicted_observations(filt) == yhat
end

@testset "predict (SSMSpec)" begin
    spec = local_level()
    θ = (var_obs = 15099.0, var_level = 1469.1)
    yhat = predict(spec, θ, y_cov)
    ss = build_linear_state_space(spec, θ, y_cov)
    @test yhat ≈ predict(ss.p, y_cov, ss.a1, ss.P1)
    @test size(yhat) == size(y_cov)
end

@testset "forecast (KFParms)" begin
    h = 5
    fc = forecast(p_cov, y_cov, a1_cov, P1_cov, h)
    filt = kalman_filter(p_cov, y_cov, a1_cov, P1_cov)
    n = size(y_cov, 2)
    @test size(fc.yhat) == (1, h)
    @test size(fc.a) == (1, h)
    @test size(fc.P) == (1, 1, h)
    @test size(fc.F) == (1, 1, h)
    # Local level: the point forecast is flat at the last filtered level,
    # and the state variance grows by Q each step.
    @test all(fc.yhat .≈ filt.att[1, n])
    @test fc.P[1, 1, 1] ≈ filt.Ptt[1, 1, n] + 1469.1
    @test diff(vec(fc.P)) ≈ fill(1469.1, h - 1)
    @test fc.F[1, 1, :] ≈ fc.P[1, 1, :] .+ 15099.0
    @test fc isa ForecastResult{Float64}
    @test forecast_periods(fc) == (n + 1):(n + h)
    @test forecast_observations(fc) === fc.yhat
    @test forecast_observations_cov(fc) === fc.F
    @test forecast_states(fc) === fc.a
    @test forecast_states_cov(fc) === fc.P
end

@testset "forecast from a filter result" begin
    h = 4
    filt = kalman_filter(p_cov, y_cov, a1_cov, P1_cov)
    fc = forecast(filt, h)
    fc2 = forecast(p_cov, y_cov, a1_cov, P1_cov, h)
    @test fc.yhat == fc2.yhat && fc.F == fc2.F && fc.a == fc2.a && fc.P == fc2.P
    @test_throws ArgumentError forecast(filt, 0)

    # A scalar system forecasts like its 1 × 1 matrix version
    ps = KFParms(1.0, 15099.0, 1.0, 1.0, 1469.1)
    fcs = forecast(kalman_filter(ps, vec(y_cov), 0.0, 1e7), h)
    @test fcs.yhat ≈ fc.yhat && fcs.F ≈ fc.F && fcs.P ≈ fc.P

    # Exact diffuse: forecasts start after the sample once the diffuse period ends
    dfilt = kalman_filter(p_cov, y_cov, a1_cov, zeros(1, 1), ones(1, 1))
    dfc = forecast(dfilt, h)
    n = size(y_cov, 2)
    @test dfc.yhat ≈ fill(dfilt.att[1, n], 1, h)
    @test forecast_periods(dfc) == (n + 1):(n + h)
    @test_throws "diffuse period" forecast(
        kalman_filter(p_cov, y_cov[:, 1:1] .* NaN, a1_cov, zeros(1, 1), ones(1, 1)), 1)
end

@testset "forecast_interval" begin
    fc = forecast(p_cov, y_cov, a1_cov, P1_cov, 3)
    lo, hi = forecast_interval(fc, 0.9)
    z = Siphon.quantile_normal(0.95)
    @test lo ≈ fc.yhat .- z .* sqrt.(fc.F[1, 1, :])'
    @test hi ≈ fc.yhat .+ z .* sqrt.(fc.F[1, 1, :])'
    slo, shi = forecast_interval(fc; what = :states)
    @test size(slo) == size(fc.a) && all(slo .< fc.a .< shi)
    @test_throws ArgumentError forecast_interval(fc; what = :factors)
end

@testset "forecast (SSMSpec)" begin
    spec = local_level()
    θ = (var_obs = 15099.0, var_level = 1469.1)
    fc = forecast(spec, θ, y_cov, 3)
    ss = build_linear_state_space(spec, θ, y_cov)
    fc2 = forecast(ss.p, y_cov, ss.a1, ss.P1, 3)
    @test fc.yhat ≈ fc2.yhat
    @test fc.F ≈ fc2.F
end

@testset "forecast_paths" begin
    h, n_paths = 4, 4000
    paths = forecast_paths(p_cov, y_cov, a1_cov, P1_cov, h, n_paths; rng = Xoshiro(42))
    @test size(paths) == (1, h, n_paths)
    fc = forecast(p_cov, y_cov, a1_cov, P1_cov, h)
    # Simulated paths are centred on the point forecast with variance F
    m = vec(sum(paths; dims = 3)) ./ n_paths
    @test all(abs.(m .- vec(fc.yhat)) .< 4 .* sqrt.(fc.F[1, 1, :] ./ n_paths))
    v = [var(paths[1, j, :]) for j in 1:h]
    @test all(abs.(v ./ fc.F[1, 1, :] .- 1) .< 0.1)

    # The rng keyword makes draws reproducible, from the filter result too
    filt = kalman_filter(p_cov, y_cov, a1_cov, P1_cov)
    @test forecast_paths(filt, h, 5; rng = Xoshiro(1)) ==
          forecast_paths(p_cov, y_cov, a1_cov, P1_cov, h, 5; rng = Xoshiro(1))

    # Singular covariances (H = 0, rank-deficient Q) are allowed
    p0 = KFParms([1.0 0.0], zeros(1, 1), [1.0 1.0; 0.0 1.0], [0.0; 1.0;;], [1.0;;])
    paths0 = forecast_paths(p0, y_cov, zeros(2), 1e7 * Matrix(1.0I, 2, 2), 3, 10;
        rng = Xoshiro(2))
    @test size(paths0) == (1, 3, 10) && all(isfinite, paths0)

    # The element type follows the system
    p32 = KFParms(Float32[1;;], Float32[15099;;], Float32[1;;], Float32[1;;], Float32[1469;;])
    @test eltype(forecast_paths(p32, Float32.(y_cov), Float32[0], Float32[1e7;;], 2, 3;
        rng = Xoshiro(3))) == Float32
end

@testset "missing-data utilities" begin
    ym = [1.0 missing 3.0; 4.0 5.0 missing]
    yn = missing_to_nan(ym)
    @test yn isa Matrix{Float64}
    @test isnan(yn[1, 2]) && isnan(yn[2, 3])
    @test yn[1, 1] == 1.0
    back = nan_to_missing(yn)
    @test ismissing(back[1, 2]) && ismissing(back[2, 3])
    @test isequal(back, ym)

    # Element types are kept rather than forced to Float64
    @test missing_to_nan(Union{Missing, Float32}[1.0f0 missing]) isa Matrix{Float32}
    @test missing_to_nan([1 2; 3 4]) isa Matrix{Float64}
    @test missing_to_nan(fill(missing, 2, 2)) isa Matrix{Float64}
    @test nan_to_missing(Float32[1 NaN]) isa Matrix{Union{Missing, Float32}}

    y = [1.0 NaN 3.0 NaN; 4.0 5.0 6.0 NaN]
    # "missing" periods have no value; "incomplete" periods lack at least one
    @test count_missing_periods(y) == 1
    @test count_incomplete_periods(y) == 2
    @test ismissing_period(y[:, 4]) && !ismissing_period(y[:, 2])
    @test isincomplete_period(y[:, 2]) && !isincomplete_period(y[:, 1])
    @test ismissing_period([missing, NaN]) && isincomplete_period([1.0, missing])
    r = kalman_filter(KFParms(ones(2, 1), Matrix(1.0I, 2, 2), [1.0;;], [1.0;;], [1.0;;]),
        y, [0.0], [1.0;;])
    @test count(Siphon.missing_mask(r)) == count_missing_periods(y)
end

@testset "static-array helpers" begin
    ps = KFParms_static([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.1;;])
    @test ps.Z isa SMatrix
    @test ps.Q isa SMatrix
    ll_s = kalman_loglik(ps, y_cov, SVector{1}(a1_cov), SMatrix{1, 1}(P1_cov))
    @test ll_s ≈ kalman_loglik(p_cov, y_cov, a1_cov, P1_cov)

    @test to_static_if_small(ones(2, 2)) isa SMatrix{2, 2}
    @test to_static_if_small(ones(3)) isa SVector{3}
    big = ones(STATIC_THRESHOLD + 1, STATIC_THRESHOLD + 1)
    @test to_static_if_small(big) isa Matrix
end

@testset "update_params!" begin
    n = size(y_cov, 2)
    ws = KalmanWorkspace(p_cov.Z, p_cov.H, p_cov.T, p_cov.R, p_cov.Q,
        a1_cov, P1_cov, n)
    ll0 = kalman_filter!(ws, y_cov)
    @test ll0 ≈ kalman_loglik(p_cov, y_cov, a1_cov, P1_cov)

    update_params!(ws; H = [10000.0;;], Q = [2000.0;;])
    ll1 = kalman_filter!(ws, y_cov)
    p1 = KFParms([1.0;;], [10000.0;;], [1.0;;], [1.0;;], [2000.0;;])
    @test ll1 ≈ kalman_loglik(p1, y_cov, a1_cov, P1_cov)

    # Updating a subset keeps the other matrices
    update_params!(ws; Q = [1469.1;;])
    p2 = KFParms([1.0;;], [10000.0;;], [1.0;;], [1.0;;], [1469.1;;])
    @test kalman_filter!(ws, y_cov) ≈ kalman_loglik(p2, y_cov, a1_cov, P1_cov)

    # The transition matrix is updated with T
    update_params!(ws; T = [0.9;;])
    @test transition_matrix(ws) == [0.9;;]
    @test_throws MethodError update_params!(ws; Tmat = [1.0;;])
end

@testset "StateSpaceModel status and covariance accessors" begin
    spec = local_level()
    n = size(y_cov, 2)
    model = StateSpaceModel(spec, n)
    @test !isfitted(model)
    @test_throws ArgumentError loglikelihood(model)

    fit!(MLE(), model, y_cov)
    @test isfitted(model)
    @test isconverged(model) isa Bool
    @test niterations(model) isa Integer

    θ = parameters(model)
    ss = build_linear_state_space(spec, θ, y_cov)
    filt = kalman_filter(ss.p, y_cov, ss.a1, ss.P1)
    @test predicted_states_cov(model) ≈ filt.Pt
    @test filtered_states_cov(model) ≈ filt.Ptt
    @test prediction_errors_cov(model) ≈ filt.Ft
    @test size(smoothed_states_cov(model)) == (1, 1, n)
end

end # API coverage
