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
        pr = predict(p_cov, y_cov, a1_cov, P1_cov)
        filt = kalman_filter(p_cov, y_cov, a1_cov, P1_cov)
        n = size(y_cov, 2)
        @test size(pr.yhat) == (1, n)
        @test pr.yhat ≈ p_cov.Z * filt.at
        @test pr.at == filt.at
        @test pr.Ptt == filt.Ptt
        @test pr.loglik ≈ kalman_loglik(p_cov, y_cov, a1_cov, P1_cov)
    end

    @testset "predict (SSMSpec)" begin
        spec = local_level()
        θ = (var_obs = 15099.0, var_level = 1469.1)
        pr = predict(spec, θ, y_cov)
        ss = build_linear_state_space(spec, θ, y_cov)
        @test pr.loglik ≈ kalman_loglik(ss.p, y_cov, ss.a1, ss.P1)
        @test size(pr.yhat) == size(y_cov)
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
        Random.seed!(42)
        paths = forecast_paths(p_cov, y_cov, a1_cov, P1_cov, h, n_paths)
        @test size(paths) == (1, h, n_paths)
        fc = forecast(p_cov, y_cov, a1_cov, P1_cov, h)
        # Simulated paths are centred on the point forecast with variance F
        m = vec(sum(paths; dims = 3)) ./ n_paths
        @test all(abs.(m .- vec(fc.yhat)) .< 4 .* sqrt.(fc.F[1, 1, :] ./ n_paths))
        v = [var(paths[1, j, :]) for j in 1:h]
        @test all(abs.(v ./ fc.F[1, 1, :] .- 1) .< 0.1)
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

        y = [1.0 NaN 3.0 NaN; 4.0 5.0 6.0 NaN]
        # count_missing counts periods with any missing element
        @test count_missing(y) == 2
        @test ismissing_obs(y[:, 2])
        @test !ismissing_obs(y[:, 1])
        @test ismissing_obs(NaN)
        @test !ismissing_obs(1.0)
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

        # The transition matrix is updated with T, as documented, or the alias Tmat
        update_params!(ws; T = [0.9;;])
        @test ws.Tmat == [0.9;;]
        update_params!(ws; Tmat = [1.0;;])
        @test ws.Tmat == [1.0;;]
        @test_throws ArgumentError update_params!(ws; T = [0.9;;], Tmat = [0.9;;])
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
