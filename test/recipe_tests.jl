"""
Tests for RecipesBase plotting recipes.
"""

using Siphon
using Siphon: quantile_normal, select_vars, confidence_bands, SmootherResult,
    ForecastResult, ObservablePlot
using RecipesBase
using LinearAlgebra
using CSV
using DataFrames

@testset "Plotting recipes" begin

@testset "quantile_normal" begin

    # Standard values
    @test quantile_normal(0.975) ≈ 1.96 atol=0.01
    @test quantile_normal(0.95) ≈ 1.645 atol=0.01
    @test quantile_normal(0.90) ≈ 1.28 atol=0.01
    @test quantile_normal(0.5 + eps()) ≈ 0.0 atol=0.01

    # Exact on both sides of the median
    @test quantile_normal(0.975) ≈ 1.959963984540054
    @test quantile_normal(0.025) ≈ -1.959963984540054
    @test quantile_normal(0.5) == 0.0
    @test quantile_normal(1e-10) ≈ -6.361340902404056
    @test_throws DomainError quantile_normal(0.0)
    @test_throws DomainError quantile_normal(1.0)
end

@testset "select_vars" begin

    # :all and nothing
    @test select_vars(:all, 5) == 1:5
    @test select_vars(nothing, 5) == 1:5

    # Single integer
    @test select_vars(3, 5) == [3]
    @test select_vars(1, 5) == [1]

    # Vector of integers
    @test select_vars([1, 3], 5) == [1, 3]
    @test select_vars([2, 4, 5], 5) == [2, 4, 5]

    # Range
    @test select_vars(2:4, 5) == 2:4
    @test select_vars(1:5, 5) == 1:5

    # Error cases
    @test_throws ArgumentError select_vars(6, 5)
    @test_throws ArgumentError select_vars([1, 6], 5)
    @test_throws ArgumentError select_vars(3:7, 5)
    @test_throws ArgumentError select_vars("invalid", 5)
end

@testset "confidence_bands validates and accepts mixed array types" begin
    @test_throws DomainError confidence_bands([0.0], [1.0], 95)
    @test_throws DomainError confidence_bands(zeros(1, 2), ones(1, 1, 2), 1.0)
    @test_throws DimensionMismatch confidence_bands(zeros(2, 3), ones(2, 2, 4))

    # Float64 means with Float32 covariances, and a view as the mean
    lower, upper = confidence_bands(view(zeros(2, 4), :, 1:3), ones(Float32, 2, 2, 3), 0.9)
    @test upper ≈ fill(quantile_normal(0.95), 2, 3)
    @test lower ≈ -upper
end

@testset "confidence_bands vector" begin

    mean = [1.0, 2.0, 3.0]
    var = [1.0, 4.0, 9.0]  # std = [1, 2, 3]

    lower, upper = confidence_bands(mean, var, 0.95)

    # 95% CI: mean ± 1.96 * std
    @test length(lower) == 3
    @test length(upper) == 3
    @test all(lower .< mean .< upper)

    # Check approximate values
    @test lower[1] ≈ 1.0 - 1.96 * 1.0 atol=0.05
    @test upper[1] ≈ 1.0 + 1.96 * 1.0 atol=0.05
    @test lower[2] ≈ 2.0 - 1.96 * 2.0 atol=0.05
    @test upper[2] ≈ 2.0 + 1.96 * 2.0 atol=0.05

    # 90% CI should be narrower
    lower90, upper90 = confidence_bands(mean, var, 0.90)
    @test all(lower90 .> lower)
    @test all(upper90 .< upper)
end

@testset "confidence_bands matrix" begin

    m, n = 2, 10
    mean = rand(m, n)
    cov = zeros(m, m, n)
    for t in 1:n
        cov[1, 1, t] = 1.0
        cov[2, 2, t] = 4.0
    end

    lower, upper = confidence_bands(mean, cov, 0.95)

    @test size(lower) == (m, n)
    @test size(upper) == (m, n)
    @test all(lower .< mean .< upper)
end

@testset "SmootherResult accessors" begin
    m, n = 2, 10
    αs, Vs, Pcross = rand(m, n), rand(m, m, n), rand(m, m, n - 1)
    sr = SmootherResult(αs, Vs, nothing)

    @test smoothed_states(sr) === αs
    @test smoothed_states_cov(sr) === Vs
    @test_throws "crosscov = true" crosslag_covs(sr)
    @test crosslag_covs(SmootherResult(αs, Vs, Pcross)) === Pcross
end

@testset "ForecastResult accessors" begin

    p_obs, m_state, h = 3, 2, 12
    fr = ForecastResult(rand(p_obs, h), rand(p_obs, p_obs, h), rand(m_state, h),
        rand(m_state, m_state, h), 100)

    @test forecast_observations(fr) == fr.yhat
    @test forecast_observations_cov(fr) == fr.F
    @test forecast_states(fr) == fr.a
    @test forecast_states_cov(fr) == fr.P
    @test forecast_periods(fr) == 101:112
end

@testset "kalman_smoother returns a SmootherResult" begin

    # Load Nile data
    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = CSV.read(nile_path, DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)

    # Local level model parameters
    p = KFParms([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.0;;])
    a1 = [0.0]
    P1 = [1e7;;]

    # One call runs filter + smoother
    sr = kalman_smoother(p, y, a1, P1)

    @test sr isa SmootherResult
    @test size(smoothed_states(sr)) == (1, size(y, 2))
    @test size(smoothed_states_cov(sr)) == (1, 1, size(y, 2))
end

@testset "KalmanFilterResult recipe produces series" begin

    # Load Nile data
    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = CSV.read(nile_path, DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)

    # Local level model
    p = KFParms([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.0;;])
    a1 = [0.0]
    P1 = [1e7;;]

    result = kalman_filter(p, y, a1, P1)

    # Recipe should produce output
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), result)
    @test !isempty(rec)

    # With options
    rec2 = RecipesBase.apply_recipe(Dict{Symbol, Any}(:vars => 1, :level => 0.90), result)
    @test !isempty(rec2)

    # Predicted instead of filtered
    rec3 = RecipesBase.apply_recipe(Dict{Symbol, Any}(:filtered => false), result)
    @test !isempty(rec3)
end

@testset "SmootherResult recipe produces series" begin

    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = CSV.read(nile_path, DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)

    p = KFParms([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.0;;])
    a1 = [0.0]
    P1 = [1e7;;]

    sr = kalman_smoother(p, y, a1, P1)

    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), sr)
    @test !isempty(rec)
    rec2 = RecipesBase.apply_recipe(Dict{Symbol, Any}(:time => 1871:1970), sr)
    @test !isempty(rec2)
end

@testset "ForecastResult recipe produces series" begin

    # Create mock forecast result
    h = 12
    fr = ForecastResult(rand(1, h), rand(1, 1, h), rand(1, h), rand(1, 1, h), 100)

    # Observations (default)
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), fr)
    @test !isempty(rec)

    # States
    rec2 = RecipesBase.apply_recipe(Dict{Symbol, Any}(:what => :states), fr)
    @test !isempty(rec2)

    # The x axis is the forecast periods unless `time` is given
    @test rec[end].args[1] == 101:112
    rec3 = RecipesBase.apply_recipe(Dict{Symbol, Any}(:time => 1:12), fr)
    @test rec3[end].args[1] == 1:12
end

@testset "Comparison recipe (filter vs smoother)" begin

    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = CSV.read(nile_path, DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)

    p = KFParms([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.0;;])
    a1 = [0.0]
    P1 = [1e7;;]

    # The tuple from kalman_filter_and_smooth plots directly
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), kalman_filter_and_smooth(p, y, a1, P1))
    @test !isempty(rec)
end

@testset "ObservablePlot recipe" begin

    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = CSV.read(nile_path, DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)

    p = KFParms([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.0;;])
    a1 = [0.0]
    P1 = [1e7;;]

    result = kalman_filter(p, y, a1, P1)

    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), result, ObservablePlot)
    @test !isempty(rec)

    # With actual data overlay
    rec2 = RecipesBase.apply_recipe(Dict{Symbol, Any}(:actual => y), result, ObservablePlot)
    @test !isempty(rec2)
end

@testset "Multivariate filter recipe" begin

    # 2 observables, 3 states
    m, p_obs, n = 3, 2, 50
    Z = rand(p_obs, m)
    H = 0.1 * I(p_obs) |> Matrix
    T_mat = 0.9 * I(m) |> Matrix
    R = I(m) |> Matrix
    Q = 0.1 * I(m) |> Matrix

    p = KFParms(Z, H, T_mat, R, Q)
    y = rand(p_obs, n)
    a1 = zeros(m)
    P1 = 10.0 * I(m) |> Matrix

    result = kalman_filter(p, y, a1, P1)

    # All states
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(:vars => :all), result)
    @test !isempty(rec)

    # Subset of states
    rec2 = RecipesBase.apply_recipe(Dict{Symbol, Any}(:vars => [1, 3]), result)
    @test !isempty(rec2)

    # Single state
    rec3 = RecipesBase.apply_recipe(Dict{Symbol, Any}(:vars => 2), result)
    @test !isempty(rec3)
end

# ============================================================================
# KalmanWorkspace (In-place API) Tests
# ============================================================================

@testset "KalmanWorkspace construction" begin

    # Create workspace from matrices
    p_obs, m, r, n = 2, 3, 3, 50
    Z = rand(p_obs, m)
    H = 0.1 * I(p_obs) |> Matrix
    T_mat = 0.9 * I(m) |> Matrix
    R_mat = I(m) |> Matrix
    Q = 0.1 * I(m) |> Matrix
    a1 = zeros(m)
    P1 = 10.0 * I(m) |> Matrix

    ws = KalmanWorkspace(Z, H, T_mat, R_mat, Q, a1, P1, n)

    @test ws.obs_dim == p_obs
    @test ws.state_dim == m
    @test ws.n_times == n
end

@testset "KalmanWorkspace filter and smoother" begin

    # Load Nile data
    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = CSV.read(nile_path, DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)
    n = size(y, 2)

    # Local level model
    Z = [1.0;;]
    H = [15099.0;;]
    T_mat = [1.0;;]
    R_mat = [1.0;;]
    Q = [1469.0;;]
    a1 = [0.0]
    P1 = [1e7;;]

    ws = KalmanWorkspace(Z, H, T_mat, R_mat, Q, a1, P1, n)

    # Run filter
    ll = kalman_filter!(ws, y)
    @test isfinite(ll)
    @test ll < 0

    # Check accessor methods
    @test size(filtered_states(ws)) == (1, n)
    @test size(predicted_states(ws)) == (1, n)
    @test loglikelihood(ws) == ll

    # Run smoother
    kalman_smoother!(ws)
    @test size(smoothed_states(ws)) == (1, n)
    @test all(isfinite.(smoothed_states(ws)))
end

@testset "KalmanWorkspace recipe" begin

    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = CSV.read(nile_path, DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)
    n = size(y, 2)

    ws = KalmanWorkspace(
        [1.0;;],
        [15099.0;;],
        [1.0;;],
        [1.0;;],
        [1469.0;;],
        [0.0],
        [1e7;;],
        n
    )
    kalman_filter!(ws, y)
    kalman_smoother!(ws)

    # Smoothed (default)
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), ws)
    @test !isempty(rec)

    # Filtered
    rec2 = RecipesBase.apply_recipe(Dict{Symbol, Any}(:what => :filtered), ws)
    @test !isempty(rec2)

    # Predicted
    rec3 = RecipesBase.apply_recipe(Dict{Symbol, Any}(:what => :predicted), ws)
    @test !isempty(rec3)
end

@testset "SmootherResult from KalmanWorkspace" begin

    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = CSV.read(nile_path, DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)
    n = size(y, 2)

    ws = KalmanWorkspace(
        [1.0;;],
        [15099.0;;],
        [1.0;;],
        [1.0;;],
        [1469.0;;],
        [0.0],
        [1e7;;],
        n
    )
    kalman_filter!(ws, y)
    kalman_smoother!(ws)

    # Create SmootherResult from workspace
    sr = SmootherResult(ws)
    @test size(smoothed_states(sr)) == (1, n)
    @test size(smoothed_states_cov(sr)) == (1, 1, n)

    # Recipe should work
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), sr)
    @test !isempty(rec)
end

end # Plotting recipes
