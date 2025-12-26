"""
Tests for RecipesBase plotting recipes.
"""

using TestItems

@testitem "quantile_normal" tags=[:recipes, :smoke] begin
    using Siphon: quantile_normal

    # Standard values
    @test quantile_normal(0.975) ≈ 1.96 atol=0.01
    @test quantile_normal(0.95) ≈ 1.645 atol=0.01
    @test quantile_normal(0.90) ≈ 1.28 atol=0.01
    @test quantile_normal(0.5 + eps()) ≈ 0.0 atol=0.01
end

@testitem "select_vars" tags=[:recipes, :smoke] begin
    using Siphon: select_vars

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

@testitem "confidence_bands vector" tags=[:recipes, :smoke] begin
    using Siphon: confidence_bands

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

@testitem "confidence_bands matrix" tags=[:recipes] begin
    using Siphon: confidence_bands

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

@testitem "SmootherResult wrapper" tags=[:recipes] begin
    using Siphon
    using Siphon: SmootherResult

    # Test wrapping NamedTuple
    m, n = 2, 10
    nt = (alpha=rand(m, n), V=rand(m, m, n))
    sr = SmootherResult(nt)

    @test sr.alpha == nt.alpha
    @test sr.V == nt.V
    @test sr.p === nothing
    @test sr.time === nothing

    # With optional arguments
    sr2 = SmootherResult(nt; time=1:n)
    @test sr2.time == 1:n

    # Accessor methods
    @test smoothed_states(sr) == sr.alpha
    @test variances_smoothed_states(sr) == sr.V
end

@testitem "ForecastResult wrapper" tags=[:recipes] begin
    using Siphon
    using Siphon: ForecastResult

    p_obs, m_state, h = 3, 2, 12
    nt = (yhat=rand(p_obs, h), a=rand(m_state, h),
          P=rand(m_state, m_state, h), F=rand(p_obs, p_obs, h))

    fr = ForecastResult(nt)

    @test fr.yhat == nt.yhat
    @test fr.a == nt.a
    @test fr.P == nt.P
    @test fr.F == nt.F
    @test fr.time === nothing

    # With time
    fr2 = ForecastResult(nt; time=101:112)
    @test fr2.time == 101:112

    # Accessor methods
    @test forecast_observations(fr) == fr.yhat
    @test forecast_states(fr) == fr.a
end

@testitem "SmootherResult convenience constructor" tags=[:recipes] begin
    using Siphon
    using Siphon: SmootherResult
    using DelimitedFiles

    # Load Nile data
    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = readdlm(nile_path, ',', Float64)
    y = reshape(nile[:, 1], 1, :)

    # Local level model parameters
    p = KFParms([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.0;;])
    a1 = [0.0]
    P1 = [1e7;;]

    # Convenience constructor runs filter + smoother
    sr = SmootherResult(p, y, a1, P1)

    @test size(sr.alpha) == (1, size(y, 2))
    @test size(sr.V) == (1, 1, size(y, 2))
    @test sr.p === p
end

@testitem "KalmanFilterResult recipe produces series" tags=[:recipes] begin
    using Siphon
    using RecipesBase
    using DelimitedFiles

    # Load Nile data
    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = readdlm(nile_path, ',', Float64)
    y = reshape(nile[:, 1], 1, :)

    # Local level model
    p = KFParms([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.0;;])
    a1 = [0.0]
    P1 = [1e7;;]

    result = kalman_filter(p, y, a1, P1)

    # Recipe should produce output
    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), result)
    @test !isempty(rec)

    # With options
    rec2 = RecipesBase.apply_recipe(Dict{Symbol,Any}(:vars => 1, :level => 0.90), result)
    @test !isempty(rec2)

    # Predicted instead of filtered
    rec3 = RecipesBase.apply_recipe(Dict{Symbol,Any}(:filtered => false), result)
    @test !isempty(rec3)
end

@testitem "SmootherResult recipe produces series" tags=[:recipes] begin
    using Siphon
    using Siphon: SmootherResult
    using RecipesBase
    using DelimitedFiles

    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = readdlm(nile_path, ',', Float64)
    y = reshape(nile[:, 1], 1, :)

    p = KFParms([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.0;;])
    a1 = [0.0]
    P1 = [1e7;;]

    sr = SmootherResult(p, y, a1, P1)

    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), sr)
    @test !isempty(rec)
end

@testitem "ForecastResult recipe produces series" tags=[:recipes] begin
    using Siphon
    using Siphon: ForecastResult
    using RecipesBase

    # Create mock forecast result
    h = 12
    nt = (yhat=rand(1, h), a=rand(1, h), P=rand(1, 1, h), F=rand(1, 1, h))
    fr = ForecastResult(nt)

    # Observations (default)
    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), fr)
    @test !isempty(rec)

    # States
    rec2 = RecipesBase.apply_recipe(Dict{Symbol,Any}(:what => :states), fr)
    @test !isempty(rec2)
end

@testitem "Comparison recipe (filter vs smoother)" tags=[:recipes] begin
    using Siphon
    using Siphon: SmootherResult
    using RecipesBase
    using DelimitedFiles

    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = readdlm(nile_path, ',', Float64)
    y = reshape(nile[:, 1], 1, :)

    p = KFParms([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.0;;])
    a1 = [0.0]
    P1 = [1e7;;]

    filt = kalman_filter(p, y, a1, P1)
    smooth = SmootherResult(p, y, a1, P1)

    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), (filt, smooth))
    @test !isempty(rec)
end

@testitem "ObservablePlot recipe" tags=[:recipes] begin
    using Siphon
    using Siphon: ObservablePlot
    using RecipesBase
    using DelimitedFiles

    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = readdlm(nile_path, ',', Float64)
    y = reshape(nile[:, 1], 1, :)

    p = KFParms([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.0;;])
    a1 = [0.0]
    P1 = [1e7;;]

    result = kalman_filter(p, y, a1, P1)

    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), result, ObservablePlot)
    @test !isempty(rec)

    # With actual data overlay
    rec2 = RecipesBase.apply_recipe(Dict{Symbol,Any}(:actual => y), result, ObservablePlot)
    @test !isempty(rec2)
end

@testitem "Multivariate filter recipe" tags=[:recipes] begin
    using Siphon
    using RecipesBase

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
    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(:vars => :all), result)
    @test !isempty(rec)

    # Subset of states
    rec2 = RecipesBase.apply_recipe(Dict{Symbol,Any}(:vars => [1, 3]), result)
    @test !isempty(rec2)

    # Single state
    rec3 = RecipesBase.apply_recipe(Dict{Symbol,Any}(:vars => 2), result)
    @test !isempty(rec3)
end

# ============================================================================
# KalmanWorkspace (In-place API) Tests
# ============================================================================

@testitem "KalmanWorkspace construction" tags=[:recipes, :inplace] begin
    using Siphon
    using LinearAlgebra

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

@testitem "KalmanWorkspace filter and smoother" tags=[:recipes, :inplace] begin
    using Siphon
    using LinearAlgebra
    using DelimitedFiles

    # Load Nile data
    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = readdlm(nile_path, ',', Float64)
    y = reshape(nile[:, 1], 1, :)
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

@testitem "KalmanWorkspace recipe" tags=[:recipes, :inplace] begin
    using Siphon
    using RecipesBase
    using LinearAlgebra
    using DelimitedFiles

    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = readdlm(nile_path, ',', Float64)
    y = reshape(nile[:, 1], 1, :)
    n = size(y, 2)

    ws = KalmanWorkspace([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.0;;],
                          [0.0], [1e7;;], n)
    kalman_filter!(ws, y)
    kalman_smoother!(ws)

    # Smoothed (default)
    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), ws)
    @test !isempty(rec)

    # Filtered
    rec2 = RecipesBase.apply_recipe(Dict{Symbol,Any}(:what => :filtered), ws)
    @test !isempty(rec2)

    # Predicted
    rec3 = RecipesBase.apply_recipe(Dict{Symbol,Any}(:what => :predicted), ws)
    @test !isempty(rec3)
end

@testitem "SmootherResult from KalmanWorkspace" tags=[:recipes, :inplace] begin
    using Siphon
    using Siphon: SmootherResult
    using RecipesBase
    using LinearAlgebra
    using DelimitedFiles

    nile_path = joinpath(@__DIR__, "Nile.csv")
    nile = readdlm(nile_path, ',', Float64)
    y = reshape(nile[:, 1], 1, :)
    n = size(y, 2)

    ws = KalmanWorkspace([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.0;;],
                          [0.0], [1e7;;], n)
    kalman_filter!(ws, y)
    kalman_smoother!(ws)

    # Create SmootherResult from workspace
    sr = SmootherResult(ws)
    @test size(sr.alpha) == (1, n)
    @test size(sr.V) == (1, 1, n)

    # Recipe should work
    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), sr)
    @test !isempty(rec)
end
