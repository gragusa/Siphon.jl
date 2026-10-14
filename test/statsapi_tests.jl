# StatsAPI and LogDensityProblems generics on Siphon models.

using Siphon
using Test
using LinearAlgebra
using Random
import LogDensityProblems
import StatsAPI
import StatsBase

@testset "StatsAPI: shared bindings" begin
    @test AbstractStateSpaceModel <: StatsAPI.StatisticalModel
    for f in (:fit!, :loglikelihood, :predict, :isfitted, :nobs, :coef, :coefnames,
        :vcov, :stderror, :dof, :aic, :aicc, :bic, :coeftable)
        @test getfield(Siphon, f) === getfield(StatsAPI, f)
    end
    @test Siphon.logdensity === LogDensityProblems.logdensity
end

@testset "StatsAPI: StateSpaceModel" begin
    rng = Xoshiro(1)
    n = 200
    y = cumsum(randn(rng, 1, n); dims = 2) .+ 3 .* randn(rng, 1, n)
    spec = local_level()

    model = StateSpaceModel(spec, n)
    @test_throws "Model not fitted" coef(model)
    @test_throws "Model not fitted" vcov(model)
    @test nobs(model) == n
    @test dof(model) == 2
    @test coefnames(model) == ["var_obs", "var_level"]

    fit!(MLE(), model, y)
    ll = loglikelihood(model)
    @test coef(model) == collect(parameters(model))
    @test aic(model) ≈ -2ll + 4
    @test aicc(model) ≈ -2ll + 4 + 12 / (n - 3)
    @test bic(model) ≈ -2ll + 2log(n)

    ref = optimize_ssm_with_stderr(spec, y)
    V = vcov(model)
    @test V ≈ V'
    @test V ≈ ref.vcov rtol = 1e-6
    @test stderror(model) ≈ ref.stderr rtol = 1e-6

    # The model keeps its own copy of the data.
    y2 = copy(y)
    em = fit!(EM(), StateSpaceModel(spec, n), y2)
    se = stderror(em)
    y2 .= 0
    @test stderror(em) == se
    @test all(isfinite, se)

    # A model with given parameters has a covariance once it has seen data.
    known = StateSpaceModel(spec, parameters(model), n)
    @test_throws "No data stored" vcov(known)
    kalman_filter!(known, y)
    @test vcov(known) ≈ V rtol = 1e-6

    tab = coeftable(model)
    @test tab isa StatsBase.CoefTable
    @test tab.rownms == ["var_obs", "var_level"]
    @test tab.colnms[(end - 1):end] == ["Lower 95%", "Upper 95%"]
    @test tab.cols[1] == coef(model)
    @test tab.cols[2] ≈ stderror(model)
    @test tab.cols[4][1] < 1e-10
    @test tab.cols[5] ≈ coef(model) .- 1.959963984540054 .* stderror(model)
    @test coeftable(model; level = 0.9).colnms[5] == "Lower 90%"
    @test_throws DomainError coeftable(model; level = 1.0)
end

@testset "StatsAPI: DynamicFactorModel" begin
    model = DynamicFactorModel(4, 1, 50; rng = Xoshiro(2))
    @test model isa StatsAPI.StatisticalModel
    @test nobs(model) == 50
    @test !isfitted(model)
end

# `using` Siphon next to StatsBase and Distributions leaves the shared names
# usable unqualified.
module StatsAPIClashCheck
using Siphon, StatsBase, Distributions, Test, Random
@testset "StatsAPI: no name clashes with StatsBase and Distributions" begin
    y = randn(Xoshiro(3), 1, 60)
    model = fit!(MLE(), StateSpaceModel(local_level(), 60), y)
    @test loglikelihood(model) isa Float64
    @test loglikelihood(Normal(), [0.0]) ≈ -log(2π) / 2
    @test nobs(model) == 60
    @test length(coef(model)) == dof(model)
end
end
