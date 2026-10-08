"""
Known bugs, pinned with `@test_broken`.

Each test states the correct behaviour and currently fails. When a fix lands,
the test reports an unexpected pass; in the same change, move it to the test
file for that feature as a plain `@test`.
"""

using Test
using Siphon
using LinearAlgebra
using Random
using ForwardDiff
using CSV
using DataFrames

let nile = CSV.read(joinpath(@__DIR__, "Nile.csv"), DataFrame; header = false)
    global y_bug = reshape(Float64.(nile[!, 1]), 1, :)
end

function _throws(f)
    try
        f()
        return false
    catch
        return true
    end
end

@testset "Known bugs" begin

@testset "DynamicFactorModel EM updates idiosyncratic variances" begin
    # sync_params_from_ssm! overwrites the updated variances with stale values.
    Random.seed!(1)
    N, n = 6, 300
    σ² = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
    λ = [1.0, 0.8, 0.6, 1.2, 0.9, 0.7]
    f = zeros(n)
    for t in 2:n
        f[t] = 0.7 * f[t - 1] + randn()
    end
    y = λ * f' .+ sqrt.(σ²) .* randn(N, n)
    dfm = DynamicFactorModel(N, 1, n)
    fit!(EM(), dfm, y; maxiter = 200)
    @test_broken maximum(abs.(idiosyncratic_variances(dfm) .- 1.0)) > 0.05
end

@testset "profile_em_ssm returns a consistent θ" begin
    df = CSV.read(joinpath(@__DIR__, "marss_dns_data.csv"), DataFrame)
    y = Matrix(Matrix(df)')
    spec = dns_model([3, 12, 24, 60, 120];
        T_structure = :diagonal, H_structure = :diagonal,
        Q_structure = :full, λ_init = 0.06, diffuse = false)
    res = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        profile_em_ssm(spec, y; λ_grid = 0.05:0.01:0.07, verbose = false,
            maxiter = 50)
    end
    @test_broken ssm_loglik(spec, res.θ, y) ≈ res.loglik rtol = 1e-4
end

@testset "EM with a diagonal transition matrix" begin
    # test/em_diagonal_T_tests.jl specifies a `T_diagonal` restriction that the
    # EM does not implement; that file is not included until it does.
    m = first(methods(Siphon.DSL._em_general_ssm_full_cov))
    @test_broken :T_diagonal in Base.kwarg_decl(m)
end

end # Known bugs
