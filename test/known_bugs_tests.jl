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
    @testset "DynamicFactorModel EM moves the loadings with AR errors" begin
        # With error_lags > 0 the errors are states and H = 0, so y - e = Λf holds
        # exactly in the smoothed posterior and the EM loadings update is a fixed point:
        # the loadings never leave their random initial values.
        rng = MersenneTwister(1)
        N, n = 6, 500
        λ = [1.0, 0.8, 0.6, 1.2, 0.9, 0.7]
        σ² = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
        f = zeros(n)
        e = zeros(N, n)
        for t in 2:n
            f[t] = 0.7 * f[t - 1] + randn(rng)
            e[:, t] .= 0.6 .* e[:, t - 1] .+ sqrt.(σ²) .* randn(rng, N)
        end
        y = λ * f' .+ e
        Random.seed!(3)
        dfm = DynamicFactorModel(N, 1, n; error_lags = 1)
        Λ0 = copy(dfm.kf_ws.Z[:, 1])
        fit!(EM(), dfm, y; maxiter = 200)
        Λ = vec(loadings(dfm)[1])
        @test_broken maximum(abs.(Λ ./ Λ[1] .- λ ./ λ[1])) < 0.2
        @test Λ ≈ Λ0  # documents the fixed point; remove when the pin flips
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
