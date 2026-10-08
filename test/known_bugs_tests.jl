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

@testset "exact diffuse update after a step with nonzero Pstar" begin
    # The Finf-invertible update adds the cross term T*M∞*M*'*T'/F∞ where
    # Durbin-Koopman subtract it, so Pstar becomes asymmetric from the second
    # diffuse step on. A local linear trend has d = 2; after it, the exact filter
    # must match the κ → ∞ limit of the approximate one.
    rng = MersenneTwister(8901)
    n = 40
    y = cumsum(cumsum(randn(rng, 1, n); dims = 2); dims = 2) .+ 5 .* randn(rng, 1, n)
    p = KFParms([1.0 0.0], [25.0;;], [1.0 1.0; 0.0 1.0], Matrix(1.0I, 2, 2),
        [1.0 0.0; 0.0 0.1])
    res = kalman_filter(p, y, zeros(2), zeros(2, 2), Matrix(1.0I, 2, 2))
    d = diffuse_period(res)
    P_big = 1e8 * Matrix(1.0I, 2, 2)
    approx = kalman_filter(p, y, zeros(2), P_big)
    @test_broken res.Pt[:, :, d + 1] ≈ approx.Pt[:, :, d + 1] rtol = 1e-5
    @test_broken res.loglik ≈ kalman_loglik(p, y, zeros(2), P_big) -
                              kalman_loglik(p, y[:, 1:d], zeros(2), P_big) rtol = 1e-6
end

@testset "exact diffuse with a singular nonzero Finf" begin
    # With more observed rows than diffuse directions, Finf = Z*Pinf*Z' is
    # singular but nonzero. The filters then take the Fstar step, which leaves Pinf
    # unchanged, so the diffuse period never ends. Durbin-Koopman handle this case
    # by processing the period one row at a time.
    rng = MersenneTwister(2)
    n = 40
    y = cumsum(randn(rng, 1, n); dims = 2) .+ randn(rng, 2, n)
    p = KFParms(reshape([1.0, 1.0], 2, 1), [1.0 0.0; 0.0 2.0], [1.0;;], [1.0;;], [0.5;;])
    @test_broken diffuse_period(kalman_filter(p, y, [0.0], [0.0;;], [1.0;;])) == 1
end

end # Known bugs
