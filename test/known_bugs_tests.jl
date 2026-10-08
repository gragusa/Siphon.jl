"""
Known bugs, pinned with `@test_broken`.

Each test states the correct behaviour and currently fails. When a fix lands,
the test reports an unexpected pass; turn the `@test_broken` into `@test` in
the same change.
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

# Measurement type that implements only the functional `measurement`, not the
# in-place `measurement!` that EKFWorkspace requires.
struct IncompleteMeasurement <: AbstractEKFMeasurement end
Siphon.measurement(::IncompleteMeasurement, a, t) = a[1:1]

function _throws(f)
    try
        f()
        return false
    catch
        return true
    end
end

@testset "Known bugs" begin

@testset "MLE maximises the log-likelihood" begin
    # optimize_ssm minimises -logdensity, which includes the log-Jacobian of
    # the parameter transform, so it returns a penalised mode instead of the MLE.
    spec = local_level()
    res = optimize_ssm(spec, y_bug)
    @test_broken res.loglik ≈ ssm_loglik(spec, res.θ, y_bug) rtol = 1e-6
    # Durbin & Koopman (2012) MLE for the Nile data
    @test_broken isapprox(res.θ.var_obs, 15099.0; rtol = 0.02)
    @test_broken isapprox(res.θ.var_level, 1469.1; rtol = 0.05)
end

@testset "exact diffuse initialisation through a spec" begin
    # SSMSpec.P1_inf is never read, so diffuse = :exact runs with P1 = 0.
    θ = (var_obs = 15099.0, var_level = 1469.1)
    spec = local_level(diffuse = :exact)
    p = KFParms([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.1;;])
    exact = kalman_loglik(p, y_bug, [0.0], zeros(1, 1), ones(1, 1))
    @test_broken ssm_loglik(spec, θ, y_bug) ≈ exact
end

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

@testset "partially observed periods" begin
    spec = custom_ssm(
        Z = reshape([1.0, 1.0], 2, 1),
        H = diag_free([:h1, :h2]; init = 1.0),
        T = [1.0],
        R = [1.0],
        Q = [FreeParam(:q, init = 1.0, lower = 0.0)],
        a1 = [0.0],
        P1 = [10.0]
    )
    Random.seed!(2)
    n = 60
    level = cumsum(randn(n))
    y = vcat(level', level') .+ randn(2, n)
    y[1, 10:15] .= NaN   # row 1 missing, row 2 observed
    y[:, 30] .= NaN      # fully missing period

    θ = (h1 = 1.0, h2 = 1.0, q = 1.0)
    ss = build_linear_state_space(spec, θ, y; use_static = false)
    ref = kalman_smoother(ss.p, y, ss.a1, ss.P1)

    # The default static path rejects partially observed periods.
    @test_broken !_throws(() -> predict(spec, θ, y))

    # The StateSpaceModel smoother drops the observed-row mask.
    model = StateSpaceModel(spec, n)
    fit!(MLE(), model, y; static = :off)
    ss_fit = build_linear_state_space(spec, parameters(model), y; use_static = false)
    ref_fit = kalman_smoother(ss_fit.p, y, ss_fit.a1, ss_fit.P1)
    @test_broken smoothed_states(model) ≈ ref_fit.α

    # SmootherResult's convenience constructor drops the observed-row mask.
    @test_broken !_throws(() -> SmootherResult(ss.p, y, ss.a1, ss.P1))
    @test size(ref.α) == (1, n)
end

@testset "quantile_normal below the median" begin
    @test_broken quantile_normal(0.025) ≈ -1.96 atol = 0.01
end

@testset "InverseGammaPrior is callable" begin
    prior = InverseGammaPrior(2.0, 1.0, (:var_obs,))
    @test_broken !_throws(() -> prior((var_obs = 1.0,)))
end

@testset "parameter bounds survive the transform" begin
    spec = custom_ssm(
        Z = [1.0],
        H = [FreeParam(:x, init = 2.0, lower = 1.0)],
        T = [FreeParam(:z, init = 0.5, lower = -1.0)],
        R = [1.0],
        Q = [1.0],
        a1 = [0.0],
        P1 = [1e7]
    )
    θc, _ = transform_to_constrained(spec, [-20.0, -50.0])
    @test_broken θc.x >= 1.0
    @test_broken θc.z >= -1.0
end

@testset "EKFWorkspace checks the measurement protocol" begin
    # The AbstractEKFMeasurement stubs make the hasmethod checks always pass.
    M = [1.0;;]
    @test_broken _throws(() -> EKFWorkspace(IncompleteMeasurement(), ADJacobian(),
        M, M, M, M, [0.0], [1.0;;], 10))
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

@testset "DNS models are ForwardDiff-compatible" begin
    df = CSV.read(joinpath(@__DIR__, "marss_dns_data.csv"), DataFrame)
    y = Matrix(Matrix(df)')
    spec = dns_model([3, 12, 24, 60, 120];
        T_structure = :diagonal, H_structure = :diagonal,
        Q_structure = :diagonal, diffuse = false)
    names = Tuple(param_names(spec))
    f(v) = ssm_loglik(spec, NamedTuple{names}(Tuple(v)), y)
    @test_broken !_throws(() -> ForwardDiff.gradient(f, initial_values(spec)))
end

@testset "EM with a diagonal transition matrix" begin
    # test/em_diagonal_T_tests.jl specifies a `T_diagonal` restriction that the
    # EM does not implement; that file is not included until it does.
    m = first(methods(Siphon.DSL._em_general_ssm_full_cov))
    @test_broken :T_diagonal in Base.kwarg_decl(m)
end

end # Known bugs
