# Display, value equality and hashing of the user-facing types.

using Siphon
using Test
using LinearAlgebra
using Random
using StaticArrays

_plain(x) = sprint(show, MIME"text/plain"(), x)
_roundtrip(x) = eval(Meta.parse(repr(x)))

struct ShowTestMeasurement{ZT} <: Siphon.AbstractEKFMeasurement
    Z::ZT
end
Siphon.measurement(m::ShowTestMeasurement, a, t) = m.Z * a
Siphon.measurement!(out, m::ShowTestMeasurement, a, t) = mul!(out, m.Z, a)

@testset "show: parameters round-trip as constructor calls" begin
    prms = [
        SSMParameter(:ρ; lower = -0.99, upper = 0.99, init = 0.8),
        SSMParameter(:var_obs; lower = 0.0),
        SSMParameter(:μ, 2.5),
        SSMParameter{Float32}(:a, -1.0f0, 1.0f0, 0.5f0),
        dns_model([3, 12, 60]).params...,
        arma(2, 1).params...,
        local_level().params...
    ]
    for prm in prms
        @test _roundtrip(prm) === prm
    end
    @test repr(SSMParameter(:var_obs; lower = 0.0)) == "SSMParameter(:var_obs; lower = 0.0)"
    @test repr(SSMParameter(:ρ; lower = -1, upper = 1, init = 0.5)) ==
          "SSMParameter(:ρ; lower = -1.0, upper = 1.0, init = 0.5)"

    fps = [FreeParam(), FreeParam(:ρ), FreeParam(lower = 0.0),
        FreeParam(:ρ; lower = -0.99, upper = 0.99, init = 0.8), FreeParam(init = 2)]
    for fp in fps
        @test _roundtrip(fp) === fp
    end
    @test repr(FreeParam(lower = 0.0)) == "FreeParam(lower = 0.0)"
    @test repr(FreeParam(:ρ; init = 0.5)) == "FreeParam(:ρ; init = 0.5)"
end

@testset "SSMParameter typed constructor" begin
    @test SSMParameter{Float64}(:a, -1, 1, 0) === SSMParameter(:a, -1.0, 1.0, 0.0)
    @test SSMParameter(:a, -1, 1, 0.5) isa SSMParameter{Float64}
    @test_throws "must be below upper bound" SSMParameter{Float64}(:a, 1, -1, 0)
end

@testset "show: priors round-trip" begin
    for p in (FlatPrior(), NormalPrior(0, 1), InverseGammaPrior(2, 3),
        ParameterPriors(var_obs = InverseGammaPrior(2, 3), ρ = NormalPrior(0, 0.5)),
        CompositePrior(FlatPrior(), NormalPrior(0, 1)))
        @test _roundtrip(p) == p
    end
    @test repr(NormalPrior(0, 1)) == "NormalPrior(0.0, 1.0)"
end

@testset "show: specs" begin
    spec = local_level()
    @test repr(spec) == "SSMSpec(:LocalLevel, m = 1, p = 1, r = 1, 2 parameters)"
    s = _plain(spec)
    @test startswith(s, "SSMSpec LocalLevel with 1 state, 1 observed series, 1 shock")
    @test occursin("var_obs", s) && occursin("(default)", s)
    @test occursin(r"H\s+var_obs", s) && occursin(r"Q\s+var_level", s)

    dns = dns_model([3, 12, 60])
    s = _plain(dns)
    @test occursin(r"Z\s+λ", s)
    @test occursin("3 observed series", s)
    @test repr(dns.matrix_exprs[:Z]) == "MatrixExpr(3×3, free: λ)"

    @test repr(spec.H) == "SSMMatrixSpec(1×1, free: var_obs)"
    @test repr(spec.Z) == "SSMMatrixSpec(1×1, fixed)"
    @test _plain(local_linear_trend().T) == "2×2 SSMMatrixSpec:\n  1.0  1.0\n  0.0  1.0"

    # Specs no longer print their whole internal representation.
    @test length(repr(spec)) < 100
    @test length(_plain(spec)) < 400
end

@testset "show: filter output and workspaces" begin
    rng = MersenneTwister(3)
    y = cumsum(randn(rng, 1, 40); dims = 2)
    p = KFParms([1.0;;], [1.0;;], [1.0;;], [1.0;;], [1.0;;])
    @test repr(p) == "KFParms(p = 1, m = 1, r = 1)"
    @test occursin("Q: 1×1 Matrix{Float64}", _plain(p))
    @test repr(KFParms(1.0, 1.0, 1.0, 1.0, 1.0)) == "KFParms(scalar)"

    res = kalman_filter(p, y, [0.0], [1e7;;])
    @test startswith(repr(res), "KalmanFilterResult(p = 1, m = 1, n = 40, loglik = ")
    @test repr(kalman_smoother(res)) == "SmootherResult(m = 1, n = 40)"

    ws = KalmanWorkspace(p, [0.0], [1e7;;], 40)
    @test repr(ws) == "KalmanWorkspace{Float64}(p = 1, m = 1, r = 1, n = 40, not run)"
    kalman_filter!(ws, y)
    @test occursin("loglik = ", repr(ws))
    kalman_smoother!(ws)
    @test endswith(repr(ws), ", smoothed)")
    @test length(repr(ws)) < 120
    @test repr(Siphon.EMWorkspace(ws)) ==
          "EMWorkspace{Float64}(p = 1, m = 1, r = 1, n = 40)"

    dws = DiffuseKalmanWorkspace(p.Z, p.H, p.T, p.R, p.Q, [0.0], [0.0;;], [1.0;;], 40)
    @test endswith(repr(dws), "not run)")
    kalman_filter!(dws, y)
    @test occursin("d = 1, loglik = ", repr(dws))
    dres = kalman_filter(p, y, [0.0], [0.0;;], [1.0;;])
    @test occursin("d = 1, loglik = ", repr(dres))

    # A failed run reports the period in which it stopped.
    bad = KFParms([1.0;;], [-1.0;;], [1.0;;], [1.0;;], [0.0;;])
    @test repr(kalman_filter(bad, y, [0.0], [0.0;;])) ==
          "KalmanFilterResult(p = 1, m = 1, n = 40, failed in period 1)"
end

@testset "show: extended Kalman filter" begin
    Z = [1.0 0.0; 0.3 1.0]
    H, Tm, R, Q = Matrix(0.5I, 2, 2), [0.9 0.0; 0.1 0.7], Matrix(1.0I, 2, 2),
    Matrix(0.2I, 2, 2)
    y = randn(MersenneTwister(4), 2, 20)
    ep = EKFParms(ShowTestMeasurement(Z), H, Tm, R, Q)
    @test repr(ep) == "EKFParms(p = 2, m = 2, r = 2, ShowTestMeasurement, ADJacobian())"
    @test startswith(repr(kalman_filter(ep, y, zeros(2), Matrix(1.0I, 2, 2))),
        "EKFFilterResult(p = 2, m = 2, n = 20, loglik = ")
    ws = EKFWorkspace(ShowTestMeasurement(Z), ADJacobian(), H, Tm, R, Q, zeros(2),
        Matrix(1.0I, 2, 2), 20)
    @test endswith(repr(ws), "ShowTestMeasurement, ADJacobian(), not run)")
    spec = custom_ekf(measurement = ShowTestMeasurement(Z), H = diag_free(2, :var_h),
        T = I, R = I, Q = diag_free(2, :var_q), a1 = zeros(2), P1 = 10.0I)
    @test repr(spec) == "EKFSpec(:CustomEKF, m = 2, p = 2, r = 2, 4 parameters)"
    @test occursin(r"H\s+var_h_1, var_h_2", _plain(spec))
end

@testset "show: models" begin
    rng = MersenneTwister(5)
    y = cumsum(randn(rng, 1, 40); dims = 2) .+ randn(rng, 1, 40)
    model = StateSpaceModel(local_level(), 40)
    @test repr(model) == "StateSpaceModel{Float64}(:LocalLevel, n = 40, not fitted)"
    @test occursin("Parameters not set", _plain(model))
    fit!(MLE(), model, y)
    @test occursin(r"^StateSpaceModel\{Float64\}\(:LocalLevel, n = 40, fitted, loglik = ",
        repr(model))
    @test occursin(r"var_obs\s+= ", _plain(model))
    @test length(_plain(model)) < 300

    known = StateSpaceModel(local_level(), (var_obs = 1.0, var_level = 2.0), 40)
    @test repr(known) == "StateSpaceModel{Float64}(:LocalLevel, n = 40, fitted)"

    dfm = DynamicFactorModel(4, 1, 30; rng = MersenneTwister(1))
    @test repr(dfm) == "DynamicFactorModel{Float64}(N = 4, k = 1, n = 30, not fitted)"
    @test occursin("factor_lags = 1", _plain(dfm))

    ld = SSMLogDensity(local_level(), y; prior = ParameterPriors(var_obs = NormalPrior(0, 1)))
    @test repr(ld) ==
          "SSMLogDensity(:LocalLevel, 2 parameters, 1×40 data, " *
          "prior = ParameterPriors(var_obs = NormalPrior(0.0, 1.0)))"
end

@testset "value equality and hashing" begin
    mats = [3, 6, 12, 24, 60, 120]
    for f in (local_level, local_linear_trend, () -> ar1(), () -> arma(2, 1),
        () -> dynamic_factor(4, 1), () -> dns_model(mats),
        () -> local_level(; diffuse = :exact))
        a, b = f(), f()
        @test a == b
        @test isequal(a, b)
        @test hash(a) == hash(b)
    end
    @test local_level() != local_linear_trend()
    @test local_level(var_obs = 2.0) != local_level()
    @test length(Set([local_level(), local_level(), ar1()])) == 2
    @test Dict(local_level() => 1)[local_level()] == 1

    # Equality is by value, regardless of the array types holding the values.
    dense = KFParms(ones(1, 1), ones(1, 1), ones(1, 1), ones(1, 1), ones(1, 1))
    static = KFParms(SMatrix{1, 1}(1.0), SMatrix{1, 1}(1.0), SMatrix{1, 1}(1.0),
        SMatrix{1, 1}(1.0), SMatrix{1, 1}(1.0))
    @test dense == static
    @test hash(dense) == hash(static)
    @test dense != KFParms(ones(1, 1), 2ones(1, 1), ones(1, 1), ones(1, 1), ones(1, 1))

    # `isequal` treats NaN as equal to itself, `==` does not.
    nan = KFParms(fill(NaN, 1, 1), ones(1, 1), ones(1, 1), ones(1, 1), ones(1, 1))
    nan2 = KFParms(fill(NaN, 1, 1), ones(1, 1), ones(1, 1), ones(1, 1), ones(1, 1))
    @test nan != nan2
    @test isequal(nan, nan2)
    @test hash(nan) == hash(nan2)

    spec = local_level()
    @test spec.H == local_level().H
    @test hash(spec.H) == hash(local_level().H)
end
