# Element types of workspaces and models, and the simulation-smoother return types.

using Siphon
using Test
using LinearAlgebra
using Random

struct ConstructionTestMeasurement{ZT} <: Siphon.AbstractEKFMeasurement
    Z::ZT
end
Siphon.measurement(m::ConstructionTestMeasurement, a, t) = m.Z * a
Siphon.measurement!(out, m::ConstructionTestMeasurement, a, t) = mul!(out, m.Z, a)

@testset "Element type: dimension constructors" begin
    @test KalmanWorkspace(1, 2, 2, 5) isa KalmanWorkspace{Float64}
    @test KalmanWorkspace{Float32}(1, 2, 2, 5) isa KalmanWorkspace{Float32}
    @test DiffuseKalmanWorkspace(1, 2, 2, 5) isa DiffuseKalmanWorkspace{Float64}
    @test DiffuseKalmanWorkspace{Float32}(1, 2, 2, 5) isa DiffuseKalmanWorkspace{Float32}
    @test Siphon.EMWorkspace(1, 2, 2, 5) isa Siphon.EMWorkspace{Float64}
    @test Siphon.EMWorkspace{Float32}(1, 2, 2, 5) isa Siphon.EMWorkspace{Float32}
    @test SmootherWorkspace(2, 5) isa SmootherWorkspace{Float64}
    @test SmootherWorkspace{Float32}(2, 5) isa SmootherWorkspace{Float32}
end

@testset "Element type: from system matrices" begin
    Z, H, Tm, R, Q = [1 0], [1;;], [1 1; 0 1], [1 0; 0 1], [1 0; 0 1]
    a1, P1 = [0, 0], [10 0; 0 10]

    # Integer arrays give a floating-point workspace.
    ws = KalmanWorkspace(Z, H, Tm, R, Q, a1, P1, 5)
    @test ws isa KalmanWorkspace{Float64}
    @test obs_matrix(ws) == Z
    @test DiffuseKalmanWorkspace(Z, H, Tm, R, Q, a1, 0P1, P1, 5) isa
          DiffuseKalmanWorkspace{Float64}

    # The element type holds every argument, not only Z.
    f32 = map(x -> Float32.(x), (Z, H, Tm, R, Q, a1, P1))
    @test KalmanWorkspace(f32..., 5) isa KalmanWorkspace{Float32}
    @test KalmanWorkspace(f32[1:6]..., Float64.(P1), 5) isa KalmanWorkspace{Float64}

    # An explicit type parameter wins.
    @test KalmanWorkspace{Float32}(Z, H, Tm, R, Q, a1, P1, 5) isa KalmanWorkspace{Float32}
    @test DiffuseKalmanWorkspace{Float32}(Z, H, Tm, R, Q, a1, 0P1, P1, 5) isa
          DiffuseKalmanWorkspace{Float32}
    kf = KFParms(Z, H, Tm, R, Q)
    @test KalmanWorkspace(kf, a1, P1, 5) isa KalmanWorkspace{Float64}
    @test KalmanWorkspace{Float32}(kf, a1, P1, 5) isa KalmanWorkspace{Float32}

    # Workspaces built from a workspace share its element type.
    ws32 = KalmanWorkspace{Float32}(kf, a1, P1, 5)
    @test Siphon.EMWorkspace(ws32) isa Siphon.EMWorkspace{Float32}
    @test SimulationSmootherWorkspace(ws32) isa SimulationSmootherWorkspace{Float32}

    meas = ConstructionTestMeasurement(Float32[1 0; 0 1])
    I2 = Matrix{Float32}(I, 2, 2)
    @test EKFWorkspace(meas, ADJacobian(), I2, I2, I2, I2, zeros(Float32, 2), I2, 5) isa
          EKFWorkspace{Float32}
    @test EKFWorkspace(meas, ADJacobian(), [1 0; 0 1], I2, I2, I2, zeros(2), I2, 5) isa
          EKFWorkspace{Float64}
end

@testset "Element type: models" begin
    y = cumsum(randn(MersenneTwister(1), 1, 40); dims = 2)
    spec = local_level()
    @test StateSpaceModel(spec, 40) isa StateSpaceModel{Float64}

    model = StateSpaceModel{Float32}(spec, 40)
    @test model isa StateSpaceModel{Float32}
    fit!(MLE(), model, y)
    @test loglikelihood(model) isa Float32
    @test eltype(filtered_states(model)) == Float32

    known = StateSpaceModel{Float32}(spec, (var_obs = 1.0, var_level = 2.0), 40)
    @test parameters(known) == (var_obs = 1.0f0, var_level = 2.0f0)
    @test StateSpaceModel{Float32}(spec, [1.0, 2.0], 40) isa StateSpaceModel{Float32}
    @test StateSpaceModel(spec, [1.0, 2.0], 40) isa StateSpaceModel{Float64}

    @test DynamicFactorModel(4, 1, 30) isa DynamicFactorModel{Float64}
    @test DynamicFactorModel{Float32}(4, 1, 30; factor_lags = 2) isa
          DynamicFactorModel{Float32}

    # The element type is a type parameter, not a keyword.
    @test_throws MethodError StateSpaceModel(spec, 40; T = Float32)
    @test_throws MethodError DynamicFactorModel(4, 1, 30; T = Float32)
end

@testset "simulation_smoother return types" begin
    rng = MersenneTwister(2)
    m, n = 2, 30
    Z, H = [1.0 0.0], [0.5;;]
    Tm, R, Q = [1.0 1.0; 0.0 1.0], Matrix(1.0I, 2, 2), Matrix(0.1I, 2, 2)
    y = cumsum(randn(rng, 1, n); dims = 2)
    ws = KalmanWorkspace(Z, H, Tm, R, Q, zeros(m), Matrix(10.0I, m, m), n)

    one_draw = @inferred simulation_smoother(ws, y; rng = Xoshiro(7))
    @test one_draw isa Matrix{Float64}
    @test size(one_draw) == (m, n)

    draws = @inferred simulation_smoother(ws, y, 1; rng = Xoshiro(7))
    @test draws isa Array{Float64, 3}
    @test size(draws) == (m, n, 1)
    @test draws[:, :, 1] ≈ one_draw

    @test size(simulation_smoother(ws, y, 3; rng = Xoshiro(7))) == (m, n, 3)
end

@testset "Diffuse workspaces have no EM or simulation-smoother storage" begin
    p = KFParms([1.0;;], [1.0;;], [1.0;;], [1.0;;], [1.0;;])
    dws = DiffuseKalmanWorkspace(p.Z, p.H, p.T, p.R, p.Q, [0.0], [0.0;;], [1.0;;], 10)
    @test_throws "EMWorkspace needs a KalmanWorkspace" Siphon.EMWorkspace(dws)
    @test_throws "the simulation smoother needs a KalmanWorkspace" SimulationSmootherWorkspace(dws)
    @test_throws "the simulation smoother needs a KalmanWorkspace" simulation_smoother(
        dws, randn(1, 10))
end
