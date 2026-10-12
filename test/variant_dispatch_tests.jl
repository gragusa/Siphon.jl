# The filter variant is chosen by the argument types: EKFParms/EKFWorkspace for the
# extended Kalman filter, a KFParms of numbers for the scalar kernel, and a P1_inf
# argument (or a DiffuseKalmanWorkspace) for exact diffuse initialisation.

using Siphon
using Test
using LinearAlgebra
using Random

@testset "Variant names are gone" begin
    for name in (:ekf_loglik, :ekf_filter, :ekf_filter!, :ekf_smoother!,
        :ekf_filter_and_smooth!, :optimize_ekf, :kalman_loglik_scalar,
        :kalman_filter_scalar, :kalman_smoother_scalar, :KalmanFilterResultScalar,
        :kalman_filter_diffuse!)
        @test !isdefined(Siphon, name)
    end
end

@testset "Exact diffuse: functional and in-place agree through the generic names" begin
    rng = MersenneTwister(31)
    n = 40
    Z, H = [1.0 0.0], [0.8;;]
    Tm, R, Q = [1.0 1.0; 0.0 1.0], Matrix(1.0I, 2, 2), Diagonal([0.3, 0.05]) |> Matrix
    a1, P1_star, P1_inf = zeros(2), zeros(2, 2), Matrix(1.0I, 2, 2)
    y = cumsum(randn(rng, 1, n); dims = 2)
    y[1, 10] = NaN
    p = KFParms(Z, H, Tm, R, Q)

    r = kalman_filter(p, y, a1, P1_star, P1_inf)
    @test r isa DiffuseFilterResult
    @test kalman_loglik(p, y, a1, P1_star, P1_inf) ≈ loglikelihood(r)

    ws = DiffuseKalmanWorkspace(Z, H, Tm, R, Q, a1, P1_star, P1_inf, n)
    @test kalman_filter!(ws, y) ≈ loglikelihood(r)
    @test filtered_states(ws) ≈ filtered_states(r)
    @test_throws DimensionMismatch kalman_filter!(ws, y[:, 1:(n - 1)])
end

@testset "Scalar KFParms is AD-compatible and matches the 1 × 1 system" begin
    y = randn(MersenneTwister(32), 25)
    y[7] = NaN
    f(θ) = kalman_loglik(KFParms(1.0, θ[1], 0.9, 1.0, θ[2]), y, 0.0, 2.0)
    g(θ) = kalman_loglik(KFParms([1.0;;], [θ[1];;], [0.9;;], [1.0;;], [θ[2];;]),
        reshape(y, 1, :), [0.0], [2.0;;])
    θ = [0.7, 0.4]
    @test f(θ) ≈ g(θ)
    @test Siphon.ForwardDiff.gradient(f, θ) ≈ Siphon.ForwardDiff.gradient(g, θ)

    filt, sm = kalman_filter_and_smooth(KFParms(1.0, 0.7, 0.9, 1.0, 0.4), y, 0.0, 2.0)
    @test isnan(prediction_errors(filt)[1, 7])
    @test kalman_gains(filt)[1, 1, 7] == 0
    @test size(smoothed_states(sm)) == (1, 25)
end
