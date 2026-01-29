using Siphon
using Test
using LinearAlgebra
using CSV
using DataFrames
using ForwardDiff

# Load Nile data
nile = CSV.read(joinpath(@__DIR__, "Nile.csv"), DataFrame; header = false)
y = reshape(Float64.(nile[!, 1]), 1, :)  # 1 x 100 matrix

# MLE estimates from Durbin & Koopman (2012)
const Z_nile = [1.0;;]
const H_nile = [15099.0;;]
const T_nile = [1.0;;]
const R_nile = [1.0;;]
const Q_nile = [1469.1;;]
const a1_nile = [0.0]
const P1_nile = [1e7;;]

@testset "KFParms" begin
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    @test p.Z == Z_nile
    @test p.H == H_nile
    @test p.T == T_nile
    @test p.R == R_nile
    @test p.Q == Q_nile
    @test size(p) == (1, 1, 1)
end

@testset "kalman_loglik" begin
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    ll = kalman_loglik(p, y, a1_nile, P1_nile)

    # Log-likelihood should be finite and negative
    @test isfinite(ll)
    @test ll < 0

    # With approximate diffuse (P1=1e7), log-likelihood differs from exact diffuse
    # The value should be in a reasonable range for 100 observations
    @test ll > -700  # Not too small
    @test ll < -600  # Not too large
end

@testset "kalman_loglik AD" begin
    function ll_wrapper(theta)
        H = [theta[1];;]
        Q = [theta[2];;]
        p = KFParms(Z_nile, H, T_nile, R_nile, Q)
        return kalman_loglik(p, y, a1_nile, P1_nile)
    end

    theta = [15099.0, 1469.1]
    g = ForwardDiff.gradient(ll_wrapper, theta)

    @test length(g) == 2
    @test all(isfinite.(g))

    # Gradients should be near zero at MLE
    @test all(abs.(g) .< 1.0)
end

@testset "Exact vs Approximate Diffuse - Nile" begin
    # Test exact diffuse vs approximate diffuse for the Nile local level model
    # Reference values from R's KFAS package with same parameters:
    #   KFAS approximate (P1=1e7): -641.5856
    #   KFAS exact diffuse: -632.5456
    #   Difference: ~9.04

    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)

    # Exact diffuse initialization
    a1 = [0.0]
    P1_star = [0.0;;]   # No finite uncertainty
    P1_inf = [1.0;;]    # Full diffuse on level

    # Compute log-likelihoods
    ll_exact = kalman_loglik(p, y, a1, P1_star, P1_inf)
    ll_approx = kalman_loglik(p, y, a1, P1_nile)  # P1 = 1e7

    # Test that both give finite, negative log-likelihood
    @test isfinite(ll_exact)
    @test isfinite(ll_approx)
    @test ll_exact < 0
    @test ll_approx < 0

    # Match KFAS reference values (within tolerance)
    @test ll_approx ≈ -641.5856 rtol=1e-4
    @test ll_exact ≈ -632.5456 rtol=1e-4

    # The difference should be ~9.04 (exact is higher/less negative)
    @test (ll_approx - ll_exact) ≈ -9.04 atol=0.1

    # Test full filter output
    result_exact = kalman_filter(p, y, a1, P1_star, P1_inf)
    result_approx = kalman_filter(p, y, a1, P1_nile)

    # Check diffuse period (should be 1 for univariate local level)
    d = diffuse_period(result_exact)
    @test d == 1

    # Log-likelihoods should match the standalone function
    @test result_exact.loglik ≈ ll_exact

    # After diffuse period, filtered states should be similar
    for t in (d + 2):size(y, 2)
        @test result_exact.att[1, t] ≈ result_approx.att[1, t] rtol=0.01
    end

    # Test in-place version matches pure functional
    ws = DiffuseKalmanWorkspace(
        Z_nile,
        H_nile,
        T_nile,
        R_nile,
        Q_nile,
        a1,
        P1_star,
        P1_inf,
        size(y, 2)
    )
    ll_inplace = kalman_filter!(ws, y)
    @test ll_inplace ≈ ll_exact rtol=1e-10
    @test diffuse_period(ws) == d
end

@testset "MLE with Exact Diffuse - KFAS Validation" begin
    # Test that MLE with exact diffuse matches KFAS estimates
    # KFAS reference (from R):
    #   H = 15098.53, Q = 1469.178, loglik = -632.5456
    using Optimization, OptimizationOptimJL

    # Negative log-likelihood with exact diffuse
    function negloglik_diffuse(θ, y)
        H, Q = exp(θ[1]), exp(θ[2])
        p = KFParms([1.0;;], [H;;], [1.0;;], [1.0;;], [Q;;])
        a1 = [0.0]
        P1_star = [0.0;;]
        P1_inf = [1.0;;]
        return -kalman_loglik(p, y, a1, P1_star, P1_inf)
    end

    # Initial values
    y_var = sum((y .- sum(y)/length(y)) .^ 2) / (length(y) - 1)
    θ0 = [log(y_var/2), log(y_var/2)]

    # Optimize
    optf = OptimizationFunction(negloglik_diffuse, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, θ0, y)
    sol = solve(prob, LBFGS())

    H_mle = exp(sol.u[1])
    Q_mle = exp(sol.u[2])
    ll_mle = -sol.objective

    # Compare with KFAS reference values
    @test H_mle ≈ 15098.53 rtol=1e-3
    @test Q_mle ≈ 1469.178 rtol=1e-3
    @test ll_mle ≈ -632.5456 rtol=1e-4
end

@testset "kalman_loglik_scalar" begin
    ll = kalman_loglik_scalar(
        1.0, 15099.0, 1.0, 1.0, 1469.1, 0.0, 1e7, Float64.(nile[!, 1]))

    @test isfinite(ll)
    @test ll < 0

    # Should match matrix version
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    ll_mat = kalman_loglik(p, y, a1_nile, P1_nile)
    @test isapprox(ll, ll_mat, rtol = 1e-10)
end

@testset "kalman_filter" begin
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    result = kalman_filter(p, y, a1_nile, P1_nile)

    # Check struct type
    @test result isa Siphon.KalmanFilterResult

    # Check fields exist via hasproperty
    @test hasproperty(result, :loglik)
    @test hasproperty(result, :at)
    @test hasproperty(result, :Pt)
    @test hasproperty(result, :att)
    @test hasproperty(result, :Ptt)
    @test hasproperty(result, :vt)
    @test hasproperty(result, :Ft)
    @test hasproperty(result, :Kt)

    n = size(y, 2)
    # States have n entries (one per time point)
    @test size(result.at) == (1, n)
    @test size(result.Pt) == (1, 1, n)
    @test size(result.att) == (1, n)
    @test size(result.Ptt) == (1, 1, n)
    # Innovations have n entries (one per observation)
    @test size(result.vt) == (1, n)
    @test size(result.Ft) == (1, 1, n)

    # First predicted state is initial state
    @test result.at[1, 1] == a1_nile[1]
    @test result.Pt[1, 1, 1] == P1_nile[1, 1]

    # Log-likelihood should match kalman_loglik
    ll_direct = kalman_loglik(p, y, a1_nile, P1_nile)
    @test isapprox(result.loglik, ll_direct, rtol = 1e-10)

    # Test accessor methods
    @test Siphon.predicted_states(result) === result.at
    @test Siphon.filtered_states(result) === result.att
    @test Siphon.loglikelihood(result) == result.loglik
end

@testset "kalman_smoother" begin
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    result = kalman_filter(p, y, a1_nile, P1_nile)

    smooth = kalman_smoother(Z_nile, T_nile, result.at, result.Pt, result.vt, result.Ft)
    alpha_smooth = smooth.alpha
    V_smooth = smooth.V

    n = size(y, 2)
    @test size(alpha_smooth) == (1, n)
    @test size(V_smooth) == (1, 1, n)

    # Smoothed states should be finite
    @test all(isfinite.(alpha_smooth))

    # Smoothed variances should be positive
    @test all(V_smooth .> 0)

    # Smoothed variance should be smaller than predicted variance
    for t in 2:(n - 1)
        @test V_smooth[1, 1, t] <= result.Pt[1, 1, t]
    end

    # Reference value at t=100 from D&K Table 2.1
    @test isapprox(alpha_smooth[1, 100], 798.4, rtol = 0.01)
end

@testset "kalman_smoother AD" begin
    function smooth_sum(theta)
        H = [theta[1];;]
        Q = [theta[2];;]
        p = KFParms(Z_nile, H, T_nile, R_nile, Q)
        result = kalman_filter(p, y, a1_nile, P1_nile)
        smooth = kalman_smoother(Z_nile, T_nile, result.at, result.Pt, result.vt, result.Ft)
        return sum(smooth.alpha)
    end

    theta = [15099.0, 1469.1]
    g = ForwardDiff.gradient(smooth_sum, theta)

    @test length(g) == 2
    @test all(isfinite.(g))
end

@testset "kalman_filter_and_smooth" begin
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    result = kalman_filter_and_smooth(p, y, a1_nile, P1_nile)

    @test haskey(result, :loglik)
    @test haskey(result, :a_filtered)
    @test haskey(result, :P_filtered)
    @test haskey(result, :alpha_smooth)
    @test haskey(result, :V_smooth)

    # Compare with separate calls
    ll = kalman_loglik(p, y, a1_nile, P1_nile)
    @test isapprox(result.loglik, ll, rtol = 1e-10)
end

@testset "kalman_smoother_scalar" begin
    nile_vec = Float64.(nile[!, 1])
    result_scalar = kalman_filter_scalar(
        1.0, 15099.0, 1.0, 1.0, 1469.1, 0.0, 1e7, nile_vec)

    # Check struct type
    @test result_scalar isa Siphon.KalmanFilterResultScalar

    alpha,
    V = kalman_smoother_scalar(
        1.0,
        1.0,
        result_scalar.at,
        result_scalar.Pt,
        result_scalar.vt,
        result_scalar.Ft
    )

    n = length(nile_vec)
    @test length(alpha) == n
    @test length(V) == n

    # Should match matrix version
    p = KFParms(Z_nile, H_nile, T_nile, R_nile, Q_nile)
    result = kalman_filter(p, y, a1_nile, P1_nile)
    alpha_mat,
    V_mat = kalman_smoother(Z_nile, T_nile, result.at, result.Pt, result.vt, result.Ft)

    @test isapprox(alpha, vec(alpha_mat), rtol = 1e-10)
end

# DSL tests
include("dsl_tests.jl")

# EM algorithm tests (uses TestItems)
include("em_tests.jl")

# Generalized EM tests (MARSS validation - diagonal covariances)
include("em_generalized_tests.jl")

# General EM tests (MARSS validation - full Z and T estimation)
include("em_general_tests.jl")

# FKF validation tests (R FKF package comparison)
include("fkf_validation_tests.jl")

# Plotting recipe tests
include("recipe_tests.jl")

# ARMA model tests
include("arma_tests.jl")

# Exact diffuse filter tests
include("diffuse_tests.jl")

# Aqua.jl quality assurance tests
include("Aqua.jl")
