"""
Tests for EM algorithm implementation.
"""

using Test
using Siphon
using LinearAlgebra
using DelimitedFiles
using ForwardDiff
using Random

@testset "kalman_smoother basic" begin
    # Simple local level model
    Z = [1.0;;]
    H = [100.0;;]
    T = [1.0;;]
    R = [1.0;;]
    Q = [50.0;;]
    a1 = [0.0]
    P1 = [1e7;;]

    p = KFParms(Z, H, T, R, Q)

    # Generate some data
    Random.seed!(42)
    y = randn(1, 30) .* 10 .+ 100

    filt = kalman_filter(p, y, a1, P1)

    # Without cross-covariances
    result1 = kalman_smoother(
        Z, T, filt.at, filt.Pt, filt.vt, filt.Ft; compute_crosscov = false)
    @test haskey(result1, :alpha)
    @test haskey(result1, :V)
    @test size(result1.alpha) == (1, 30)
    @test size(result1.V) == (1, 1, 30)

    # With cross-covariances
    result2 = kalman_smoother(
        Z, T, filt.at, filt.Pt, filt.vt, filt.Ft; compute_crosscov = true)
    @test haskey(result2, :P_crosslag)
    @test size(result2.P_crosslag) == (1, 1, 29)  # n-1 cross-covariances

    # Cross-covariances should be positive for local level
    @test all(result2.P_crosslag .> 0)
end

@testset "fit!(EM(), ...) local_level basic" begin
    spec = local_level()

    # Generate data from known parameters
    true_var_obs = 100.0
    true_var_level = 25.0
    n = 200

    # Simulate local level model
    Random.seed!(123)
    states = zeros(n)
    y_data = zeros(1, n)
    states[1] = 0.0
    for t in 1:n
        if t > 1
            states[t] = states[t - 1] + sqrt(true_var_level) * randn()
        end
        y_data[1, t] = states[t] + sqrt(true_var_obs) * randn()
    end

    # Use unified fit!(EM(), ...) API
    model = StateSpaceModel(spec, n)
    fit!(EM(), model, y_data; maxiter = 100, verbose = false)

    @test isfinite(loglikelihood(model))
end

@testset "fit!(EM(), ...) Nile data" begin
    # Load Nile data
    nile = readdlm(joinpath(@__DIR__, "Nile.csv"), ',', Float64)
    y = reshape(nile[:, 1], 1, :)

    spec = local_level()
    model = StateSpaceModel(spec, size(y, 2))
    fit!(EM(), model, y; maxiter = 300, tol = 1e-8)

    # Get parameter names and values
    names = param_names(spec)
    var_obs_idx = findfirst(==(:var_obs), names)
    var_level_idx = findfirst(==(:var_level), names)

    # Compare with known MLE from Durbin & Koopman
    # var_obs ≈ 15099, var_level ≈ 1469.1
    # Note: fit!(EM(), ...) uses a different code path than _em_local_level
    # so we use slightly higher tolerance
    @test isapprox(model.theta_values[var_obs_idx], 15099.0, rtol = 0.10)
    @test isapprox(model.theta_values[var_level_idx], 1469.1, rtol = 0.10)
end

@testset "_em_local_level log-likelihood monotonicity" begin
    spec = local_level()
    Random.seed!(456)
    y = randn(1, 50) .* 10 .+ 100

    # Use internal function directly for testing
    result = Siphon.DSL._em_local_level(spec, y; maxiter = 50, verbose = false)

    # Log-likelihood should be monotonically non-decreasing
    for i in 2:length(result.loglik_history)
        @test result.loglik_history[i] >= result.loglik_history[i - 1] - 1e-8
    end
end

@testset "_em_local_level finds MLE" begin
    # EM should find the maximum likelihood estimate
    # Test on Nile data where the MLE is known from Durbin & Koopman (2012)
    nile = readdlm(joinpath(@__DIR__, "Nile.csv"), ',', Float64)
    y = reshape(nile[:, 1], 1, :)

    spec = local_level()
    result_em = Siphon.DSL._em_local_level(spec, y; maxiter = 300, tol_ll = 1e-8)

    @test result_em.converged

    # EM should match D&K reference MLE values
    # var_obs ≈ 15099, var_level ≈ 1469.1
    @test isapprox(result_em.theta.var_obs, 15099.0, rtol = 0.01)
    @test isapprox(result_em.theta.var_level, 1469.1, rtol = 0.01)

    # Log-likelihood should match D&K value
    # ll ≈ -641.59
    @test isapprox(result_em.loglik, -641.59, rtol = 0.001)
end

@testset "_em_local_level with fixed var_obs" begin
    # Fix var_obs, estimate var_level only
    spec_fixed = local_level(var_obs = 15099.0)

    nile = readdlm(joinpath(@__DIR__, "Nile.csv"), ',', Float64)
    y = reshape(nile[:, 1], 1, :)

    result = Siphon.DSL._em_local_level(spec_fixed, y; maxiter = 300, tol_ll = 1e-8)

    @test result.converged
    @test length(param_names(spec_fixed)) == 1
    @test :var_level in param_names(spec_fixed)

    # Should recover var_level ≈ 1469.1
    @test isapprox(result.theta.var_level, 1469.1, rtol = 0.1)
end

@testset "_em_local_level with fixed var_level" begin
    # Fix var_level, estimate var_obs only
    spec_fixed = local_level(var_level = 1469.0)

    nile = readdlm(joinpath(@__DIR__, "Nile.csv"), ',', Float64)
    y = reshape(nile[:, 1], 1, :)

    result = Siphon.DSL._em_local_level(spec_fixed, y; maxiter = 300, tol_ll = 1e-8)

    @test result.converged
    @test length(param_names(spec_fixed)) == 1
    @test :var_obs in param_names(spec_fixed)

    # Should recover var_obs ≈ 15099
    @test isapprox(result.theta.var_obs, 15099.0, rtol = 0.1)
end

@testset "EMResult fields" begin
    spec = local_level()
    Random.seed!(111)
    y = randn(1, 50) .* 10 .+ 100

    # Use internal function to get EMResult
    result = Siphon.DSL._em_local_level(spec, y; maxiter = 50)

    # Check all fields exist and have correct types
    @test result.theta isa NamedTuple
    @test result.theta_vec isa Vector{Float64}
    @test result.loglik isa Float64
    @test result.loglik_history isa Vector{Float64}
    @test result.converged isa Bool
    @test result.iterations isa Int
    @test result.smoothed_states isa Matrix{Float64}
    @test result.smoothed_cov isa Array{Float64, 3}

    # Check dimensions
    @test size(result.smoothed_states) == (1, 50)
    @test size(result.smoothed_cov) == (1, 1, 50)
    @test length(result.theta_vec) == length(param_names(spec))
end

@testset "kalman_smoother AD compatibility" begin
    Z = [1.0;;]
    T = [1.0;;]
    a1 = [0.0]
    P1 = [1e7;;]
    Random.seed!(222)
    y = randn(1, 20) .* 10 .+ 100

    function test_func(theta)
        H = [theta[1];;]
        Q = [theta[2];;]
        R = [1.0;;]
        p = KFParms(Z, H, T, R, Q)

        filt = kalman_filter(p, y, a1, P1)
        smooth = kalman_smoother(
            Z,
            T,
            filt.at,
            filt.Pt,
            filt.vt,
            filt.Ft;
            compute_crosscov = true
        )

        # Sum of smoothed states and cross-covariances
        return sum(smooth.alpha) + sum(smooth.P_crosslag)
    end

    theta = [100.0, 50.0]
    g = ForwardDiff.gradient(test_func, theta)

    @test length(g) == 2
    @test all(isfinite.(g))
end

# ============================================
# Full Covariance EM Tests
# ============================================

@testset "_em_general_ssm_full_cov basic bivariate" begin
    # Simple bivariate model: 2 observations, 2 states
    # yₜ = Z αₜ + εₜ, εₜ ~ N(0, H)
    # αₜ₊₁ = T αₜ + ηₜ, ηₜ ~ N(0, Q)

    Random.seed!(42)

    # True parameters
    Z_true = [1.0 0.0; 0.0 1.0]
    T_true = [0.9 0.0; 0.0 0.8]
    R = [1.0 0.0; 0.0 1.0]
    H_true = [1.0 0.3; 0.3 1.5]  # Correlated obs noise
    Q_true = [0.5 0.1; 0.1 0.4]  # Correlated state noise

    n = 200
    m = 2
    p = 2

    # Simulate data
    states = zeros(m, n)
    y = zeros(p, n)

    # Cholesky for sampling
    L_Q = cholesky(Symmetric(Q_true)).L
    L_H = cholesky(Symmetric(H_true)).L

    states[:, 1] = zeros(m)
    for t in 1:n
        if t > 1
            states[:, t] = T_true * states[:, t - 1] + L_Q * randn(m)
        end
        y[:, t] = Z_true * states[:, t] + L_H * randn(p)
    end

    # Initial values (different from true)
    Z_init = [1.0 0.0; 0.0 1.0]
    T_init = [0.5 0.0; 0.0 0.5]
    H_init = [2.0 0.0; 0.0 2.0]
    Q_init = [1.0 0.0; 0.0 1.0]
    a1 = zeros(m)
    P1 = 10.0 * Matrix{Float64}(I, m, m)

    # Fix Z and estimate T, H, Q
    Z_free = falses(p, m)
    T_free = trues(m, m)
    H_free = trues(p, p)
    Q_free = trues(m, m)

    result = Siphon.DSL._em_general_ssm_full_cov(
        Z_init,
        T_init,
        R,
        H_init,
        Q_init,
        y,
        a1,
        P1;
        Z_free = Z_free,
        T_free = T_free,
        H_free = H_free,
        Q_free = Q_free,
        maxiter = 500,
        tol_ll = 1e-5,
        verbose = false
    )

    # May not fully converge but should make progress
    @test result.iterations > 1
    @test isfinite(result.loglik)

    # Check H is positive definite
    @test all(eigvals(Symmetric(result.H)) .> 0)
    # Check Q is positive definite
    @test all(eigvals(Symmetric(result.Q)) .> 0)

    # Check H is symmetric
    @test result.H ≈ result.H' atol=1e-10

    # Check Q is symmetric
    @test result.Q ≈ result.Q' atol=1e-10

    # Estimated parameters should be in reasonable range
    # (not exact due to finite sample)
    @test 0.5 < result.T[1, 1] < 1.0
    @test 0.5 < result.T[2, 2] < 1.0
end

@testset "_em_general_ssm_full_cov log-likelihood monotonicity" begin
    Random.seed!(123)

    # Simple 2x2 model
    Z = [1.0 0.0; 0.0 1.0]
    T = [0.8 0.0; 0.0 0.7]
    R = [1.0 0.0; 0.0 1.0]
    H = [1.0 0.2; 0.2 1.0]
    Q = [0.5 0.0; 0.0 0.5]
    a1 = [0.0, 0.0]
    P1 = 10.0 * Matrix{Float64}(I, 2, 2)

    y = randn(2, 50)

    result = Siphon.DSL._em_general_ssm_full_cov(
        Z,
        T,
        R,
        H,
        Q,
        y,
        a1,
        P1;
        maxiter = 50,
        verbose = false
    )

    # Log-likelihood should be monotonically non-decreasing
    for i in 2:length(result.loglik_history)
        @test result.loglik_history[i] >= result.loglik_history[i - 1] - 1e-6
    end
end

@testset "_em_general_ssm_full_cov fixed parameters" begin
    Random.seed!(456)

    # Test with some parameters fixed
    Z = [1.0 0.0; 0.0 1.0]
    T = [0.9 0.0; 0.0 0.8]
    R = [1.0 0.0; 0.0 1.0]
    H_init = [1.0 0.0; 0.0 1.0]
    Q_init = [0.5 0.0; 0.0 0.5]
    a1 = [0.0, 0.0]
    P1 = 10.0 * Matrix{Float64}(I, 2, 2)

    y = randn(2, 100)

    # Fix Z and T, only estimate H and Q
    Z_free = falses(2, 2)
    T_free = falses(2, 2)
    H_free = trues(2, 2)
    Q_free = trues(2, 2)

    result = Siphon.DSL._em_general_ssm_full_cov(
        Z,
        T,
        R,
        H_init,
        Q_init,
        y,
        a1,
        P1;
        Z_free = Z_free,
        T_free = T_free,
        H_free = H_free,
        Q_free = Q_free,
        maxiter = 100,
        verbose = false
    )

    # Z and T should remain unchanged
    @test result.Z ≈ Z
    @test result.T ≈ T

    # H and Q should be estimated (and different from init)
    @test !(result.H ≈ H_init)
    @test !(result.Q ≈ Q_init)
end

@testset "project_psd" begin
    # Test that project_psd produces positive definite matrices

    # Already PSD matrix
    A = [2.0 1.0; 1.0 2.0]
    A_proj = Siphon.DSL.project_psd(A)
    @test all(eigvals(A_proj) .> 0)
    @test A_proj ≈ A atol=1e-8

    # Matrix with negative eigenvalue
    B = [1.0 2.0; 2.0 1.0]  # eigenvalues: 3, -1
    B_proj = Siphon.DSL.project_psd(B)
    @test all(eigvals(B_proj) .> 0)
    @test B_proj ≈ B_proj' atol=1e-10  # symmetric

    # Zero matrix should become positive definite
    C = zeros(2, 2)
    C_proj = Siphon.DSL.project_psd(C)
    @test all(eigvals(C_proj) .>= 1e-10)
end

@testset "em_ssm_diagonal matches full_cov with diagonal constraints" begin
    # Test that em_ssm_diagonal convenience wrapper produces
    # the same results as _em_general_ssm_full_cov with diagonal constraints
    Random.seed!(42)

    # Setup: simple 2-state, 2-observation model
    p, m, n = 2, 2, 100
    r = 2

    # Initial parameters
    Z_init = [1.0 0.0; 0.0 1.0]
    T_init = [0.8 0.0; 0.0 0.7]
    R = [1.0 0.0; 0.0 1.0]
    H_init_diag = [1.0, 1.5]
    Q_init_diag = [0.5, 0.8]
    a1 = zeros(m)
    P1 = 10.0 * Matrix{Float64}(I, m, m)

    # Generate data
    y = randn(p, n)

    # Free masks
    Z_free = trues(p, m)
    T_free = trues(m, m)
    H_free_diag = trues(p)
    Q_free_diag = trues(r)

    # Method 1: em_ssm_diagonal wrapper
    result_wrapper = Siphon.DSL.em_ssm_diagonal(
        Z_init,
        T_init,
        R,
        H_init_diag,
        Q_init_diag,
        y,
        a1,
        P1;
        Z_free = Z_free,
        T_free = T_free,
        H_free = H_free_diag,
        Q_free = Q_free_diag,
        maxiter = 200,
        tol_ll = 1e-8,
        verbose = false
    )

    # Method 2: _em_general_ssm_full_cov with diagonal constraints
    H_init_full = Diagonal(H_init_diag) |> Matrix
    Q_init_full = Diagonal(Q_init_diag) |> Matrix

    # Only diagonal elements are free
    H_free_full = falses(p, p)
    H_free_full[1, 1] = true
    H_free_full[2, 2] = true

    Q_free_full = falses(r, r)
    Q_free_full[1, 1] = true
    Q_free_full[2, 2] = true

    result_full = Siphon.DSL._em_general_ssm_full_cov(
        Z_init,
        T_init,
        R,
        H_init_full,
        Q_init_full,
        y,
        a1,
        P1;
        Z_free = Z_free,
        T_free = T_free,
        H_free = H_free_full,
        Q_free = Q_free_full,
        maxiter = 200,
        tol_ll = 1e-8,
        verbose = false
    )

    # Both methods should produce the same results
    @test maximum(abs.(result_wrapper.Z - result_full.Z)) < 1e-10
    @test maximum(abs.(result_wrapper.T - result_full.T)) < 1e-10
    @test maximum(abs.(result_wrapper.H_diag - diag(result_full.H))) < 1e-10
    @test maximum(abs.(result_wrapper.Q_diag - diag(result_full.Q))) < 1e-10
    @test abs(result_wrapper.loglik - result_full.loglik) < 1e-10
    # Off-diagonals should be exactly zero (they were fixed)
    @test result_full.H[1, 2] == 0.0
    @test result_full.Q[1, 2] == 0.0
end

@testset "_mstep_full_cov produces symmetric matrices" begin
    Random.seed!(789)

    # Setup test data
    p, m, n = 2, 2, 50
    r = 2

    Z = [1.0 0.0; 0.0 1.0]
    T = [0.9 0.0; 0.0 0.8]
    R = [1.0 0.0; 0.0 1.0]
    H = [1.0 0.3; 0.3 1.0]
    Q = [0.5 0.1; 0.1 0.5]
    a1 = zeros(m)
    P1 = 10.0 * Matrix{Float64}(I, m, m)

    y = randn(p, n)

    # Run filter and smoother
    kfp = KFParms(Z, H, T, R, Q)
    filt = kalman_filter(kfp, y, a1, P1)
    smooth = kalman_smoother(
        Z, T, filt.at, filt.Pt, filt.vt, filt.Ft; compute_crosscov = true)

    # Run M-step
    Z_new, T_new,
    H_new,
    Q_new = Siphon.DSL._mstep_full_cov(
        Z, T, R, y, smooth.alpha, smooth.V, smooth.P_crosslag)

    # Check symmetry
    @test H_new ≈ H_new' atol=1e-10
    @test Q_new ≈ Q_new' atol=1e-10

    # Check dimensions
    @test size(H_new) == (p, p)
    @test size(Q_new) == (r, r)
    @test size(Z_new) == (p, m)
    @test size(T_new) == (m, m)
end

# ============================================
# DynamicFactorModel Identification Tests
# ============================================

# TODO: These tests require the `identification` keyword argument which is not yet implemented
# @testset "DynamicFactorModel identification validation" begin
#     # n_obs < n_factors should error for identified models
#     @test_throws ArgumentError DynamicFactorModel(2, 5, 100; identification = :named_factor)
#     @test_throws ArgumentError DynamicFactorModel(
#         2,
#         5,
#         100;
#         identification = :lower_triangular,
#     )
#
#     # But :none should work (no identification constraints)
#     @test_nowarn DynamicFactorModel(2, 5, 100; identification = :none)
#
#     # Invalid identification scheme
#     @test_throws ArgumentError DynamicFactorModel(10, 2, 100; identification = :invalid)
#
#     # Valid configurations
#     @test_nowarn DynamicFactorModel(10, 2, 100; identification = :named_factor)
#     @test_nowarn DynamicFactorModel(10, 2, 100; identification = :lower_triangular)
#     @test_nowarn DynamicFactorModel(10, 2, 100; identification = :none)
# end

# TODO: These tests require the `identification` keyword argument which is not yet implemented
# @testset "DynamicFactorModel identification initialization" begin
#     # Test named_factor identification initialization
#     model_nf = DynamicFactorModel(10, 3, 100; identification = :named_factor)
#     Z = model_nf.kf_ws.Z
#
#     # Check identity block in first k rows
#     @test Z[1, 1] ≈ 1.0
#     @test Z[1, 2] ≈ 0.0
#     @test Z[1, 3] ≈ 0.0
#     @test Z[2, 2] ≈ 1.0
#     @test Z[2, 3] ≈ 0.0
#     @test Z[3, 3] ≈ 1.0
#
#     # Check Z_free mask
#     Z_free = model_nf.em_ws.Z_free
#     @test Z_free[1, 1] == false  # diagonal fixed
#     @test Z_free[1, 2] == false  # upper triangle fixed
#     @test Z_free[2, 1] == true   # lower triangle free
#     @test Z_free[2, 2] == false  # diagonal fixed
#     @test Z_free[3, 1] == true   # lower triangle free
#     @test Z_free[3, 2] == true   # lower triangle free
#     @test Z_free[3, 3] == false  # diagonal fixed
#     @test Z_free[4, 1] == true   # row > k, all free
#     @test Z_free[4, 2] == true
#     @test Z_free[4, 3] == true
#
#     # Test lower_triangular identification initialization (Harvey 1989)
#     model_lt = DynamicFactorModel(10, 3, 100; identification = :lower_triangular)
#     Z_lt = model_lt.kf_ws.Z
#     Z_free_lt = model_lt.em_ws.Z_free
#     Q_lt = model_lt.kf_ws.Q
#     Q_free_lt = model_lt.em_ws.Q_free
#
#     # Upper triangle = 0 (fixed)
#     @test Z_lt[1, 2] ≈ 0.0
#     @test Z_lt[1, 3] ≈ 0.0
#     @test Z_lt[2, 3] ≈ 0.0
#
#     # Diagonal is FREE (not fixed to 1)
#     @test Z_free_lt[1, 1] == true   # diagonal free
#     @test Z_free_lt[2, 2] == true   # diagonal free
#     @test Z_free_lt[3, 3] == true   # diagonal free
#     @test Z_free_lt[1, 2] == false  # upper triangle fixed
#     @test Z_free_lt[1, 3] == false  # upper triangle fixed
#     @test Z_free_lt[2, 3] == false  # upper triangle fixed
#     @test Z_free_lt[2, 1] == true   # lower triangle free
#
#     # Q factor block = I (fixed)
#     k = 3
#     @test Q_lt[1, 1] ≈ 1.0
#     @test Q_lt[2, 2] ≈ 1.0
#     @test Q_lt[3, 3] ≈ 1.0
#     @test Q_free_lt[1, 1] == false  # Q factor block is fixed
#     @test Q_free_lt[2, 2] == false
#     @test Q_free_lt[3, 3] == false
#
#     # Test :none identification (all free)
#     model_none = DynamicFactorModel(10, 3, 100; identification = :none)
#     Z_free_none = model_none.em_ws.Z_free
#
#     # All factor loadings should be free
#     @test all(Z_free_none[1:10, 1:3])
# end

# TODO: These tests require the `identification` keyword argument which is not yet implemented
# @testset "DynamicFactorModel identification constraints preserved" begin
#     Random.seed!(42)
#
#     # Generate simple factor model data
#     N, k, n_time = 10, 2, 100
#     true_factors = randn(k, n_time)
#     true_loadings = randn(N, k)
#     noise = randn(N, n_time) * 0.5
#     y = true_loadings * true_factors + noise
#
#     # Fit with named_factor identification
#     model = DynamicFactorModel(N, k, n_time; identification = :named_factor)
#     fit!(EM(), model, y; maxiter = 50, verbose = false)
#
#     # Check constraints are preserved after fitting
#     Λ = loadings(model)
#     Λ₀ = Λ[1]  # Contemporaneous loadings
#
#     @test Λ₀[1, 1] ≈ 1.0 atol=1e-10
#     @test Λ₀[1, 2] ≈ 0.0 atol=1e-10
#     @test Λ₀[2, 2] ≈ 1.0 atol=1e-10
#
#     # Lower triangle should be estimated (not necessarily 0)
#     # Just check they exist and are finite
#     @test isfinite(Λ₀[2, 1])
#     @test isfinite(Λ₀[3, 1])
#     @test isfinite(Λ₀[3, 2])
#
#     # Fit with lower_triangular identification (Harvey 1989)
#     model_lt = DynamicFactorModel(N, k, n_time; identification = :lower_triangular)
#     fit!(EM(), model_lt, y; maxiter = 50, verbose = false)
#
#     Λ_lt = loadings(model_lt)[1]
#     # Upper triangle should be 0 (fixed)
#     @test Λ_lt[1, 2] ≈ 0.0 atol=1e-10
#
#     # Diagonal is FREE, so should be estimated (not necessarily 1)
#     @test isfinite(Λ_lt[1, 1])
#     @test isfinite(Λ_lt[2, 2])
#
#     # Q should remain at identity
#     Q_lt = model_lt.kf_ws.Q
#     @test Q_lt[1, 1] ≈ 1.0 atol=1e-10
#     @test Q_lt[2, 2] ≈ 1.0 atol=1e-10
# end

# TODO: These tests require the `identification` keyword argument which is not yet implemented
# @testset "DynamicFactorModel identification schemes differ" begin
#     Random.seed!(123)
#
#     # Generate data
#     N, k, n_time = 10, 2, 100
#     true_factors = randn(k, n_time)
#     true_loadings = randn(N, k)
#     noise = randn(N, n_time) * 0.5
#     y = true_loadings * true_factors + noise
#
#     # Fit with different identification schemes
#     model_nf = DynamicFactorModel(N, k, n_time; identification = :named_factor)
#     model_none = DynamicFactorModel(N, k, n_time; identification = :none)
#
#     fit!(EM(), model_nf, y; maxiter = 50, verbose = false)
#     fit!(EM(), model_none, y; maxiter = 50, verbose = false)
#
#     # Loadings should differ (up to rotation for :none)
#     Λ_nf = loadings(model_nf)[1]
#     Λ_none = loadings(model_none)[1]
#
#     # The identified model should have the identity block
#     @test Λ_nf[1, 1] ≈ 1.0 atol=1e-10
#     @test Λ_nf[1, 2] ≈ 0.0 atol=1e-10
#
#     # The unidentified model likely does not
#     @test !(Λ_none[1, 1] ≈ 1.0 && Λ_none[1, 2] ≈ 0.0)
# end
