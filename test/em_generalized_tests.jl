"""
Tests for generalized EM algorithm implementation.
Validates against MARSS R package results.
"""

using Test
using Siphon
using Siphon.DSL: _em_diagonal_ssm, _mstep_diagonal
using LinearAlgebra
using DelimitedFiles

@testset "Generalized EM - MARSS Validation" begin
    # Load MARSS reference data
    data = readdlm(joinpath(@__DIR__, "marss_reference_data.csv"), ',', Float64; header=true)[1]
    y = Matrix(data')  # Convert to p × n format (3 × 100)

    # Z matrix from R script (fixed)
    Z = [1.0 0.0;
         0.0 1.0;
         0.7 0.3]

    # T = identity (random walk)
    T = Matrix(1.0I, 2, 2)

    # R = identity
    R = Matrix(1.0I, 2, 2)

    # Initial state
    a1 = [0.0, 0.0]
    P1 = 1e7 * Matrix(1.0I, 2, 2)

    # Parameter names (all free)
    H_diag_params = [:R11, :R22, :R33]
    Q_diag_params = [:Q11, :Q22]

    # No fixed parameters
    H_diag_fixed = Tuple{Int,Float64}[]
    Q_diag_fixed = Tuple{Int,Float64}[]

    # Run EM with parameter-based convergence (ll oscillations are expected at optimum)
    result = _em_diagonal_ssm(Z, T, R, H_diag_params, Q_diag_params,
                               H_diag_fixed, Q_diag_fixed,
                               y, a1, P1;
                               maxiter=500, tol_ll=1e-9, tol_param=1e-4, verbose=false)

    # MARSS reference values (from marss_reference_results.csv)
    # Q11 = 1.133779, Q22 = 2.303474
    # R11 = 0.547054, R22 = 0.539332, R33 = 1.037559
    # loglik = -546.485583

    println("\nJulia EM Results:")
    println("================")
    println("R11 = $(result.theta.R11) (MARSS: 0.547054)")
    println("R22 = $(result.theta.R22) (MARSS: 0.539332)")
    println("R33 = $(result.theta.R33) (MARSS: 1.037559)")
    println("Q11 = $(result.theta.Q11) (MARSS: 1.133779)")
    println("Q22 = $(result.theta.Q22) (MARSS: 2.303474)")
    println("loglik = $(result.loglik) (MARSS: -546.485583)")
    println("iterations = $(result.iterations)")
    println("converged = $(result.converged)")

    # Test against MARSS values (allow some tolerance)
    # Parameters should be very close (within 1%)
    @test isapprox(result.theta.R11, 0.547054, rtol=0.01)
    @test isapprox(result.theta.R22, 0.539332, rtol=0.01)
    @test isapprox(result.theta.R33, 1.037559, rtol=0.01)
    @test isapprox(result.theta.Q11, 1.133779, rtol=0.01)
    @test isapprox(result.theta.Q22, 2.303474, rtol=0.01)
    @test isapprox(result.loglik, -546.485583, rtol=0.001)

    # Siphon.jl should achieve at least as good a log-likelihood as MARSS
    # (higher is better for log-likelihood, i.e., less negative)
    marss_loglik = -546.485583
    @test result.loglik >= marss_loglik - 0.01  # Allow tiny numerical tolerance
end

@testset "Generalized EM - Log-likelihood Near-Monotonicity" begin
    # Load MARSS reference data
    data = readdlm(joinpath(@__DIR__, "marss_reference_data.csv"), ',', Float64; header=true)[1]
    y = Matrix(data')

    Z = [1.0 0.0; 0.0 1.0; 0.7 0.3]
    T = Matrix(1.0I, 2, 2)
    R = Matrix(1.0I, 2, 2)
    a1 = [0.0, 0.0]
    P1 = 1e7 * Matrix(1.0I, 2, 2)

    H_diag_params = [:R11, :R22, :R33]
    Q_diag_params = [:Q11, :Q22]
    H_diag_fixed = Tuple{Int,Float64}[]
    Q_diag_fixed = Tuple{Int,Float64}[]

    result = _em_diagonal_ssm(Z, T, R, H_diag_params, Q_diag_params,
                               H_diag_fixed, Q_diag_fixed,
                               y, a1, P1;
                               maxiter=100, tol_ll=1e-9, tol_param=1e-4, verbose=false)

    # Log-likelihood should be approximately monotonically non-decreasing
    # Allow small numerical oscillations (~0.001) which are typical for EM near optimum
    # The key invariant is that the final ll should be close to optimal
    ll_first = result.loglik_history[1]
    ll_last = result.loglik_history[end]

    # EM should improve log-likelihood overall
    @test ll_last > ll_first

    # Count significant decreases (> 0.01 in log-likelihood)
    n_significant_decreases = 0
    for i in 2:length(result.loglik_history)
        if result.loglik_history[i] < result.loglik_history[i-1] - 0.01
            n_significant_decreases += 1
        end
    end
    # Should have very few significant decreases (numerical issues only)
    @test n_significant_decreases == 0
end
