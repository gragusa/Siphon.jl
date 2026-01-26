"""
Tests for general EM algorithm implementation.
Validates against MARSS R package results for a model with estimable Z and T.
"""

using Test
using Siphon
using Siphon.DSL: _em_general_ssm_full_cov
using LinearAlgebra
using DelimitedFiles

@testset "General EM - MARSS Validation" begin
    # Load MARSS reference data
    data =
        readdlm(joinpath(@__DIR__, "marss_general_data.csv"), ',', Float64; header = true)[1]
    y = Matrix(data')  # Convert to p × n format (3 × 200)

    # Model dimensions
    p = 3  # observations
    m = 2  # states
    r = 2  # shocks

    # Initial values
    Z_init = [
        1.0 0.0;
        0.0 1.0;
        0.5 0.5
    ]  # Start with different values

    T_init = [
        0.5 0.0;
        0.0 0.5
    ]  # Start with smaller AR coefficients

    R = Matrix(1.0I, m, r)
    H_init = Matrix(Diagonal([1.0, 1.0, 1.0]))
    Q_init = Matrix(Diagonal([1.0, 1.0]))

    # Initial state - MARSS uses tinitx=0, meaning x0 is at time 0.
    # To match, we need to propagate: a1 = T*x0, P1 = T*V0*T' + R*Q*R'
    # With x0=0, V0=10*I, this gives a1=0 but P1 differs from V0.
    # For EM with initial T_init, we propagate accordingly:
    x0 = [0.0, 0.0]
    V0 = 10.0 * Matrix(1.0I, m, m)
    a1 = T_init * x0  # = [0, 0]
    P1 = T_init * V0 * T_init' + R * Q_init * R'

    # Free parameter masks
    # Z: first two rows fixed (identity), third row free
    Z_free = [
        false false;
        false false;
        true true
    ]

    # T: all elements free
    T_free = trues(m, m)

    # H, Q: only diagonal elements free (use diagonal-only BitMatrix)
    H_free = falses(p, p)
    H_free[1, 1] = true
    H_free[2, 2] = true
    H_free[3, 3] = true

    Q_free = falses(r, r)
    Q_free[1, 1] = true
    Q_free[2, 2] = true

    # Run EM
    result = _em_general_ssm_full_cov(
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
        maxiter = 2000,
        tol_ll = 1e-8,
        tol_param = 1e-6,
        verbose = false,
    )

    # MARSS reference values (from marss_general_results.csv)
    marss_B11 = 0.748645313949298
    marss_B12 = 0.125738889440894
    marss_B21 = 0.257176170000329
    marss_B22 = 0.480617263848418
    marss_Z31 = 0.654122240934409
    marss_Z32 = 0.36696428819377
    marss_Q11 = 0.581968763826898
    marss_Q22 = 1.10246726845249
    marss_R11 = 0.281575888766717
    marss_R22 = 0.205538099347484
    marss_R33 = 0.498644989174163
    marss_loglik = -828.495156092862

    println("\nJulia General EM Results:")
    println("=========================")
    println("T (transition matrix):")
    println("  T[1,1] = $(result.T[1,1]) (MARSS: $marss_B11)")
    println("  T[1,2] = $(result.T[1,2]) (MARSS: $marss_B12)")
    println("  T[2,1] = $(result.T[2,1]) (MARSS: $marss_B21)")
    println("  T[2,2] = $(result.T[2,2]) (MARSS: $marss_B22)")
    println("\nZ (observation matrix):")
    println("  Z[3,1] = $(result.Z[3,1]) (MARSS: $marss_Z31)")
    println("  Z[3,2] = $(result.Z[3,2]) (MARSS: $marss_Z32)")
    println("\nQ (state variances):")
    println("  Q[1,1] = $(result.Q[1,1]) (MARSS: $marss_Q11)")
    println("  Q[2,2] = $(result.Q[2,2]) (MARSS: $marss_Q22)")
    println("\nH (observation variances):")
    println("  H[1,1] = $(result.H[1,1]) (MARSS: $marss_R11)")
    println("  H[2,2] = $(result.H[2,2]) (MARSS: $marss_R22)")
    println("  H[3,3] = $(result.H[3,3]) (MARSS: $marss_R33)")
    println("\nloglik = $(result.loglik) (MARSS: $marss_loglik)")
    println("iterations = $(result.iterations)")
    println("converged = $(result.converged)")

    # Test against MARSS values (5% tolerance for parameters)
    @test isapprox(result.T[1, 1], marss_B11, rtol = 0.05)
    @test isapprox(result.T[1, 2], marss_B12, rtol = 0.05)
    @test isapprox(result.T[2, 1], marss_B21, rtol = 0.05)
    @test isapprox(result.T[2, 2], marss_B22, rtol = 0.05)
    @test isapprox(result.Z[3, 1], marss_Z31, rtol = 0.05)
    @test isapprox(result.Z[3, 2], marss_Z32, rtol = 0.05)
    @test isapprox(result.Q[1, 1], marss_Q11, rtol = 0.05)
    @test isapprox(result.Q[2, 2], marss_Q22, rtol = 0.05)
    @test isapprox(result.H[1, 1], marss_R11, rtol = 0.05)
    @test isapprox(result.H[2, 2], marss_R22, rtol = 0.05)
    @test isapprox(result.H[3, 3], marss_R33, rtol = 0.05)
    # Log-likelihood should match closely now that initial state is propagated correctly
    @test isapprox(result.loglik, marss_loglik, rtol = 0.001)

    # Siphon.jl should achieve at least as good a log-likelihood as MARSS
    # (higher is better for log-likelihood, i.e., less negative)
    @test result.loglik >= marss_loglik - 0.01  # Allow tiny numerical tolerance
end

@testset "General EM - Log-likelihood Improvement" begin
    # Load MARSS reference data
    data =
        readdlm(joinpath(@__DIR__, "marss_general_data.csv"), ',', Float64; header = true)[1]
    y = Matrix(data')

    p, m, r = 3, 2, 2

    Z_init = [1.0 0.0; 0.0 1.0; 0.5 0.5]
    T_init = [0.5 0.0; 0.0 0.5]
    R = Matrix(1.0I, m, r)
    H_init = Matrix(Diagonal([1.0, 1.0, 1.0]))
    Q_init = Matrix(Diagonal([1.0, 1.0]))

    # Propagate initial state (MARSS tinitx=0 convention)
    x0 = [0.0, 0.0]
    V0 = 10.0 * Matrix(1.0I, m, m)
    a1 = T_init * x0
    P1 = T_init * V0 * T_init' + R * Q_init * R'

    Z_free = [false false; false false; true true]
    T_free = trues(m, m)

    H_free = falses(p, p)
    H_free[1, 1] = true
    H_free[2, 2] = true
    H_free[3, 3] = true

    Q_free = falses(r, r)
    Q_free[1, 1] = true
    Q_free[2, 2] = true

    result = _em_general_ssm_full_cov(
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
        maxiter = 100,
        tol_ll = 1e-9,
        tol_param = 1e-6,
        verbose = false,
    )

    # EM should improve log-likelihood overall
    ll_first = result.loglik_history[1]
    ll_last = result.loglik_history[end]
    @test ll_last > ll_first

    # Count significant decreases (> 0.1 in log-likelihood)
    n_significant_decreases = 0
    for i = 2:length(result.loglik_history)
        if result.loglik_history[i] < result.loglik_history[i-1] - 0.1
            n_significant_decreases += 1
        end
    end
    @test n_significant_decreases == 0
end
