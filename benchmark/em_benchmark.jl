"""
    em_benchmark.jl

Benchmark comparison of Julia (Siphon.jl) EM vs R (MARSS) EM algorithm
on the same underlying data using RCall.

Run with: julia --project benchmark/em_benchmark.jl
"""

using Siphon
using Siphon.DSL: _em_general_ssm
using LinearAlgebra
using Statistics
using Printf
using RCall

# ============================================
# Data Generation (in R for exact MARSS compatibility)
# ============================================

function generate_data_in_R(; n_obs = 200, seed = 54321)
    R"""
    set.seed($seed)

    # Model parameters
    m <- 2  # states
    p <- 3  # observations
    n <- $n_obs

    # True transition matrix B
    B_true <- matrix(c(0.8, 0.1, 0.2, 0.7), nrow=2, ncol=2, byrow=TRUE)

    # True observation matrix Z (with identification constraints)
    Z_true <- matrix(c(1.0, 0.0, 0.0, 1.0, 0.6, 0.4), nrow=3, ncol=2, byrow=TRUE)

    # True variances
    Q_true <- diag(c(0.5, 0.8))
    R_true <- diag(c(0.3, 0.4, 0.5))

    # Generate states
    states <- matrix(0, nrow=m, ncol=n)
    states[, 1] <- c(0, 0)
    for (t in 2:n) {
        states[, t] <- B_true %*% states[, t-1] +
                       rnorm(m, 0, sqrt(c(0.5, 0.8)))
    }

    # Generate observations
    y <- matrix(0, nrow=p, ncol=n)
    for (t in 1:n) {
        y[, t] <- Z_true %*% states[, t] +
                  rnorm(p, 0, sqrt(c(0.3, 0.4, 0.5)))
    }

    list(y=y, B_true=B_true, Z_true=Z_true, Q_true=Q_true, R_true=R_true,
         states=states, m=m, p=p, n=n)
    """
end

# ============================================
# MARSS EM (R)
# ============================================

function run_marss_em(y_r; maxiter = 100, tol = 1e-8, silent = true)
    # Note: MARSS may not converge with very tight tolerances in few iterations
    # We set allow.degen=FALSE to avoid degenerate solutions
    R"""
    suppressPackageStartupMessages(library(MARSS))

    y <- $y_r
    m <- 2
    p <- 3

    B_model <- matrix(list("b11", "b12", "b21", "b22"), nrow=2, ncol=2, byrow=TRUE)
    Z_model <- matrix(list(1, 0, 0, 1, "z31", "z32"), nrow=3, ncol=2, byrow=TRUE)

    model_list <- list(
        B = B_model,
        U = matrix(0, m, 1),
        Q = "diagonal and unequal",
        Z = Z_model,
        A = matrix(0, p, 1),
        R = "diagonal and unequal",
        x0 = matrix(0, m, 1),
        V0 = diag(10, m),
        tinitx = 0
    )

    # Suppress all warnings and output during fit
    fit <- suppressWarnings(suppressMessages(
        MARSS(y, model = model_list, method = "kem",
              control = list(maxit = $maxiter,
                            conv.test.slope.tol = $tol,
                            abstol = $tol,
                            allow.degen = FALSE),
              silent = TRUE)
    ))

    # Check if fit was successful
    if (is.null(fit$par)) {
        list(loglik = NA, iterations = $maxiter, converged = FALSE,
             B = matrix(NA, 2, 2), Z = matrix(NA, 3, 2),
             Q = matrix(NA, 2, 2), R = matrix(NA, 3, 3))
    } else {
        B_est <- coef(fit, type="matrix")$B
        Z_est <- coef(fit, type="matrix")$Z
        Q_est <- coef(fit, type="matrix")$Q
        R_est <- coef(fit, type="matrix")$R

        list(loglik = fit$logLik,
             iterations = fit$numIter,
             converged = (fit$convergence == 0),
             B = B_est, Z = Z_est, Q = Q_est, R = R_est)
    }
    """
end

# ============================================
# Julia EM
# ============================================

function run_julia_em(y::Matrix{Float64}; maxiter = 100, tol_ll = 1e-8, tol_param = 1e-6)
    p, n = size(y)
    m, r = 2, 2

    # Initial values
    Z_init = [1.0 0.0; 0.0 1.0; 0.5 0.5]
    T_init = [0.5 0.0; 0.0 0.5]
    R_mat = Matrix(1.0I, m, r)
    H_init = [1.0, 1.0, 1.0]
    Q_init = [1.0, 1.0]

    # Propagate initial state (MARSS tinitx=0 convention)
    x0 = [0.0, 0.0]
    V0 = 10.0 * Matrix(1.0I, m, m)
    a1 = T_init * x0
    P1 = T_init * V0 * T_init' + R_mat * Diagonal(Q_init) * R_mat'

    # Free parameter masks
    Z_free = [false false; false false; true true]
    T_free = trues(m, m)
    H_free = trues(p)
    Q_free = trues(r)

    result = _em_general_ssm(
        Z_init,
        T_init,
        R_mat,
        H_init,
        Q_init,
        y,
        a1,
        P1;
        Z_free = Z_free,
        T_free = T_free,
        H_free = H_free,
        Q_free = Q_free,
        maxiter = maxiter,
        tol_ll = tol_ll,
        tol_param = tol_param,
        verbose = false,
    )

    return result
end

# ============================================
# Benchmark Functions
# ============================================

function benchmark_single(y_jl, y_r; maxiter = 100, n_runs = 5)
    # Warmup
    run_julia_em(y_jl; maxiter = 10)
    run_marss_em(y_r; maxiter = 10, tol = 0.1)

    # Julia timing - use loose tolerance so it runs exactly maxiter iterations
    julia_times = Float64[]
    julia_result = nothing
    for _ = 1:n_runs
        t = @elapsed begin
            julia_result =
                run_julia_em(y_jl; maxiter = maxiter, tol_ll = 1e-15, tol_param = 1e-15)
        end
        push!(julia_times, t * 1000)  # Convert to ms
    end

    # MARSS timing - use loose tolerance so it runs exactly maxiter iterations
    marss_times = Float64[]
    marss_result = nothing
    for _ = 1:n_runs
        t = @elapsed begin
            marss_result = run_marss_em(y_r; maxiter = maxiter, tol = 1e-15)
        end
        push!(marss_times, t * 1000)
    end

    return (
        julia_times = julia_times,
        julia_result = julia_result,
        marss_times = marss_times,
        marss_result = marss_result,
    )
end

function benchmark_convergence(y_jl, y_r; n_runs = 3)
    # Warmup
    run_julia_em(y_jl; maxiter = 10)
    run_marss_em(y_r; maxiter = 10, tol = 0.1)

    # Julia to convergence
    julia_times = Float64[]
    julia_result = nothing
    for _ = 1:n_runs
        t = @elapsed begin
            julia_result =
                run_julia_em(y_jl; maxiter = 5000, tol_ll = 1e-8, tol_param = 1e-6)
        end
        push!(julia_times, t * 1000)
    end

    # MARSS to convergence
    marss_times = Float64[]
    marss_result = nothing
    for _ = 1:n_runs
        t = @elapsed begin
            marss_result = run_marss_em(y_r; maxiter = 5000, tol = 1e-8)
        end
        push!(marss_times, t * 1000)
    end

    return (
        julia_times = julia_times,
        julia_result = julia_result,
        marss_times = marss_times,
        marss_result = marss_result,
    )
end

# ============================================
# Main
# ============================================

function main()
    println("=" ^ 70)
    println("EM Algorithm Benchmark: Julia (Siphon.jl) vs R (MARSS)")
    println("=" ^ 70)
    println()

    # Generate data in R
    println("Generating data in R...")
    data = generate_data_in_R(n_obs = 200, seed = 54321)

    # Extract data for both Julia and R
    y_r = rcopy(R"$data$y")
    y_jl = convert(Matrix{Float64}, y_r)

    n_obs = size(y_jl, 2)
    p = size(y_jl, 1)

    println("Model: $p observations × 2 states × $n_obs time points")
    println("Free parameters: 11 (T: 4, Z: 2, H: 3, Q: 2)")
    println()

    # Benchmark fixed iterations
    println("-" ^ 70)
    println("Benchmark: 100 EM iterations (fixed)")
    println("-" ^ 70)

    results_100 = benchmark_single(y_jl, y_r; maxiter = 100, n_runs = 5)

    julia_med = median(results_100.julia_times)
    marss_med = median(results_100.marss_times)
    speedup = marss_med / julia_med

    @printf(
        "Julia:  %8.1f ms (median), %8.1f ms (min)\n",
        julia_med,
        minimum(results_100.julia_times)
    )
    @printf(
        "MARSS:  %8.1f ms (median), %8.1f ms (min)\n",
        marss_med,
        minimum(results_100.marss_times)
    )
    @printf("Speedup: %.1fx\n", speedup)
    println()

    # Benchmark to convergence
    println("-" ^ 70)
    println("Benchmark: To convergence (tol=1e-8)")
    println("-" ^ 70)

    results_conv = benchmark_convergence(y_jl, y_r; n_runs = 3)

    julia_med_conv = median(results_conv.julia_times)
    marss_med_conv = median(results_conv.marss_times)
    speedup_conv = marss_med_conv / julia_med_conv

    julia_iters = results_conv.julia_result.iterations
    marss_iters = rcopy(R"$(results_conv.marss_result)$iterations")

    @printf("Julia:  %8.1f ms, %d iterations\n", julia_med_conv, julia_iters)
    @printf("MARSS:  %8.1f ms, %d iterations\n", marss_med_conv, marss_iters)
    @printf(
        "Speedup: %.1fx (%.1fx fewer iterations)\n",
        speedup_conv,
        marss_iters/julia_iters
    )
    println()

    # Compare final estimates
    println("-" ^ 70)
    println("Final Estimates Comparison")
    println("-" ^ 70)

    julia_ll = results_conv.julia_result.loglik
    marss_ll = rcopy(R"$(results_conv.marss_result)$loglik")

    @printf("Log-likelihood:  Julia = %.4f, MARSS = %.4f\n", julia_ll, marss_ll)

    # True values from R
    B_true = rcopy(R"$data$B_true")
    Z_true = rcopy(R"$data$Z_true")
    Q_true = rcopy(R"$data$Q_true")
    R_true = rcopy(R"$data$R_true")

    println("\nTransition matrix T (true B[1,1]=0.8, B[2,2]=0.7):")
    T_julia = results_conv.julia_result.T
    T_marss = rcopy(R"$(results_conv.marss_result)$B")
    @printf(
        "  Julia: [%.4f, %.4f; %.4f, %.4f]\n",
        T_julia[1, 1],
        T_julia[1, 2],
        T_julia[2, 1],
        T_julia[2, 2]
    )
    @printf(
        "  MARSS: [%.4f, %.4f; %.4f, %.4f]\n",
        T_marss[1, 1],
        T_marss[1, 2],
        T_marss[2, 1],
        T_marss[2, 2]
    )

    println("\nObservation matrix Z[3,:] (true: [0.6, 0.4]):")
    Z_julia = results_conv.julia_result.Z
    Z_marss = rcopy(R"$(results_conv.marss_result)$Z")
    @printf("  Julia: [%.4f, %.4f]\n", Z_julia[3, 1], Z_julia[3, 2])
    @printf("  MARSS: [%.4f, %.4f]\n", Z_marss[3, 1], Z_marss[3, 2])

    println("\nState variances Q (true: [0.5, 0.8]):")
    Q_julia = results_conv.julia_result.Q_diag
    Q_marss = rcopy(R"$(results_conv.marss_result)$Q")
    @printf("  Julia: [%.4f, %.4f]\n", Q_julia[1], Q_julia[2])
    @printf("  MARSS: [%.4f, %.4f]\n", Q_marss[1, 1], Q_marss[2, 2])

    println("\nObservation variances H (true: [0.3, 0.4, 0.5]):")
    H_julia = results_conv.julia_result.H_diag
    H_marss = rcopy(R"$(results_conv.marss_result)$R")
    @printf("  Julia: [%.4f, %.4f, %.4f]\n", H_julia[1], H_julia[2], H_julia[3])
    @printf("  MARSS: [%.4f, %.4f, %.4f]\n", H_marss[1, 1], H_marss[2, 2], H_marss[3, 3])

    println()
    println("=" ^ 70)
    println("Summary")
    println("=" ^ 70)
    @printf("Per-iteration speedup:     %.1fx\n", speedup)
    @printf("Total speedup (converge):  %.1fx\n", speedup_conv)
    @printf(
        "Julia iterations:          %d (%.1fx fewer than MARSS)\n",
        julia_iters,
        marss_iters/julia_iters
    )
    println()
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
