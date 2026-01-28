"""
Dynamic Nelson-Siegel Model Estimation Example

This example demonstrates:
1. Setting up a Dynamic Nelson-Siegel (DNS) model
2. Simulating yield curve data from the model
3. Estimating parameters via MLE (since λ enters non-linearly)
4. Comparing with EM estimation (for linear parameters given λ)

The DNS model:
    y_t = Z(λ) * f_t + ε_t,    ε_t ~ N(0, H)
    f_{t+1} = μ + Φ * (f_t - μ) + η_t,    η_t ~ N(0, Q)

where:
- y_t: yields at different maturities (p × 1)
- f_t = [L_t, S_t, C_t]': Level, Slope, Curvature factors (3 × 1)
- Z(λ): Nelson-Siegel factor loadings (p × 3), depends non-linearly on λ
- H: Observation noise covariance (p × p)
- Φ: Factor persistence matrix (3 × 3, typically diagonal)
- Q: Factor innovation covariance (3 × 3)
"""

using Siphon
using LinearAlgebra
using Random
using Optimization
using OptimizationOptimJL
using Printf
using Statistics

# ============================================
# 1. Model Setup
# ============================================

# Maturities in months
maturities = [3, 6, 12, 24, 36, 60, 84, 120]
n_maturities = length(maturities)

# True parameters for simulation
λ_true = 0.0609  # Decay parameter (typical value ~0.06 for monthly data)

# Factor dynamics: f_{t+1} = μ + Φ(f_t - μ) + η_t
# We use mean-adjusted form, so state is (f_t - μ)
Φ_true = Diagonal([0.99, 0.95, 0.90])  # Persistence

# Factor covariance (allow some correlation)
Q_true = [0.01 0.002 0.0;
          0.002 0.02 0.005;
          0.0 0.005 0.03]

# Observation noise (diagonal for simplicity)
H_true = 0.0001 * I(n_maturities)  # Small measurement error

# Factor means (unconditional)
μ_true = [5.0, -1.0, 0.5]  # Level ~5%, negative slope, small curvature

# ============================================
# 2. Build DNS Loading Matrix
# ============================================

function dns_loadings(λ::T, maturities) where {T}
    p = length(maturities)
    Z = ones(T, p, 3)
    for (i, τ) in enumerate(maturities)
        x = λ * τ
        if x < 1e-10
            Z[i, 2] = one(T) - x/2
        else
            Z[i, 2] = (one(T) - exp(-x)) / x
        end
        Z[i, 3] = Z[i, 2] - exp(-x)
    end
    return Z
end

# Check loadings at true λ
Z_true = dns_loadings(λ_true, maturities)
println("DNS Loadings at λ = $λ_true:")
println("Maturity | Level | Slope  | Curvature")
for (i, τ) in enumerate(maturities)
    @printf("%4d mo  |  %.3f | %.3f  | %.3f\n", τ, Z_true[i, 1], Z_true[i, 2], Z_true[i, 3])
end

# ============================================
# 3. Simulate Data
# ============================================

Random.seed!(42)
n_obs = 200  # Monthly observations

# Simulate factors (mean-adjusted: state = f - μ)
L_Q = cholesky(Symmetric(Q_true)).L
L_H = cholesky(Symmetric(Matrix(H_true))).L

factors = zeros(3, n_obs)
yields = zeros(n_maturities, n_obs)

# Initial state (start at mean)
factors[:, 1] = zeros(3)

for t in 1:n_obs
    if t > 1
        factors[:, t] = Φ_true * factors[:, t - 1] + L_Q * randn(3)
    end
    # Yields = Z * (factors + μ) + noise
    yields[:, t] = Z_true * (factors[:, t] + μ_true) + L_H * randn(n_maturities)
end

println("\nSimulated yield statistics:")
println("Mean yields: ", round.(mean(yields, dims = 2)[:], digits = 3))
println("Std yields:  ", round.(std(yields, dims = 2)[:], digits = 3))

# ============================================
# 4. MLE Estimation (Full Model)
# ============================================

"""
Negative log-likelihood for DNS model.
Parameters: [λ, φ_L, φ_S, φ_C, log(q_L), log(q_S), log(q_C), log(h)]
where q_* are diagonal Q elements and h is common H diagonal element.
"""
function dns_negloglik(θ, maturities, y)
    # Unpack parameters
    λ = θ[1]
    φ = θ[2:4]
    q_diag = exp.(θ[5:7])  # Ensure positivity
    h = exp(θ[8])

    # Build model matrices
    Z = dns_loadings(λ, maturities)
    T = Diagonal(φ)
    R = Matrix{Float64}(I, 3, 3)
    Q = Diagonal(q_diag)
    H = h * Matrix{Float64}(I, size(y, 1), size(y, 1))

    # Initial state
    a1 = zeros(3)
    P1 = 1e4 * Matrix{Float64}(I, 3, 3)  # Diffuse prior

    # Build KFParms and compute log-likelihood
    p = KFParms(Z, H, Matrix(T), R, Matrix(Q))
    ll = kalman_loglik(p, y, a1, P1)

    return -ll
end

# Initial parameter guess
θ_init = [
    0.05,   # λ
    0.95,
    0.90,
    0.85,  # φ_L, φ_S, φ_C
    log(0.01),
    log(0.02),
    log(0.03),  # log(q_L), log(q_S), log(q_C)
    log(0.0001)  # log(h)
]

println("\n=== MLE Estimation ===")
println("Initial parameters:")
println("  λ = ", θ_init[1])
println("  Φ = ", θ_init[2:4])
println("  Q_diag = ", exp.(θ_init[5:7]))
println("  H_diag = ", exp(θ_init[8]))

# Define objective
obj_fn = (θ, p) -> dns_negloglik(θ, maturities, yields)

# Set up optimization
opt_prob = OptimizationProblem(
    OptimizationFunction(obj_fn, Optimization.AutoForwardDiff()),
    θ_init
)

# Run optimization
sol = solve(opt_prob, LBFGS(), maxiters = 1000)

θ_mle = sol.u
println("\nMLE estimates:")
println("  λ = ", round(θ_mle[1], digits = 4), " (true: $λ_true)")
println("  Φ = ", round.(θ_mle[2:4], digits = 4), " (true: ", diag(Φ_true), ")")
println("  Q_diag = ", round.(exp.(θ_mle[5:7]), digits = 5), " (true: ", diag(Q_true), ")")
println("  H_diag = ", round(exp(θ_mle[8]), digits = 6), " (true: ", H_true[1, 1], ")")
println("  Final -loglik = ", round(sol.objective, digits = 2))

# ============================================
# 5. Two-Step Estimation: MLE for λ, EM for rest
# ============================================

"""
Profile likelihood: optimize λ while using EM for other parameters.
This is useful when λ enters non-linearly but other parameters have
closed-form EM updates.
"""
function profile_negloglik_with_em(λ_val, maturities, y; em_iters = 50)
    Z = dns_loadings(λ_val, maturities)
    p_obs, n = size(y)
    m = 3  # Number of factors

    # Initial values for EM
    T_init = 0.9 * Matrix{Float64}(I, m, m)
    R = Matrix{Float64}(I, m, m)
    H_init = 0.001 * Matrix{Float64}(I, p_obs, p_obs)
    Q_init = 0.01 * Matrix{Float64}(I, m, m)
    a1 = zeros(m)
    P1 = 1e4 * Matrix{Float64}(I, m, m)

    # Fix Z (it's determined by λ), estimate T, H, Q
    Z_free = falses(p_obs, m)  # Z is fixed given λ
    T_free = trues(m, m)
    H_free = trues(p_obs, p_obs)
    Q_free = trues(m, m)

    # Run EM with full covariance
    result = Siphon.DSL._em_general_ssm_full_cov(
        Z,
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
        maxiter = em_iters,
        tol_ll = 1e-6,
        verbose = false
    )

    return -result.loglik, result
end

println("\n=== Two-Step Estimation (Profile Likelihood + EM) ===")

# Grid search over λ to find profile MLE
λ_grid = range(0.02, 0.15, length = 20)
profile_ll = Float64[]

println("Profiling over λ...")
for λ_val in λ_grid
    neg_ll, _ = profile_negloglik_with_em(λ_val, maturities, yields; em_iters = 100)
    push!(profile_ll, -neg_ll)
end

# Find best λ
λ_profile_mle = λ_grid[argmax(profile_ll)]
println("Profile MLE for λ: ", round(λ_profile_mle, digits = 4), " (true: $λ_true)")

# Get full results at profile MLE
_, em_result = profile_negloglik_with_em(λ_profile_mle, maturities, yields; em_iters = 200)

println("\nEM estimates at λ = $λ_profile_mle:")
println("  T (persistence):")
display(round.(em_result.T, digits = 3))
println("\n  H (observation cov) diagonal: ", round.(diag(em_result.H), digits = 6))
println("  Q (factor cov):")
display(round.(em_result.Q, digits = 5))
println("\n  Loglik = ", round(em_result.loglik, digits = 2))

# ============================================
# 6. Extract Smoothed Factors
# ============================================

# Build final model at MLE estimates
Z_final = dns_loadings(θ_mle[1], maturities)
T_final = Diagonal(θ_mle[2:4])
R_final = Matrix{Float64}(I, 3, 3)
Q_final = Diagonal(exp.(θ_mle[5:7]))
H_final = exp(θ_mle[8]) * Matrix{Float64}(I, n_maturities, n_maturities)
a1 = zeros(3)
P1 = 1e4 * Matrix{Float64}(I, 3, 3)

p_final = KFParms(Z_final, H_final, Matrix(T_final), R_final, Matrix(Q_final))
filt = kalman_filter(p_final, yields, a1, P1)
smooth = kalman_smoother(Z_final, Matrix(T_final), filt.at, filt.Pt, filt.vt, filt.Ft)

println("\n=== Smoothed Factors ===")
println("Correlation between true and smoothed factors:")
for (i, name) in enumerate(["Level", "Slope", "Curvature"])
    corr = cor(factors[i, :], smooth.alpha[i, :])
    println("  $name: ", round(corr, digits = 4))
end

println("\n=== Summary (Simulated Data) ===")
println("The Dynamic Nelson-Siegel model was successfully estimated using:")
println("1. Full MLE with numerical optimization (handles non-linear λ)")
println("2. Profile likelihood with EM for linear parameters")
println("\nBoth approaches recover the true parameters reasonably well.")

# ============================================
# 7. Real Data: US Treasury Yields
# ============================================

println("\n" * "="^60)
println("PART 2: REAL DATA ESTIMATION")
println("="^60)

using DelimitedFiles

# Load real yield data
data_path = joinpath(@__DIR__, "monthlyyields.csv")
if isfile(data_path)
    # Read CSV manually (skip header)
    lines = readlines(data_path)
    header = split(lines[1], ',')

    # Extract maturities from column names (Y3 -> 3, Y12 -> 12, etc.)
    real_maturities = [parse(Int, replace(h, "Y" => "")) for h in header[2:end]]

    # Parse data
    n_real = length(lines) - 1
    n_mat = length(real_maturities)
    real_yields = zeros(n_mat, n_real)

    for (t, line) in enumerate(lines[2:end])
        vals = split(line, ',')
        for (i, v) in enumerate(vals[2:end])
            real_yields[i, t] = parse(Float64, v)
        end
    end

    println("\nLoaded real yield data:")
    println("  Observations: ", n_real, " (monthly)")
    println("  Maturities: ", real_maturities[1], " to ", real_maturities[end], " months")
    println("  Date range: ", split(lines[2], ',')[1], " to ", split(lines[end], ',')[1])
    println("\nYield statistics:")
    println(
        "  Mean yields: ",
        round.(mean(real_yields, dims = 2)[1:5], digits = 2),
        " ... ",
        round.(mean(real_yields, dims = 2)[(end - 2):end], digits = 2)
    )
    println(
        "  Std yields:  ",
        round.(std(real_yields, dims = 2)[1:5], digits = 2),
        " ... ",
        round.(std(real_yields, dims = 2)[(end - 2):end], digits = 2)
    )

    # ============================================
    # 7.1 MLE Estimation on Real Data
    # ============================================

    println("\n=== MLE Estimation on Real Data ===")

    # Use a subset of maturities for faster estimation
    # Select: 3, 12, 24, 60, 120, 240, 360 months
    mat_idx = findall(m -> m in [3, 12, 24, 60, 120, 240, 360], real_maturities)
    selected_maturities = real_maturities[mat_idx]
    selected_yields = real_yields[mat_idx, :]

    println("Using maturities: ", selected_maturities, " months")

    # Define negative log-likelihood for real data
    function dns_negloglik_real(θ, maturities, y)
        λ = θ[1]
        φ = θ[2:4]
        q_diag = exp.(θ[5:7])
        h = exp(θ[8])

        # Bounds check on λ
        if λ <= 0.001 || λ > 0.5
            return Inf
        end

        Z = dns_loadings(λ, maturities)
        T_mat = Diagonal(φ)
        R_mat = Matrix{Float64}(I, 3, 3)
        Q_mat = Diagonal(q_diag)
        H_mat = h * Matrix{Float64}(I, size(y, 1), size(y, 1))

        a1 = zeros(3)
        P1 = 1e4 * Matrix{Float64}(I, 3, 3)

        p = KFParms(Z, H_mat, Matrix(T_mat), R_mat, Matrix(Q_mat))
        ll = kalman_loglik(p, y, a1, P1)

        return -ll
    end

    # Initial parameters for real data
    θ_init_real = [
        0.06,   # λ (typical value for monthly data)
        0.99,
        0.95,
        0.90,  # φ_L, φ_S, φ_C (high persistence expected)
        log(0.1),
        log(0.5),
        log(0.5),  # log(q_L), log(q_S), log(q_C)
        log(0.01)  # log(h) - measurement error
    ]

    println("Initial λ = ", θ_init_real[1])

    # Optimize
    obj_fn_real = (θ, p) -> dns_negloglik_real(θ, selected_maturities, selected_yields)

    opt_prob_real = OptimizationProblem(
        OptimizationFunction(obj_fn_real, Optimization.AutoForwardDiff()),
        θ_init_real
    )

    sol_real = solve(opt_prob_real, LBFGS(), maxiters = 2000)

    θ_mle_real = sol_real.u
    println("\nMLE estimates (real data):")
    println("  λ = ", round(θ_mle_real[1], digits = 4))
    println("  Φ_diag = ", round.(θ_mle_real[2:4], digits = 4))
    println("  Q_diag = ", round.(exp.(θ_mle_real[5:7]), digits = 4))
    println("  H_diag = ", round(exp(θ_mle_real[8]), digits = 6))
    println("  Final -loglik = ", round(sol_real.objective, digits = 2))

    # Implied decay half-life (in months)
    τ_star = 1 / θ_mle_real[1]  # Maturity where slope/curvature loading maximized
    println(
        "\n  Implied τ* (max curvature loading): ",
        round(τ_star, digits = 1),
        " months"
    )

    # ============================================
    # 7.2 Extract and Display Factors
    # ============================================

    println("\n=== Extracted Factors (Real Data) ===")

    Z_real = dns_loadings(θ_mle_real[1], selected_maturities)
    T_real = Diagonal(θ_mle_real[2:4])
    R_real = Matrix{Float64}(I, 3, 3)
    Q_real = Diagonal(exp.(θ_mle_real[5:7]))
    H_real = exp(θ_mle_real[8]) *
             Matrix{Float64}(I, length(selected_maturities), length(selected_maturities))
    a1_real = zeros(3)
    P1_real = 1e4 * Matrix{Float64}(I, 3, 3)

    p_real = KFParms(Z_real, H_real, Matrix(T_real), R_real, Matrix(Q_real))
    filt_real = kalman_filter(p_real, selected_yields, a1_real, P1_real)
    smooth_real = kalman_smoother(
        Z_real,
        Matrix(T_real),
        filt_real.at,
        filt_real.Pt,
        filt_real.vt,
        filt_real.Ft
    )

    # Factor statistics
    println("Smoothed factor statistics:")
    for (i, name) in enumerate(["Level", "Slope", "Curvature"])
        factor = smooth_real.alpha[i, :]
        println(
            "  $name: mean = ",
            round(mean(factor), digits = 2),
            ", std = ",
            round(std(factor), digits = 2),
            ", range = [",
            round(minimum(factor), digits = 2),
            ", ",
            round(maximum(factor), digits = 2),
            "]"
        )
    end

    # ============================================
    # 7.3 Profile Likelihood + EM (Full Covariance)
    # ============================================

    println("\n=== Profile Likelihood + EM (Full Covariance, Real Data) ===")

    function profile_negloglik_real(λ_val, maturities, y; em_iters = 100)
        Z = dns_loadings(λ_val, maturities)
        p_obs, n = size(y)
        m = 3

        T_init = 0.95 * Matrix{Float64}(I, m, m)
        R = Matrix{Float64}(I, m, m)
        H_init = 0.01 * Matrix{Float64}(I, p_obs, p_obs)
        Q_init = 0.1 * Matrix{Float64}(I, m, m)
        a1 = zeros(m)
        P1 = 1e4 * Matrix{Float64}(I, m, m)

        Z_free = falses(p_obs, m)
        T_free = trues(m, m)
        H_free = trues(p_obs, p_obs)
        Q_free = trues(m, m)

        result = Siphon.DSL._em_general_ssm_full_cov(
            Z,
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
            maxiter = em_iters,
            tol_ll = 1e-6,
            verbose = false
        )

        return -result.loglik, result
    end

    # Grid search over λ
    λ_grid_real = range(0.02, 0.15, length = 15)
    profile_ll_real = Float64[]

    println("Profiling over λ...")
    for λ_val in λ_grid_real
        neg_ll,
        _ = profile_negloglik_real(
            λ_val,
            selected_maturities,
            selected_yields;
            em_iters = 150
        )
        push!(profile_ll_real, -neg_ll)
    end

    λ_profile_real = λ_grid_real[argmax(profile_ll_real)]
    println("Profile MLE for λ: ", round(λ_profile_real, digits = 4))

    # Final EM at profile MLE
    _,
    em_result_real = profile_negloglik_real(
        λ_profile_real,
        selected_maturities,
        selected_yields;
        em_iters = 300
    )

    println("\nEM estimates (full covariance):")
    println("  T (persistence matrix):")
    display(round.(em_result_real.T, digits = 3))
    println("\n  Q (factor covariance):")
    display(round.(em_result_real.Q, digits = 4))
    println("\n  H diagonal: ", round.(diag(em_result_real.H), digits = 5))
    println("  Loglik = ", round(em_result_real.loglik, digits = 2))

    # ============================================
    # 7.4 Model Comparison
    # ============================================

    println("\n=== Model Comparison ===")

    # Compute AIC/BIC for both models
    n_params_mle = 8  # λ, 3 φ's, 3 q's, 1 h
    n_params_em = 1 + 9 + 6 + length(selected_maturities)  # λ + T(3x3) + Q(6 unique) + H(diagonal)

    aic_mle = 2 * sol_real.objective + 2 * n_params_mle
    aic_em = 2 * (-em_result_real.loglik) + 2 * n_params_em

    bic_mle = 2 * sol_real.objective + log(n_real) * n_params_mle
    bic_em = 2 * (-em_result_real.loglik) + log(n_real) * n_params_em

    println("Diagonal Model (MLE):")
    println(
        "  -LogLik = ",
        round(sol_real.objective, digits = 2),
        ", AIC = ",
        round(aic_mle, digits = 2),
        ", BIC = ",
        round(bic_mle, digits = 2)
    )
    println("Full Covariance Model (EM):")
    println(
        "  -LogLik = ",
        round(-em_result_real.loglik, digits = 2),
        ", AIC = ",
        round(aic_em, digits = 2),
        ", BIC = ",
        round(bic_em, digits = 2)
    )

    # ============================================
    # 8. Forecast Evaluation (Parallel with Distributed)
    # ============================================

    println("\n" * "="^60)
    println("PART 3: FORECAST EVALUATION (PARALLEL)")
    println("="^60)

    # ============================================
    # 8.0 Setup Distributed Computing
    # ============================================

    using Distributed

    # Configuration: number of workers
    const N_WORKERS = min(28, Sys.CPU_THREADS)  # Use up to 28 cores or available threads

    # Add workers if not already present
    if nworkers() < N_WORKERS
        println("Adding $(N_WORKERS - nworkers()) workers...")
        addprocs(N_WORKERS - nworkers())
    end
    println("Running with $(nworkers()) workers")

    # Load packages on all workers
    @everywhere begin
        using Siphon
        using LinearAlgebra
        using Optimization
        using OptimizationOptimJL
        using Statistics
    end

    # Define functions on all workers
    @everywhere function dns_loadings_worker(λ::T, maturities) where {T}
        p = length(maturities)
        Z = ones(T, p, 3)
        for (i, τ) in enumerate(maturities)
            x = λ * τ
            if x < 1e-10
                Z[i, 2] = one(T) - x/2
            else
                Z[i, 2] = (one(T) - exp(-x)) / x
            end
            Z[i, 3] = Z[i, 2] - exp(-x)
        end
        return Z
    end

    @everywhere function estimate_dns_diagonal_worker(y, maturities; maxiters = 300)
        θ_init = [0.06, 0.99, 0.95, 0.90, log(0.1), log(0.5), log(0.5), log(0.01)]

        function negloglik(θ, mats, data)
            λ = θ[1]
            (λ <= 0.001 || λ > 0.5) && return Inf
            φ = θ[2:4]
            q_diag = exp.(θ[5:7])
            h = exp(θ[8])

            Z = dns_loadings_worker(λ, mats)
            T_mat = Diagonal(φ)
            R_mat = Matrix{Float64}(I, 3, 3)
            Q_mat = Diagonal(q_diag)
            H_mat = h * Matrix{Float64}(I, size(data, 1), size(data, 1))

            a1 = zeros(3)
            P1 = 1e4 * Matrix{Float64}(I, 3, 3)

            p = KFParms(Z, H_mat, Matrix(T_mat), R_mat, Matrix(Q_mat))
            return -kalman_loglik(p, data, a1, P1)
        end

        obj_fn = (θ, p) -> negloglik(θ, maturities, y)
        opt_prob = OptimizationProblem(
            OptimizationFunction(obj_fn, Optimization.AutoForwardDiff()),
            θ_init
        )
        sol = solve(opt_prob, LBFGS(), maxiters = maxiters)
        return sol.u
    end

    @everywhere function estimate_dns_full_cov_worker(
            y,
            maturities;
            λ_grid = range(0.02, 0.12, length = 8),
            em_iters = 80
    )
        p_obs, n = size(y)
        m = 3

        best_ll = -Inf
        best_λ = 0.06
        best_T = zeros(m, m)
        best_Q = zeros(m, m)
        best_H = zeros(p_obs, p_obs)

        for λ_val in λ_grid
            Z = dns_loadings_worker(λ_val, maturities)
            T_init = 0.95 * Matrix{Float64}(I, m, m)
            R = Matrix{Float64}(I, m, m)
            H_init = 0.01 * Matrix{Float64}(I, p_obs, p_obs)
            Q_init = 0.1 * Matrix{Float64}(I, m, m)
            a1 = zeros(m)
            P1 = 1e4 * Matrix{Float64}(I, m, m)

            Z_free = falses(p_obs, m)
            T_free = trues(m, m)
            H_free = Diagonal(trues(p_obs))
            Q_free = trues(m, m)

            result = Siphon.DSL._em_general_ssm_full_cov(
                Z,
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
                maxiter = em_iters,
                tol_ll = 1e-5,
                verbose = false
            )

            if result.loglik > best_ll
                best_ll = result.loglik
                best_λ = λ_val
                best_T = result.T
                best_Q = result.Q
                best_H = result.H
            end
        end

        return (λ = best_λ, T = best_T, Q = best_Q, H = best_H)
    end

    @everywhere function forecast_single_origin(
            y_full,
            maturities,
            t::Int,
            max_horizon::Int,
            model_type::Symbol,
            window_type::Symbol,
            window_size::Int
    )
        n_maturities = length(maturities)

        # Determine training window
        if window_type == :expanding
            train_start = 1
            train_end = t
        else  # rolling
            train_start = max(1, t - window_size + 1)
            train_end = t
        end

        y_train = y_full[:, train_start:train_end]

        # Estimate model
        if model_type == :diagonal
            θ = estimate_dns_diagonal_worker(y_train, maturities; maxiters = 300)
            λ = θ[1]
            Z = dns_loadings_worker(λ, maturities)
            T_mat = Matrix(Diagonal(θ[2:4]))
            R_mat = Matrix{Float64}(I, 3, 3)
            Q_mat = Matrix(Diagonal(exp.(θ[5:7])))
            H_mat = exp(θ[8]) * Matrix{Float64}(I, n_maturities, n_maturities)
        else  # full_cov
            result = estimate_dns_full_cov_worker(y_train, maturities; em_iters = 80)
            λ = result.λ
            Z = dns_loadings_worker(λ, maturities)
            T_mat = result.T
            R_mat = Matrix{Float64}(I, 3, 3)
            Q_mat = result.Q
            H_mat = result.H
        end

        # Run filter on training data to get final state
        a1 = zeros(3)
        P1 = 1e4 * Matrix{Float64}(I, 3, 3)
        p_kf = KFParms(Z, H_mat, T_mat, R_mat, Q_mat)
        filt = kalman_filter(p_kf, y_train, a1, P1)

        # Get filtered state at end of training
        filtered_state = filt.att[:, end]

        # Generate forecasts for each horizon
        errors = zeros(n_maturities, max_horizon)
        a = copy(filtered_state)
        for h in 1:max_horizon
            a = T_mat * a  # Propagate state
            y_fc = Z * a
            y_actual = y_full[:, t + h]
            errors[:, h] = y_fc - y_actual
        end

        return errors
    end

    # ============================================
    # 8.1 Parallel Forecast Evaluation Function
    # ============================================

    function run_forecast_eval_parallel(
            y,
            maturities,
            model_type::Symbol,
            start_idx::Int,
            end_idx::Int;
            window_type::Symbol = :expanding,
            window_size::Int = 84,
            max_horizon::Int = 12
    )
        n_maturities = length(maturities)
        forecast_origins = collect(start_idx:(end_idx - max_horizon))
        n_forecasts = length(forecast_origins)

        if n_forecasts <= 0
            return zeros(0, n_maturities, max_horizon)
        end

        println("  Running $(n_forecasts) forecast origins on $(nworkers()) workers...")

        # Parallel map over forecast origins
        results = pmap(
            t -> forecast_single_origin(
                y,
                maturities,
                t,
                max_horizon,
                model_type,
                window_type,
                window_size
            ),
            forecast_origins
        )

        # Collect results into 3D array
        errors = zeros(n_forecasts, n_maturities, max_horizon)
        for (i, err) in enumerate(results)
            errors[i, :, :] = err
        end

        return errors
    end

    # ============================================
    # 8.2 Compute Metrics Function
    # ============================================

    function compute_metrics(errors)
        n_fc, n_mat, n_h = size(errors)
        mse = zeros(n_h)
        mae = zeros(n_h)
        for h in 1:n_h
            e = errors[:, :, h]
            mse[h] = mean(e .^ 2)
            mae[h] = mean(abs.(e))
        end
        return mse, mae
    end

    # ============================================
    # 8.3 Run Evaluations
    # ============================================

    max_horizon = 12  # Forecast horizons 1, 2, ..., 12 steps ahead
    initial_window = 84  # 7 years of monthly data
    rolling_window = 120  # 10 years for rolling window

    start_eval = initial_window
    end_eval = size(selected_yields, 2)
    start_eval_roll = rolling_window

    # Expanding Window Evaluation
    println("\n=== Expanding Window Forecast Evaluation ===")
    println("Initial window: $initial_window months (7 years)")
    println("Forecast horizons: 1 to $max_horizon months")
    println("Forecast origins: $(end_eval - start_eval - max_horizon + 1)")

    println("\nDiagonal model:")
    errors_diag_expanding = run_forecast_eval_parallel(
        selected_yields,
        selected_maturities,
        :diagonal,
        start_eval,
        end_eval;
        window_type = :expanding,
        max_horizon = max_horizon
    )

    println("Full covariance model:")
    errors_full_expanding = run_forecast_eval_parallel(
        selected_yields,
        selected_maturities,
        :full_cov,
        start_eval,
        end_eval;
        window_type = :expanding,
        max_horizon = max_horizon
    )

    mse_diag_exp, mae_diag_exp = compute_metrics(errors_diag_expanding)
    mse_full_exp, mae_full_exp = compute_metrics(errors_full_expanding)

    println("\nExpanding Window Results:")
    println("─"^70)
    println(
        "Horizon │     MSE (Diagonal)    MSE (Full)  │     MAE (Diagonal)    MAE (Full)",
    )
    println("─"^70)
    for h in 1:max_horizon
        @printf("  %2d    │        %8.4f      %8.4f     │        %8.4f      %8.4f\n",
            h,
            mse_diag_exp[h],
            mse_full_exp[h],
            mae_diag_exp[h],
            mae_full_exp[h])
    end
    println("─"^70)
    @printf(" Avg    │        %8.4f      %8.4f     │        %8.4f      %8.4f\n",
        mean(mse_diag_exp),
        mean(mse_full_exp),
        mean(mae_diag_exp),
        mean(mae_full_exp))
    println("─"^70)

    # Rolling Window Evaluation
    println("\n=== Rolling Window Forecast Evaluation ===")
    println("Window size: $rolling_window months (10 years)")
    println("Forecast horizons: 1 to $max_horizon months")
    println("Forecast origins: $(end_eval - start_eval_roll - max_horizon + 1)")

    println("\nDiagonal model:")
    errors_diag_rolling = run_forecast_eval_parallel(
        selected_yields,
        selected_maturities,
        :diagonal,
        start_eval_roll,
        end_eval;
        window_type = :rolling,
        window_size = rolling_window,
        max_horizon = max_horizon
    )

    println("Full covariance model:")
    errors_full_rolling = run_forecast_eval_parallel(
        selected_yields,
        selected_maturities,
        :full_cov,
        start_eval_roll,
        end_eval;
        window_type = :rolling,
        window_size = rolling_window,
        max_horizon = max_horizon
    )

    mse_diag_roll, mae_diag_roll = compute_metrics(errors_diag_rolling)
    mse_full_roll, mae_full_roll = compute_metrics(errors_full_rolling)

    println("\nRolling Window Results:")
    println("─"^70)
    println(
        "Horizon │     MSE (Diagonal)    MSE (Full)  │     MAE (Diagonal)    MAE (Full)",
    )
    println("─"^70)
    for h in 1:max_horizon
        @printf("  %2d    │        %8.4f      %8.4f     │        %8.4f      %8.4f\n",
            h,
            mse_diag_roll[h],
            mse_full_roll[h],
            mae_diag_roll[h],
            mae_full_roll[h])
    end
    println("─"^70)
    @printf(" Avg    │        %8.4f      %8.4f     │        %8.4f      %8.4f\n",
        mean(mse_diag_roll),
        mean(mse_full_roll),
        mean(mae_diag_roll),
        mean(mae_full_roll))
    println("─"^70)

    # ============================================
    # 8.4 Summary Comparison
    # ============================================

    println("\n=== Forecast Performance Summary ===")
    println("\nExpanding Window:")
    avg_mse_imp_exp = (mean(mse_diag_exp) - mean(mse_full_exp)) / mean(mse_diag_exp) * 100
    avg_mae_imp_exp = (mean(mae_diag_exp) - mean(mae_full_exp)) / mean(mae_diag_exp) * 100
    println(
        "  Full Cov vs Diagonal - MSE improvement: ",
        round(avg_mse_imp_exp, digits = 2),
        "%"
    )
    println(
        "  Full Cov vs Diagonal - MAE improvement: ",
        round(avg_mae_imp_exp, digits = 2),
        "%"
    )

    println("\nRolling Window:")
    avg_mse_imp_roll = (mean(mse_diag_roll) - mean(mse_full_roll)) / mean(mse_diag_roll) *
                       100
    avg_mae_imp_roll = (mean(mae_diag_roll) - mean(mae_full_roll)) / mean(mae_diag_roll) *
                       100
    println(
        "  Full Cov vs Diagonal - MSE improvement: ",
        round(avg_mse_imp_roll, digits = 2),
        "%"
    )
    println(
        "  Full Cov vs Diagonal - MAE improvement: ",
        round(avg_mae_imp_roll, digits = 2),
        "%"
    )

    # By horizon comparison
    println("\nMSE Ratio (Full/Diagonal) by Horizon:")
    print("  Expanding: ")
    for h in 1:max_horizon
        @printf("%.2f ", mse_full_exp[h] / mse_diag_exp[h])
    end
    println()
    print("  Rolling:   ")
    for h in 1:max_horizon
        @printf("%.2f ", mse_full_roll[h] / mse_diag_roll[h])
    end
    println()

    # Clean up workers (optional - comment out to keep for interactive use)
    # rmprocs(workers())

    println("\n=== Forecast Evaluation Complete ===")
else
    println("\nNote: Real yield data file not found at $data_path")
    println("Skipping real data estimation.")
end
