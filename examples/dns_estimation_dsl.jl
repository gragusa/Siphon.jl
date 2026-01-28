"""
Dynamic Nelson-Siegel Model Estimation via DSL

This example demonstrates using the Siphon.jl DSL to estimate DNS models:
1. `dns_model()` template for model specification
2. `optimize_ssm()` for full MLE (when λ can be estimated via AD)
3. `profile_em_ssm()` for profile likelihood + EM (more robust)

The DSL approach provides:
- Automatic parameter handling (bounds, transformations)
- Clean integration with optimization
- Support for full covariance estimation

Compare with `dns_estimation.jl` which manually constructs matrices.
"""

using Siphon
using LinearAlgebra
using Random
using Statistics
using Printf

# ============================================
# 1. Model Setup using DSL
# ============================================

# Maturities in months
maturities = [3, 6, 12, 24, 36, 60, 84, 120]
n_maturities = length(maturities)

println("="^60)
println("DNS Estimation via Siphon.jl DSL")
println("="^60)
println("\nMaturities: ", maturities, " months")

# ============================================
# 2. Create DNS Model Specification
# ============================================

println("\n=== Model Specifications ===")

# Option 1: Diagonal T and Q (simpler, fewer parameters)
spec_diag = dns_model(
    maturities;
    T_structure = :diagonal,  # 3 params: T_L, T_S, T_C
    H_structure = :diagonal,  # p params: H_1, ..., H_p
    Q_structure = :diagonal,  # 3 params: Q_L, Q_S, Q_C
    λ_init = 0.06,
    T_init = 0.95,
    Q_init = 0.01,
    H_init = 0.001
)

println("\nDiagonal model parameters: ", param_names(spec_diag))
println("Total parameters: ", n_params(spec_diag))

# Option 2: Full T and Q (for profile EM estimation)
spec_full = dns_model(
    maturities;
    T_structure = :full,      # 9 params: T_1_1, T_1_2, ..., T_3_3
    H_structure = :diagonal,  # p params
    Q_structure = :full,      # 6 params (via Cholesky): Q_σ_L, Q_σ_S, Q_σ_C, Q_corr_1, Q_corr_2, Q_corr_3
    λ_init = 0.06,
    T_init = 0.95,
    Q_init = 0.01,
    H_init = 0.001
)

println("\nFull model parameters: ", param_names(spec_full))
println("Total parameters: ", n_params(spec_full))

# ============================================
# 3. Simulate Data
# ============================================

println("\n=== Simulating Data ===")

Random.seed!(42)
n_obs = 200

# True parameters for simulation
λ_true = 0.0609
Φ_true = Diagonal([0.99, 0.95, 0.90])
Q_true = [0.01 0.002 0.0;
          0.002 0.02 0.005;
          0.0 0.005 0.03]
H_true = 0.0001 * I(n_maturities)

# Build true DNS loadings
function dns_loadings(λ, maturities)
    p = length(maturities)
    Z = ones(p, 3)
    for (i, τ) in enumerate(maturities)
        x = λ * τ
        if x < 1e-10
            Z[i, 2] = 1.0 - x/2
        else
            Z[i, 2] = (1 - exp(-x)) / x
        end
        Z[i, 3] = Z[i, 2] - exp(-x)
    end
    return Z
end

Z_true = dns_loadings(λ_true, maturities)

# Simulate
L_Q = cholesky(Symmetric(Q_true)).L
L_H = cholesky(Symmetric(Matrix(H_true))).L

factors = zeros(3, n_obs)
yields = zeros(n_maturities, n_obs)

# Factor means
μ_true = [5.0, -1.0, 0.5]

for t in 1:n_obs
    if t > 1
        factors[:, t] = Φ_true * factors[:, t - 1] + L_Q * randn(3)
    end
    yields[:, t] = Z_true * (factors[:, t] + μ_true) + L_H * randn(n_maturities)
end

println("Simulated $n_obs monthly observations")
println("Mean yields: ", round.(mean(yields, dims = 2)[:], digits = 2))
println("Std yields:  ", round.(std(yields, dims = 2)[:], digits = 2))

# ============================================
# 4. Profile EM Estimation (Recommended)
# ============================================

println("\n=== Profile EM Estimation (Full Covariance) ===")
println("This approach grids over λ and uses EM for T, H, Q")

result_em = profile_em_ssm(
    spec_full,
    yields;
    λ_grid = 0.02:0.01:0.12,
    verbose = true,
    maxiter = 200
)

println("\n--- Results ---")
println("Optimal λ: ", round(result_em.λ_optimal, digits = 4), " (true: $λ_true)")
println("Log-likelihood: ", round(result_em.loglik, digits = 2))

# Access estimated matrices from EM result
T_est = result_em.em_result.T
Q_est = result_em.em_result.Q
H_est = result_em.em_result.H

println("\nEstimated T (transition matrix):")
for i in 1:3
    @printf("  [%.4f  %.4f  %.4f]\n", T_est[i, 1], T_est[i, 2], T_est[i, 3])
end
println("True T diagonal: ", diag(Φ_true))

println("\nEstimated Q (factor covariance):")
for i in 1:3
    @printf("  [%.5f  %.5f  %.5f]\n", Q_est[i, 1], Q_est[i, 2], Q_est[i, 3])
end
println("True Q diagonal: ", diag(Q_true))

println("\nEstimated H diagonal: ", round.(diag(H_est)[1:3], digits = 6), " ...")
println("True H diagonal: ", H_true[1, 1])

# ============================================
# 5. Extract Smoothed Factors
# ============================================

println("\n=== Smoothed Factors ===")

# Build Z matrix at optimal λ
Z_opt = dns_loadings(result_em.λ_optimal, maturities)

# Build KFParms
p_final = KFParms(Z_opt, H_est, T_est, Matrix{Float64}(I, 3, 3), Q_est)
a1 = zeros(3)
P1 = 1e4 * Matrix{Float64}(I, 3, 3)

# Filter and smooth
filt = kalman_filter(p_final, yields, a1, P1)
smooth = kalman_smoother(Z_opt, T_est, filt.at, filt.Pt, filt.vt, filt.Ft)

println("Correlation between true and smoothed factors:")
for (i, name) in enumerate(["Level", "Slope", "Curvature"])
    corr = cor(factors[i, :], smooth.alpha[i, :])
    println("  $name: ", round(corr, digits = 4))
end

# ============================================
# 6. Profile Likelihood Plot Data
# ============================================

println("\n=== Profile Likelihood ===")
println("λ values searched: ", length(result_em.λ_grid))
println("λ range: [", result_em.λ_grid[1], ", ", result_em.λ_grid[end], "]")

# Find peak
peak_idx = argmax(result_em.loglik_profile)
println("Peak at λ = ", result_em.λ_grid[peak_idx])
println(
    "Profile LL range: [",
    round(minimum(result_em.loglik_profile), digits = 1),
    ", ",
    round(maximum(result_em.loglik_profile), digits = 1),
    "]"
)

# ============================================
# 7. Compare with Diagonal Model
# ============================================

println("\n=== Comparison: Full vs Diagonal ===")

# Profile EM with diagonal Q
spec_diag_T_full = dns_model(
    maturities;
    T_structure = :full,
    H_structure = :diagonal,
    Q_structure = :diagonal,  # Diagonal Q
    λ_init = 0.06,
    T_init = 0.95,
    Q_init = 0.01,
    H_init = 0.001
)

result_diag = profile_em_ssm(
    spec_diag_T_full,
    yields;
    λ_grid = 0.02:0.01:0.12,
    verbose = false,
    maxiter = 200
)

println(
    "Full Q model:     λ = ",
    round(result_em.λ_optimal, digits = 4),
    ", loglik = ",
    round(result_em.loglik, digits = 2)
)
println(
    "Diagonal Q model: λ = ",
    round(result_diag.λ_optimal, digits = 4),
    ", loglik = ",
    round(result_diag.loglik, digits = 2)
)
println(
    "Log-likelihood improvement: ",
    round(result_em.loglik - result_diag.loglik, digits = 2)
)

# ============================================
# 8. Real Data Example (if available)
# ============================================

println("\n" * "="^60)
println("REAL DATA ESTIMATION")
println("="^60)

using DelimitedFiles

data_path = joinpath(@__DIR__, "monthlyyields.csv")
if isfile(data_path)
    # Load data
    lines = readlines(data_path)
    header = split(lines[1], ',')
    real_maturities = [parse(Int, replace(h, "Y" => "")) for h in header[2:end]]

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
    println("  Observations: ", n_real)
    println("  Maturities: ", real_maturities[1], " to ", real_maturities[end], " months")

    # Select subset of maturities
    mat_idx = findall(m -> m in [3, 12, 24, 60, 120, 240, 360], real_maturities)
    selected_maturities = real_maturities[mat_idx]
    selected_yields = real_yields[mat_idx, :]

    println("  Selected: ", selected_maturities, " months")

    # Create spec for real data
    spec_real = dns_model(
        selected_maturities;
        T_structure = :full,
        H_structure = :diagonal,
        Q_structure = :full,
        λ_init = 0.06,
        T_init = 0.95,
        Q_init = 0.1,
        H_init = 0.01
    )

    println("\n=== Profile EM on Real Data ===")

    result_real = profile_em_ssm(
        spec_real,
        selected_yields;
        λ_grid = 0.02:0.005:0.12,
        verbose = true,
        maxiter = 300
    )

    println("\n--- Real Data Results ---")
    println("Optimal λ: ", round(result_real.λ_optimal, digits = 4))
    println("Log-likelihood: ", round(result_real.loglik, digits = 2))

    # Implied characteristic maturity
    τ_star = 1 / result_real.λ_optimal
    println("Implied τ* (max curvature loading): ", round(τ_star, digits = 1), " months")

    # Display T matrix
    T_real = result_real.em_result.T
    println("\nEstimated T (factor dynamics):")
    for i in 1:3
        @printf("  [%.4f  %.4f  %.4f]\n", T_real[i, 1], T_real[i, 2], T_real[i, 3])
    end

    # Display Q matrix
    Q_real = result_real.em_result.Q
    println("\nEstimated Q (factor covariance):")
    for i in 1:3
        @printf("  [%.5f  %.5f  %.5f]\n", Q_real[i, 1], Q_real[i, 2], Q_real[i, 3])
    end

    # Factor correlations implied by Q
    σ = sqrt.(diag(Q_real))
    corr_mat = Q_real ./ (σ * σ')
    println("\nFactor correlations:")
    for i in 1:3
        @printf("  [%.3f  %.3f  %.3f]\n", corr_mat[i, 1], corr_mat[i, 2], corr_mat[i, 3])
    end

    # Extract smoothed factors
    Z_real = dns_loadings(result_real.λ_optimal, selected_maturities)
    p_real = KFParms(
        Z_real, result_real.em_result.H, T_real, Matrix{Float64}(I, 3, 3), Q_real)
    a1_real = zeros(3)
    P1_real = 1e4 * Matrix{Float64}(I, 3, 3)

    filt_real = kalman_filter(p_real, selected_yields, a1_real, P1_real)
    smooth_real = kalman_smoother(
        Z_real,
        T_real,
        filt_real.at,
        filt_real.Pt,
        filt_real.vt,
        filt_real.Ft
    )

    println("\nSmoothed factor statistics:")
    for (i, name) in enumerate(["Level", "Slope", "Curvature"])
        factor = smooth_real.alpha[i, :]
        println(
            "  $name: mean = ",
            round(mean(factor), digits = 2),
            ", std = ",
            round(std(factor), digits = 2)
        )
    end
else
    println("\nReal data file not found: $data_path")
    println("Skipping real data example.")
end

# ============================================
# Summary
# ============================================

println("\n" * "="^60)
println("SUMMARY")
println("="^60)
println("""
The dns_model() template provides a clean DSL interface for DNS estimation:

1. SPECIFICATION:
   spec = dns_model(maturities; T_structure=:full, Q_structure=:full)

2. PROFILE EM ESTIMATION (recommended for DNS):
   result = profile_em_ssm(spec, y; λ_grid=0.02:0.01:0.15)

3. ACCESS RESULTS:
   result.λ_optimal      # Optimal λ
   result.θ              # All parameters as NamedTuple
   result.em_result.T    # Estimated T matrix
   result.em_result.Q    # Estimated Q matrix
   result.loglik_profile # For plotting profile likelihood

Key advantages over manual construction:
- Automatic parameter management
- Consistent API with other DSL models
- Easy switching between model structures
- Integration with smoothing/filtering utilities
""")
