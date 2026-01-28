#=
Factor Extraction Comparison: Dynamic (KFS) vs Static (PCA)

Compares factor extraction using:
- Siphon.jl's DynamicFactorModel (Kalman filter/smoother with EM)
- Factotum.jl's FactorModel (Principal Component Analysis)

Based on methodology from Poncela et al. (2021):
"Factor Extraction Using Kalman Filter and Smoothing:
 This Is Not Just Another Survey"
 International Journal of Forecasting 37 (2021) 1399–1425

Key findings from the paper:
- PCA doesn't exploit serial correlation → inefficient when factors are persistent
- KFS provides MSE estimates and handles missing data naturally
- Both methods face rotation/sign indeterminacy
- As N → ∞, PCA becomes efficient (asymptotic theory)
=#

using Siphon
using Factotum
using LinearAlgebra
using Statistics
using Random
using Printf
using DelimitedFiles

# ============================================================================
# Section 1: Data Simulation
# ============================================================================

"""
    simulate_dfm(N, T, r; phi=0.7, sigma_f=1.0, sigma_e=0.5, rng=Random.default_rng())

Simulate data from a Dynamic Factor Model (following Poncela et al. 2021):

    y_{it} = λ_i' f_t + ε_{it}
    f_t = φ f_{t-1} + η_t,  η_t ~ N(0, σ_f² I)
    ε_{it} ~ N(0, σ_e²)  (independent across i and t)

# Arguments
- `N`: Number of variables (cross-sectional dimension)
- `T`: Number of time periods
- `r`: Number of factors
- `phi`: AR(1) coefficient for factor dynamics (default: 0.7)
- `sigma_f`: Factor innovation std dev (default: 1.0)
- `sigma_e`: Idiosyncratic error std dev (default: 0.5)
- `rng`: Random number generator

# Returns
Named tuple with:
- `y`: N × T observation matrix (Siphon format)
- `f_true`: r × T true factor matrix
- `Lambda_true`: N × r true loading matrix
"""
function simulate_dfm(
        N::Int,
        T::Int,
        r::Int;
        phi::Float64 = 0.7,
        sigma_f::Float64 = 1.0,
        sigma_e::Float64 = 0.5,
        rng::AbstractRNG = Random.default_rng()
)
    # Generate loadings: Lambda ~ N(0, 1)
    Lambda = randn(rng, N, r)

    # Generate factor dynamics: f_t = phi * f_{t-1} + eta_t
    f = zeros(r, T)

    # Initialize from stationary distribution (if |phi| < 1)
    if abs(phi) < 1.0
        sigma_stationary = sigma_f / sqrt(1 - phi^2)
        f[:, 1] = sigma_stationary * randn(rng, r)
    else
        f[:, 1] = sigma_f * randn(rng, r)
    end

    # Simulate AR(1) process
    for t in 2:T
        f[:, t] = phi * f[:, t - 1] + sigma_f * randn(rng, r)
    end

    # Generate observations: y = Lambda * f + e
    e = sigma_e * randn(rng, N, T)
    y = Lambda * f + e

    return (y = y, f_true = f, Lambda_true = Lambda)
end

# ============================================================================
# Section 2: Factor Extraction Wrappers
# ============================================================================

"""
    extract_kfs(y, r; factor_lags=1, maxiter=200, tol=1e-6, verbose=false)

Extract factors using Kalman filter/smoother via Siphon.jl's DynamicFactorModel.

# Arguments
- `y`: N × T data matrix (Siphon format: variables × time)
- `r`: Number of factors
- `factor_lags`: VAR order for factor dynamics (default: 1)
- `maxiter`: Maximum EM iterations (default: 200)
- `tol`: Convergence tolerance (default: 1e-6)
- `verbose`: Print progress (default: false)

# Returns
Named tuple with:
- `factors`: r × T smoothed factors
- `loadings`: N × r loading matrix (contemporaneous)
- `model`: Fitted DynamicFactorModel
- `converged`: Whether EM converged
- `loglik`: Final log-likelihood
- `time`: Computation time in seconds
"""
function extract_kfs(
        y::Matrix{Float64},
        r::Int;
        factor_lags::Int = 1,
        maxiter::Int = 200,
        tol::Float64 = 1e-6,
        verbose::Bool = false
)
    N, T = size(y)

    # Create model (static loadings, white noise errors - matching simulation DGP)
    model = DynamicFactorModel(
        N,
        r,
        T;
        loading_lags = 0,
        factor_lags = factor_lags,
        error_lags = 0,
        identification = :named_factor
    )

    # Fit with EM algorithm
    t_start = time()
    fit!(EM(), model, y; maxiter = maxiter, tol = tol, verbose = verbose)
    t_elapsed = time() - t_start

    # Extract smoothed factors (r × T)
    F = Siphon.factors(model)

    # Extract contemporaneous loadings (first element)
    Lambda = Siphon.loadings(model)[1]

    return (
        factors = F,
        loadings = Lambda,
        model = model,
        converged = isconverged(model),
        loglik = loglikelihood(model),
        time = t_elapsed
    )
end

"""
    extract_pca(y, r; demean=true, scale=false)

Extract factors using PCA via Factotum.jl.

# Arguments
- `y`: N × T data matrix (Siphon format: variables × time)
- `r`: Number of factors
- `demean`: Center data (default: true)
- `scale`: Standardize data (default: false)

# Returns
Named tuple with:
- `factors`: r × T factors (transposed from Factotum's T × r)
- `loadings`: N × r loading matrix
- `fm`: FactorModel object
- `time`: Computation time in seconds
"""
function extract_pca(y::Matrix{Float64}, r::Int; demean::Bool = true, scale::Bool = false)
    N, T = size(y)

    # Factotum expects T × N, so transpose
    y_transposed = permutedims(y)

    t_start = time()
    fm = FactorModel(y_transposed, r; demean = demean, scale = scale)
    t_elapsed = time() - t_start

    # Factotum returns T × r factors, we want r × T
    F = permutedims(Factotum.factors(fm))

    # Loadings are N × r in Factotum (same as we want)
    Lambda = Factotum.loadings(fm)

    return (factors = F, loadings = Lambda, fm = fm, time = t_elapsed)
end

# ============================================================================
# Section 3: Procrustes Alignment and Canonical Correlations
# ============================================================================

"""
    procrustes_align(F_est, F_true)

Align estimated factors to true factors via Procrustes rotation.

Solves: min_{R: R'R=I} ||F_true - R * F_est||_F

Uses SVD approach: if C = F_est * F_true', C = USV', then R = VU'

# Arguments
- `F_est`: r × T estimated factor matrix
- `F_true`: r × T true factor matrix

# Returns
Named tuple with:
- `F_aligned`: r × T rotated/aligned factor estimates
- `R`: r × r rotation matrix used
- `procrustes_mse`: Total MSE after Procrustes alignment
"""
function procrustes_align(F_est::Matrix{Float64}, F_true::Matrix{Float64})
    r, T = size(F_est)
    @assert size(F_true) == (r, T) "Dimensions must match"

    # Standardize both (important for numerical stability)
    F_est_centered = F_est .- mean(F_est, dims = 2)
    F_true_centered = F_true .- mean(F_true, dims = 2)

    # Compute cross-covariance: C = F_est * F_true' (r × r)
    C = F_est_centered * F_true_centered' / T

    # SVD of cross-covariance
    U, S, Vt = svd(C)
    V = Vt'

    # Optimal rotation: R = V * U'
    R = V * U'

    # Check determinant - if negative, flip sign of last column to get proper rotation
    if det(R) < 0
        V[:, end] *= -1
        R = V * U'
    end

    # Apply rotation
    F_aligned = R * F_est

    # Compute total MSE after alignment (this is meaningful for any r)
    procrustes_mse = mean((F_aligned .- F_true) .^ 2)

    return (F_aligned = F_aligned, R = R, procrustes_mse = procrustes_mse)
end

"""
    canonical_correlations(F_est, F_true)

Compute canonical correlations between estimated and true factor spaces.

This is the correct way to measure similarity between factor spaces when r > 1,
as it is invariant to rotations within each space.

# Arguments
- `F_est`: r × T estimated factor matrix
- `F_true`: r × T true factor matrix

# Returns
Named tuple with:
- `correlations`: Vector of r canonical correlations (sorted descending)
- `avg_correlation`: Mean of canonical correlations
- `min_correlation`: Minimum canonical correlation (weakest link)
"""
function canonical_correlations(F_est::Matrix{Float64}, F_true::Matrix{Float64})
    r, T = size(F_est)
    @assert size(F_true) == (r, T) "Dimensions must match"

    # Center the factors
    F_est_c = F_est .- mean(F_est, dims = 2)
    F_true_c = F_true .- mean(F_true, dims = 2)

    # Compute covariance matrices (r × r)
    Σ_ee = F_est_c * F_est_c' / T      # Var(F_est)
    Σ_tt = F_true_c * F_true_c' / T    # Var(F_true)
    Σ_et = F_est_c * F_true_c' / T     # Cov(F_est, F_true)

    # Canonical correlations via generalized eigenvalue problem
    # ρ² are eigenvalues of Σ_ee^{-1/2} Σ_et Σ_tt^{-1} Σ_te Σ_ee^{-1/2}

    # Use SVD for numerical stability
    # Σ_ee = U_e S_e U_e', so Σ_ee^{-1/2} = U_e S_e^{-1/2} U_e'
    U_e, S_e, _ = svd(Σ_ee)
    U_t, S_t, _ = svd(Σ_tt)

    # Regularize small singular values
    tol = 1e-10
    S_e_inv_sqrt = Diagonal([s > tol ? 1/sqrt(s) : 0.0 for s in S_e])
    S_t_inv_sqrt = Diagonal([s > tol ? 1/sqrt(s) : 0.0 for s in S_t])

    Σ_ee_inv_sqrt = U_e * S_e_inv_sqrt * U_e'
    Σ_tt_inv_sqrt = U_t * S_t_inv_sqrt * U_t'

    # Form the matrix whose singular values are the canonical correlations
    M = Σ_ee_inv_sqrt * Σ_et * Σ_tt_inv_sqrt

    # Canonical correlations are singular values of M
    canon_corrs = svdvals(M)

    # Clamp to [0, 1] for numerical stability
    canon_corrs = clamp.(canon_corrs, 0.0, 1.0)

    # Sort descending
    sort!(canon_corrs, rev = true)

    return (
        correlations = canon_corrs,
        avg_correlation = mean(canon_corrs),
        min_correlation = minimum(canon_corrs)
    )
end

# ============================================================================
# Section 4: Comparison Metrics
# ============================================================================

"""
    compute_metrics(F_est, F_true, Lambda_est, Lambda_true)

Compute comparison metrics between estimated and true factor models.

Uses rotation-invariant metrics that work correctly for any number of factors r.

# Returns
Named tuple with:
- `mse_factors`: Mean squared error of factors (after Procrustes alignment)
- `canonical_corrs`: Vector of canonical correlations (rotation-invariant)
- `avg_canonical_corr`: Average canonical correlation
- `min_canonical_corr`: Minimum canonical correlation (weakest factor recovery)
- `space_r2`: Factor space R² (rotation-invariant)
- `mse_loadings`: MSE of loadings (after rotation alignment)
"""
function compute_metrics(
        F_est::Matrix{Float64},
        F_true::Matrix{Float64},
        Lambda_est::Matrix{Float64},
        Lambda_true::Matrix{Float64}
)
    r, T = size(F_est)
    N, _ = size(Lambda_est)

    # 1. Procrustes alignment for MSE
    aligned = procrustes_align(F_est, F_true)
    mse_factors = aligned.procrustes_mse
    R = aligned.R

    # 2. Canonical correlations (rotation-invariant measure of factor recovery)
    cc = canonical_correlations(F_est, F_true)

    # 3. Factor space R² (rotation-invariant measure)
    # R² = tr(P_true * P_est) / r
    # where P = F'(FF')^{-1}F is the projection matrix onto column space of F'
    # Simplified for computational stability using trace formula

    # Use: trace(P1 * P2) = trace((F1'F1)^{-1} F1'F2 (F2'F2)^{-1} F2'F1)
    G1 = F_true * F_true'  # r × r
    G2 = F_est * F_est'    # r × r
    C12 = F_true * F_est'  # r × r

    # Avoid singularity issues
    G1_inv = pinv(G1)
    G2_inv = pinv(G2)

    space_r2 = tr(G1_inv * C12 * G2_inv * C12') / r

    # 4. Loadings comparison (rotate loadings consistently with Procrustes)
    Lambda_rotated = Lambda_est * R'
    mse_loadings = mean((Lambda_rotated .- Lambda_true) .^ 2)

    return (
        mse_factors = mse_factors,
        canonical_corrs = cc.correlations,
        avg_canonical_corr = cc.avg_correlation,
        min_canonical_corr = cc.min_correlation,
        space_r2 = space_r2,
        mse_loadings = mse_loadings
    )
end

# ============================================================================
# Section 5: Monte Carlo Simulation
# ============================================================================

"""
    run_monte_carlo(; N_values, T, r, phi, n_reps, seed, verbose)

Run Monte Carlo comparison of KFS vs PCA factor extraction.

Following simulation design from Poncela et al. (2021) Section 3.

# Arguments
- `N_values`: Cross-sectional dimensions to test (default: [5, 50, 150])
- `T`: Number of time periods (default: 200)
- `r`: Number of factors (default: 1)
- `phi`: AR(1) coefficient for factors (default: 0.7)
- `n_reps`: Number of Monte Carlo replications (default: 20)
- `seed`: Random seed (default: 42)
- `verbose`: Print progress (default: true)

# Returns
Vector of named tuples with results for each (N, method, replication)
"""
function run_monte_carlo(;
        N_values::Vector{Int} = [5, 50, 150],
        T::Int = 200,
        r::Int = 1,
        phi::Float64 = 0.7,
        n_reps::Int = 20,
        seed::Int = 42,
        verbose::Bool = true
)
    rng = MersenneTwister(seed)

    results = []

    for N in N_values
        if verbose
            println()
            println("=" ^ 60)
            println("N = $N, T = $T, r = $r, φ = $phi")
            println("=" ^ 60)
        end

        for rep in 1:n_reps
            # Generate data
            data = simulate_dfm(N, T, r; phi = phi, rng = rng)
            y, f_true, Lambda_true = data.y, data.f_true, data.Lambda_true

            # KFS extraction
            kfs_result = extract_kfs(y, r; factor_lags = 1, verbose = false)
            kfs_metrics = compute_metrics(
                kfs_result.factors,
                f_true,
                kfs_result.loadings,
                Lambda_true
            )

            push!(
                results,
                (
                    N = N,
                    T = T,
                    r = r,
                    phi = phi,
                    rep = rep,
                    method = "KFS",
                    mse_factors = kfs_metrics.mse_factors,
                    avg_canonical_corr = kfs_metrics.avg_canonical_corr,
                    min_canonical_corr = kfs_metrics.min_canonical_corr,
                    space_r2 = kfs_metrics.space_r2,
                    time = kfs_result.time,
                    converged = kfs_result.converged
                )
            )

            # PCA extraction
            pca_result = extract_pca(y, r)
            pca_metrics = compute_metrics(
                pca_result.factors,
                f_true,
                pca_result.loadings,
                Lambda_true
            )

            push!(
                results,
                (
                    N = N,
                    T = T,
                    r = r,
                    phi = phi,
                    rep = rep,
                    method = "PCA",
                    mse_factors = pca_metrics.mse_factors,
                    avg_canonical_corr = pca_metrics.avg_canonical_corr,
                    min_canonical_corr = pca_metrics.min_canonical_corr,
                    space_r2 = pca_metrics.space_r2,
                    time = pca_result.time,
                    converged = true
                )
            )

            if verbose && rep % 5 == 0
                println("  Completed $rep / $n_reps replications")
            end
        end
    end

    return results
end

"""
    summarize_results(results; r=1)

Print summary statistics from Monte Carlo simulation.

# Arguments
- `results`: Vector of result named tuples from run_monte_carlo
- `r`: Number of factors (for display purposes)
"""
function summarize_results(results; r::Int = 1)
    println()
    println("=" ^ 70)
    println("MONTE CARLO RESULTS SUMMARY (r = $r factor$(r > 1 ? "s" : ""))")
    println("=" ^ 70)
    println()

    # Header - adjust based on r
    if r == 1
        @printf("| %-6s | %-6s | %-12s | %-12s | %-10s | %-10s |\n",
            "N",
            "Method",
            "MSE",
            "Avg Corr",
            "Space R²",
            "Time (s)")
    else
        @printf("| %-6s | %-6s | %-12s | %-12s | %-12s | %-10s |\n",
            "N",
            "Method",
            "MSE",
            "Avg CC",
            "Min CC",
            "Space R²")
    end
    println(
        "|" *
        "-"^8 *
        "|" *
        "-"^8 *
        "|" *
        "-"^14 *
        "|" *
        "-"^14 *
        "|" *
        "-"^14 *
        "|" *
        "-"^12 *
        "|",
    )

    # Group by N and method
    for N in unique(res.N for res in results)
        for method in ["KFS", "PCA"]
            subset = filter(res -> res.N == N && res.method == method, results)

            mse_mean = mean(res.mse_factors for res in subset)
            mse_std = std(res.mse_factors for res in subset)
            avg_cc_mean = mean(res.avg_canonical_corr for res in subset)
            avg_cc_std = std(res.avg_canonical_corr for res in subset)
            min_cc_mean = mean(res.min_canonical_corr for res in subset)
            min_cc_std = std(res.min_canonical_corr for res in subset)
            r2_mean = mean(res.space_r2 for res in subset)
            time_mean = mean(res.time for res in subset)

            if r == 1
                @printf("| %6d | %-6s | %5.4f (%.3f) | %5.4f (%.3f) | %10.4f | %10.4f |\n",
                    N,
                    method,
                    mse_mean,
                    mse_std,
                    avg_cc_mean,
                    avg_cc_std,
                    r2_mean,
                    time_mean)
            else
                @printf("| %6d | %-6s | %5.4f (%.3f) | %5.4f (%.3f) | %5.4f (%.3f) | %10.4f |\n",
                    N,
                    method,
                    mse_mean,
                    mse_std,
                    avg_cc_mean,
                    avg_cc_std,
                    min_cc_mean,
                    min_cc_std,
                    r2_mean)
            end
        end
        println(
            "|" *
            "-"^8 *
            "|" *
            "-"^8 *
            "|" *
            "-"^14 *
            "|" *
            "-"^14 *
            "|" *
            "-"^14 *
            "|" *
            "-"^12 *
            "|",
        )
    end

    # Key observations
    println()
    println("Key Observations (from Poncela et al. 2021):")
    println("- For small N: KFS exploits factor dynamics, may outperform PCA")
    println("- For large N: PCA approaches efficiency (N→∞ asymptotic theory)")
    println("- MSE decreases with N as more information available")
    if r > 1
        println("- Avg CC: average canonical correlation (rotation-invariant)")
        println("- Min CC: minimum canonical correlation (weakest factor recovery)")
    end
    println("- KFS provides uncertainty quantification (MSE bounds) unlike PCA")
end

# ============================================================================
# Section 6: Real Data Application
# ============================================================================

"""
    real_data_comparison(data_path; n_factors=6)

Apply both factor extraction methods to real macroeconomic data.
"""
function real_data_comparison(data_path::String; n_factors::Int = 6)
    println()
    println("=" ^ 70)
    println("REAL DATA COMPARISON (FRED-QD)")
    println("=" ^ 70)
    println()

    # Load data (same preprocessing as dfm_full_estimation.jl)
    raw_data = readdlm(data_path, ',', Any; header = true)
    data_matrix = raw_data[1]
    header = raw_data[2]

    var_names = String.(header[2:end])
    y_raw = Matrix{Float64}(data_matrix[:, 2:end])
    T_obs, n_vars = size(y_raw)

    # Transpose to N × T for Siphon
    y_raw_t = permutedims(y_raw)

    # Handle missing values (keep variables with ≤10% missing)
    missing_counts = [count(isnan, y_raw_t[i, :]) for i in 1:n_vars]
    max_missing = div(T_obs, 10)
    valid_vars = findall(x -> x <= max_missing, missing_counts)

    y = y_raw_t[valid_vars, :]
    N, n = size(y)

    println("Data dimensions: N=$N variables × T=$n time periods")
    println("Extracting $n_factors factors...")
    println()

    # Standardize (handling NaN)
    for i in 1:N
        valid_obs = filter(!isnan, y[i, :])
        μ = mean(valid_obs)
        σ = std(valid_obs)
        σ = σ < 1e-10 ? 1.0 : σ
        for t in 1:n
            if !isnan(y[i, t])
                y[i, t] = (y[i, t] - μ) / σ
            end
        end
    end

    # For PCA, we need complete cases (no NaN) - create a clean version
    # by replacing NaN with 0 (since data is standardized, 0 ≈ mean)
    y_clean = copy(y)
    y_clean[isnan.(y_clean)] .= 0.0

    # KFS extraction (handles missing values natively)
    println("Running KFS (Siphon.jl)...")
    kfs_result = extract_kfs(y, n_factors; factor_lags = 2, maxiter = 200, verbose = true)
    println()

    # PCA extraction (on cleaned data)
    println("Running PCA (Factotum.jl)...")
    pca_result = extract_pca(y_clean, n_factors)
    println()

    # Compare results
    println("-" ^ 50)
    println("COMPARISON")
    println("-" ^ 50)

    @printf("\nComputation time:\n")
    @printf("  KFS: %.3f seconds (EM iterations)\n", kfs_result.time)
    @printf("  PCA: %.3f seconds\n", pca_result.time)

    # Canonical correlations between KFS and PCA factor spaces
    println("\nCanonical correlations between KFS and PCA factors:")
    cc = canonical_correlations(kfs_result.factors, pca_result.factors)
    for i in 1:n_factors
        @printf("  Canonical correlation %d: %.4f\n", i, cc.correlations[i])
    end
    @printf("  Average: %.4f, Minimum: %.4f\n", cc.avg_correlation, cc.min_correlation)

    # Variance explained by PCA factors
    println("\nVariance explained by PCA factors:")
    fm = pca_result.fm
    # Compute eigenvalues from loadings
    Lambda = Factotum.loadings(fm)
    eigvals_approx = vec(sum(Lambda .^ 2, dims = 1))
    total_var = sum(eigvals_approx)
    for i in 1:n_factors
        pct = eigvals_approx[i] / total_var * 100
        @printf("  Factor %d: %.1f%%\n", i, pct)
    end
    @printf("  Total: %.1f%%\n", sum(eigvals_approx[1:n_factors]) / total_var * 100)

    # KFS model diagnostics
    println("\nKFS Model Diagnostics:")
    @printf("  Converged: %s\n", kfs_result.converged)
    @printf("  Log-likelihood: %.2f\n", kfs_result.loglik)

    return (kfs = kfs_result, pca = pca_result, y = y, y_clean = y_clean)
end

# ============================================================================
# Section 7: Information Criteria for Factor Selection
# ============================================================================

"""
    demonstrate_ic(y; kmax=10)

Demonstrate Factotum.jl's information criteria for determining number of factors.

Based on Bai & Ng (2002): "Determining the Number of Factors in
Approximate Factor Models" Econometrica 70(1): 191-221.
"""
function demonstrate_ic(y::Matrix{Float64}; kmax::Int = 10)
    println()
    println("=" ^ 70)
    println("INFORMATION CRITERIA FOR FACTOR SELECTION")
    println("=" ^ 70)
    println()

    N, T = size(y)
    println("Data: N=$N variables, T=$T observations")
    println("Testing k = 1, 2, ..., $kmax factors")
    println()

    # Transpose for Factotum (needs T × N)
    y_t = permutedims(y)

    # Compute multiple criteria
    println("Computing information criteria...")

    # Factotum IC computes all values at once for k=0:kmax
    # Use the IC objects directly
    ic1_obj = Factotum.IC1(y_t, kmax)
    ic2_obj = Factotum.IC2(y_t, kmax)
    bic3_obj = Factotum.BIC3(y_t, kmax)

    # Extract criterion values (index 1 = k=0, index 2 = k=1, etc.)
    # We want k=1:kmax, so indices 2:(kmax+1)
    ic1_vals = ic1_obj.crit[2:end]
    ic2_vals = ic2_obj.crit[2:end]
    bic3_vals = bic3_obj.crit[2:end]

    # Find optimal k for each criterion (minimize over k=1:kmax)
    k_ic1 = argmin(ic1_vals)
    k_ic2 = argmin(ic2_vals)
    k_bic3 = argmin(bic3_vals)

    # Print results table
    @printf("\n| %-3s | %-12s | %-12s | %-12s |\n", "k", "IC1", "IC2", "BIC3")
    println("|" * "-"^5 * "|" * "-"^14 * "|" * "-"^14 * "|" * "-"^14 * "|")

    for k in 1:kmax
        marker_ic1 = k == k_ic1 ? "*" : " "
        marker_ic2 = k == k_ic2 ? "*" : " "
        marker_bic3 = k == k_bic3 ? "*" : " "

        @printf("| %3d | %11.4f%s | %11.4f%s | %11.4f%s |\n",
            k,
            ic1_vals[k],
            marker_ic1,
            ic2_vals[k],
            marker_ic2,
            bic3_vals[k],
            marker_bic3)
    end

    println()
    println("Optimal number of factors:")
    println("  IC1:  k = $k_ic1")
    println("  IC2:  k = $k_ic2")
    println("  BIC3: k = $k_bic3")
    println()
    println("Note: * indicates minimum value for each criterion")
    println("These criteria follow Bai & Ng (2002) methodology.")

    return (
        k_ic1 = k_ic1,
        k_ic2 = k_ic2,
        k_bic3 = k_bic3,
        ic1 = ic1_vals,
        ic2 = ic2_vals,
        bic3 = bic3_vals
    )
end

# ============================================================================
# Section 8: Main Script
# ============================================================================

function main()
    println("=" ^ 70)
    println("FACTOR EXTRACTION COMPARISON")
    println("Dynamic (Kalman Filter/Smoother) vs Static (PCA)")
    println("=" ^ 70)
    println()
    println("Based on: Poncela, Ruiz & Miranda (2021)")
    println("'Factor Extraction Using Kalman Filter and Smoothing'")
    println("International Journal of Forecasting 37: 1399-1425")

    # ========================================
    # Part 1a: Monte Carlo Simulation (r=1)
    # ========================================
    println("\n")
    println("=" ^ 70)
    println("PART 1a: MONTE CARLO SIMULATION (Single Factor)")
    println("=" ^ 70)
    println()
    println("Simulation design (from paper Section 3):")
    println("  - N ∈ {5, 50, 150} (cross-sectional dimension)")
    println("  - T = 200 (time periods)")
    println("  - r = 1 (single factor)")
    println("  - φ = 0.7 (AR(1) factor persistence)")
    println("  - σ_e = 0.5 (idiosyncratic error std)")
    println("  - 20 Monte Carlo replications")

    results_r1 = run_monte_carlo(
        N_values = [5, 50, 150],
        T = 200,
        r = 1,
        phi = 0.7,
        n_reps = 20,
        verbose = true
    )

    summarize_results(results_r1; r = 1)

    # ========================================
    # Part 1b: Monte Carlo Simulation (r=2)
    # ========================================
    println("\n")
    println("=" ^ 70)
    println("PART 1b: MONTE CARLO SIMULATION (Two Factors)")
    println("=" ^ 70)
    println()
    println("Simulation design:")
    println("  - N ∈ {10, 50, 150} (cross-sectional dimension)")
    println("  - T = 200 (time periods)")
    println("  - r = 2 (two factors)")
    println("  - φ = 0.7 (AR(1) factor persistence)")
    println("  - σ_e = 0.5 (idiosyncratic error std)")
    println("  - 20 Monte Carlo replications")
    println()
    println("Note: For r > 1, we use canonical correlations (rotation-invariant)")
    println("      instead of simple correlations.")

    results_r2 = run_monte_carlo(
        N_values = [10, 50, 150],  # N=5 too small for r=2
        T = 200,
        r = 2,
        phi = 0.7,
        n_reps = 20,
        verbose = true
    )

    summarize_results(results_r2; r = 2)

    # ========================================
    # Part 2: Real Data Application
    # ========================================
    println("\n")
    println("=" ^ 70)
    println("PART 2: REAL DATA APPLICATION")
    println("=" ^ 70)

    data_path = joinpath(@__DIR__, "qt_factor_data.csv")
    if isfile(data_path)
        real_results = real_data_comparison(data_path; n_factors = 6)

        # ========================================
        # Part 3: Information Criteria Demo
        # ========================================
        println("\n")
        println("=" ^ 70)
        println("PART 3: FACTOR SELECTION VIA INFORMATION CRITERIA")
        println("=" ^ 70)

        ic_results = demonstrate_ic(real_results.y_clean; kmax = 10)
    else
        println()
        println("Data file not found: $data_path")
        println("Skipping real data comparison and IC demo")
    end

    # ========================================
    # Summary
    # ========================================
    println("\n")
    println("=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println()
    println("Key takeaways from Poncela et al. (2021):")
    println()
    println("1. KFS ADVANTAGES:")
    println("   - Efficient when factors are serially correlated")
    println("   - Handles missing data and mixed frequencies naturally")
    println("   - Provides MSE estimates for uncertainty quantification")
    println("   - Allows constraints on loadings and dynamics")
    println()
    println("2. PCA ADVANTAGES:")
    println("   - Computationally faster (no iterative estimation)")
    println("   - Robust to misspecification of dynamics")
    println("   - Well-established asymptotic theory (N,T → ∞)")
    println("   - Information criteria for selecting # of factors")
    println()
    println("3. PRACTICAL GUIDANCE:")
    println("   - Small N: KFS may be more efficient")
    println("   - Large N: PCA approaches efficiency")
    println("   - Missing data: Use KFS")
    println("   - Quick analysis: Use PCA")
    println("   - Uncertainty quantification: Use KFS")
    println()
    println("=" ^ 70)
    println("COMPARISON COMPLETE")
    println("=" ^ 70)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
