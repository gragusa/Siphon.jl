#=
Dynamic Factor Model Estimation on FRED-QD Data

Estimates a dynamic factor model using the high-level DynamicFactorModel API.
- 6 factors
- 3 lags (VAR(3) factor dynamics)
- Data: Quarterly macroeconomic data from FRED-QD

This example demonstrates:
1. Loading and preprocessing macroeconomic data
2. Using DynamicFactorModel with fit!(EM(), ...)
3. Extracting and analyzing estimated factors and loadings
=#

using DelimitedFiles
using LinearAlgebra
using Statistics
using Printf
using Siphon

println("=" ^ 70)
println("Dynamic Factor Model Estimation on FRED-QD Data")
println("=" ^ 70)
println()

# ============================================
# Load and preprocess data
# ============================================

println("Loading data...")
data_path = joinpath(@__DIR__, "qt_factor_data.csv")

# Read CSV - first row is header, first column is date
raw_data = readdlm(data_path, ',', Any; header = true)
data_matrix = raw_data[1]
header = raw_data[2]

# Extract variable names (skip DATE column)
var_names = String.(header[2:end])
println("  Variables: ", length(var_names))

# Extract numeric data (skip DATE column)
y_raw = Matrix{Float64}(data_matrix[:, 2:end])
T_obs, n_vars = size(y_raw)
println("  Time periods: ", T_obs)
println("  Raw dimensions: $n_vars × $T_obs")

# Transpose to N × n format (variables × time)
y_raw_t = permutedims(y_raw)

# ============================================
# Handle missing values
# ============================================

# Count missing values per variable
missing_counts = [count(isnan, y_raw_t[i, :]) for i in 1:n_vars]
println()
println("Missing data summary:")
println("  Variables with no missing: ", count(==(0), missing_counts))
println("  Variables with some missing: ", count(>(0), missing_counts))
println("  Max missing per variable: ", maximum(missing_counts))

# For simplicity, we'll use variables with at most 10% missing
max_missing = div(T_obs, 10)
valid_vars = findall(x -> x <= max_missing, missing_counts)
println("  Variables with ≤10% missing: ", length(valid_vars))

# Select valid variables
y = y_raw_t[valid_vars, :]
N, n = size(y)
selected_names = var_names[valid_vars]

println()
println("Selected data dimensions: N=$N variables × n=$n time periods")

# ============================================
# Standardize data (mean 0, std 1 per variable)
# ============================================

println()
println("Standardizing data...")

# Compute means and stds ignoring NaN
means = zeros(N)
stds = zeros(N)
for i in 1:N
    valid_obs = filter(!isnan, y[i, :])
    means[i] = mean(valid_obs)
    stds[i] = std(valid_obs)
    if stds[i] < 1e-10
        stds[i] = 1.0  # Avoid division by zero
    end
end

# Standardize
y_std = similar(y)
for i in 1:N
    for t in 1:n
        if isnan(y[i, t])
            y_std[i, t] = NaN
        else
            y_std[i, t] = (y[i, t] - means[i]) / stds[i]
        end
    end
end

println("  Data standardized (mean=0, std=1 per variable)")

# ============================================
# Setup and run EM
# ============================================

n_factors = 6
n_lags = 3

println()
println("Model specification:")
println("  Number of factors (k): ", n_factors)
println("  Factor VAR lags (q): ", n_lags)
println("  State dimension (m = k×q): ", n_factors * n_lags)
println()

# Create DynamicFactorModel
println("Creating DynamicFactorModel...")
model = DynamicFactorModel(
    N,
    n_factors,
    n;
    loading_lags = 0,    # Static loadings
    factor_lags = n_lags,
    error_lags = 0
)      # White noise errors

# Run EM estimation
println("Running EM algorithm...")
println("-" ^ 50)

t_start = time()
fit!(EM(), model, y_std; maxiter = 100, tol = 1e-6, verbose = true)
t_elapsed = time() - t_start

println("-" ^ 50)
println()
println("EM Results:")
println("  Converged: ", isconverged(model))
println("  Iterations: ", niterations(model))
println("  Final log-likelihood: ", round(loglikelihood(model), digits = 2))
println("  Total time: ", round(t_elapsed, digits = 2), " seconds")
println(
    "  Time per iteration: ",
    round(t_elapsed / niterations(model) * 1000, digits = 2),
    " ms"
)
println()

# ============================================
# Extract and display results
# ============================================

# Extract factor loadings (tuple of N × k matrices, one per lag)
Λ_tuple = loadings(model)
Λ = Λ_tuple[1]  # Contemporaneous loadings (static model has only one)

# Extract smoothed factors (k × n)
smoothed_factors = factors(model)

# Extract VAR coefficients for factors
Φ = var_coefficients(model)

println("Factor loadings (Λ) - first 10 variables:")
println("-" ^ 50)
for i in 1:min(10, N)
    name = selected_names[i]
    loading_vals = round.(Λ[i, :], digits = 3)
    println("  ", rpad(name[1:min(20, length(name))], 22), loading_vals)
end
println("  ...")
println()

println("Factor VAR coefficients:")
for (lag, Φ_lag) in enumerate(Φ)
    println("  Φ_$lag (diagonal): ", round.(diag(Φ_lag), digits = 3))
end
println()

# Factor innovation covariance
Σ_η = innovation_cov(model)
println("Factor innovation covariance Σ_η (diagonal):")
println("  ", round.(diag(Σ_η), digits = 4))
println()

# Variance decomposition: how much variance explained by factors
println("Variance explained by factors:")
# Var(y_i) = Λ_i' * Var(f) * Λ_i + ψ_i
# For standardized data, Var(y_i) ≈ 1

# Estimate factor covariance from smoothed factors
factor_cov = (smoothed_factors * smoothed_factors') / n

# Compute communalities properly
communalities = zeros(N)
for i in 1:N
    λ_i = Λ[i, :]
    communalities[i] = λ_i' * factor_cov * λ_i
end

# Idiosyncratic variances
idio_variances = idiosyncratic_variances(model)
# Total variance for each variable
total_var = communalities .+ idio_variances
var_explained_pct = communalities ./ total_var

println(
    "  Average variance explained by factors: ",
    round(mean(var_explained_pct) * 100, digits = 1),
    "%"
)
println(
    "  Min variance explained: ",
    round(minimum(var_explained_pct) * 100, digits = 1),
    "%"
)
println(
    "  Max variance explained: ",
    round(maximum(var_explained_pct) * 100, digits = 1),
    "%"
)
println()

# Show top-loading variables for each factor
println("Top 5 variables loading on each factor:")
println("-" ^ 50)
for k in 1:n_factors
    loadings_k = abs.(Λ[:, k])
    top_idx = sortperm(loadings_k, rev = true)[1:5]
    println("Factor $k:")
    for idx in top_idx
        name = selected_names[idx]
        println(
            "  ",
            rpad(name[1:min(25, length(name))], 27),
            "loading = ",
            round(Λ[idx, k], digits = 3)
        )
    end
    println()
end

# Factor summary statistics
println("Factor summary statistics:")
println("-" ^ 50)
for k in 1:n_factors
    f_k = smoothed_factors[k, :]
    println(
        "  Factor $k: mean=$(round(mean(f_k), digits=3)), " *
        "std=$(round(std(f_k), digits=3)), " *
        "min=$(round(minimum(f_k), digits=2)), " *
        "max=$(round(maximum(f_k), digits=2))",
    )
end
println()

println("=" ^ 70)
println("Estimation complete!")
println("=" ^ 70)
