#=
Full Dynamic Factor Model Estimation on FRED-QD Data

Estimates dynamic factor models with various configurations:
- Model 1: Simple DFM (static loadings, white noise errors)
- Model 2: DFM with AR(1) idiosyncratic errors
- Model 3: DFM with dynamic loadings (1 lag)
- Model 4: Full DFM (dynamic loadings + AR errors)

Uses the DynamicFactorModel type with fit!(EM(), ...) interface.

Data: Quarterly macroeconomic data from FRED-QD
=#

using DelimitedFiles
using LinearAlgebra
using Statistics
using Printf
using Siphon

println("=" ^ 70)
println("Full Dynamic Factor Model Estimation on FRED-QD Data")
println("=" ^ 70)
println()

# ============================================
# Load and preprocess data
# ============================================

println("Loading data...")
data_path = joinpath(@__DIR__, "qt_factor_data.csv")

# Read CSV - first row is header, first column is date
raw_data = readdlm(data_path, ',', Any; header=true)
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
# Model specifications
# ============================================

n_factors = 6
factor_lags = 3

# ============================================
# Model 1: Simple DFM (baseline)
# ============================================

println()
println("=" ^ 70)
println("Model 1: Simple DFM (static loadings, white noise errors)")
println("=" ^ 70)

println()
println("Model specification:")
println("  Number of factors (k): ", n_factors)
println("  Factor VAR lags (q): ", factor_lags)
println("  Loading lags (p): 0 (static)")
println("  Error AR lags (r): 0 (white noise)")
println()

model1 = DynamicFactorModel(N, n_factors, n;
                            loading_lags=0,
                            factor_lags=factor_lags,
                            error_lags=0)

t_start = time()
fit!(EM(), model1, y_std; maxiter=200, tol=1e-6, verbose=true)
t_elapsed = time() - t_start

println()
println("Model 1 Results:")
println("  Converged: ", isconverged(model1))
println("  Iterations: ", niterations(model1))
println("  Final log-likelihood: ", round(loglikelihood(model1), digits=2))
println("  Total time: ", round(t_elapsed, digits=2), " seconds")
println()

# ============================================
# Model 2: DFM with AR(1) errors
# ============================================

println()
println("=" ^ 70)
println("Model 2: DFM with AR(1) idiosyncratic errors")
println("=" ^ 70)

println()
println("Model specification:")
println("  Number of factors (k): ", n_factors)
println("  Factor VAR lags (q): ", factor_lags)
println("  Loading lags (p): 0 (static)")
println("  Error AR lags (r): 1")
println()

model2 = DynamicFactorModel(N, n_factors, n;
                            loading_lags=0,
                            factor_lags=factor_lags,
                            error_lags=1)

t_start = time()
fit!(EM(), model2, y_std; maxiter=200, tol=1e-6, verbose=true)
t_elapsed = time() - t_start

println()
println("Model 2 Results:")
println("  Converged: ", isconverged(model2))
println("  Iterations: ", niterations(model2))
println("  Final log-likelihood: ", round(loglikelihood(model2), digits=2))
println("  Total time: ", round(t_elapsed, digits=2), " seconds")
println()
δ2 = ar_coefficients(model2)
if !isempty(δ2)
    println("  AR(1) error coefficient δ₁: ", round(δ2[1], digits=4))
end
println()

# ============================================
# Model 3: DFM with dynamic loadings
# ============================================

println()
println("=" ^ 70)
println("Model 3: DFM with dynamic loadings (1 lag)")
println("=" ^ 70)

println()
println("Model specification:")
println("  Number of factors (k): ", n_factors)
println("  Factor VAR lags (q): ", factor_lags)
println("  Loading lags (p): 1 (dynamic)")
println("  Error AR lags (r): 0")
println()

model3 = DynamicFactorModel(N, n_factors, n;
                            loading_lags=1,
                            factor_lags=factor_lags,
                            error_lags=0)

t_start = time()
fit!(EM(), model3, y_std; maxiter=200, tol=1e-6, verbose=true)
t_elapsed = time() - t_start

println()
println("Model 3 Results:")
println("  Converged: ", isconverged(model3))
println("  Iterations: ", niterations(model3))
println("  Final log-likelihood: ", round(loglikelihood(model3), digits=2))
println("  Total time: ", round(t_elapsed, digits=2), " seconds")
println()

# ============================================
# Model 4: Full DFM (dynamic loadings + AR errors)
# ============================================

println()
println("=" ^ 70)
println("Model 4: Full DFM (dynamic loadings + AR errors)")
println("=" ^ 70)

println()
println("Model specification:")
println("  Number of factors (k): ", n_factors)
println("  Factor VAR lags (q): ", factor_lags)
println("  Loading lags (p): 1 (dynamic)")
println("  Error AR lags (r): 1")
println()

model4 = DynamicFactorModel(N, n_factors, n;
                            loading_lags=1,
                            factor_lags=factor_lags,
                            error_lags=1)

t_start = time()
fit!(EM(), model4, y_std; maxiter=200, tol=1e-6, verbose=true)
t_elapsed = time() - t_start

println()
println("Model 4 Results:")
println("  Converged: ", isconverged(model4))
println("  Iterations: ", niterations(model4))
println("  Final log-likelihood: ", round(loglikelihood(model4), digits=2))
println("  Total time: ", round(t_elapsed, digits=2), " seconds")
println()
δ4 = ar_coefficients(model4)
if !isempty(δ4)
    println("  AR(1) error coefficient δ₁: ", round(δ4[1], digits=4))
end
println()

# ============================================
# Model comparison
# ============================================

println()
println("=" ^ 70)
println("Model Comparison Summary")
println("=" ^ 70)
println()

# Compute state dimensions for each model
function state_dim(p, q, r, k, N)
    s = max(q, p + 1)
    m_f = k * s
    m_e = N * r
    return m_f + m_e
end

models = [
    ("Simple DFM (p=0, r=0)", model1, state_dim(0, factor_lags, 0, n_factors, N)),
    ("AR(1) errors (p=0, r=1)", model2, state_dim(0, factor_lags, 1, n_factors, N)),
    ("Dynamic loadings (p=1, r=0)", model3, state_dim(1, factor_lags, 0, n_factors, N)),
    ("Full DFM (p=1, r=1)", model4, state_dim(1, factor_lags, 1, n_factors, N)),
]

println("| Model                        | LogLik       | Iterations | State Dim |")
println("|------------------------------|--------------|------------|-----------|")
for (name, model, m) in models
    ll_str = @sprintf("%.2f", loglikelihood(model))
    println("| ", rpad(name, 28), " | ", lpad(ll_str, 12), " | ",
            lpad(string(niterations(model)), 10), " | ", lpad(string(m), 9), " |")
end

# Best model by log-likelihood
best_idx = argmax([loglikelihood(m) for (_, m, _) in models])
println()
println("Best model by log-likelihood: ", models[best_idx][1])

# ============================================
# Display best model parameters
# ============================================

best_model = models[best_idx][2]
best_name = models[best_idx][1]

println()
println("=" ^ 70)
println("Best Model Parameters: $best_name")
println("=" ^ 70)
println()

# Factor VAR coefficients
Φ = var_coefficients(best_model)
println("Factor VAR coefficients (diagonal of Φ matrices):")
for (lag, Φ_lag) in enumerate(Φ)
    println("  Φ_$lag diagonal: ", round.(diag(Φ_lag), digits=3))
end
println()

# Factor innovation covariance
Σ_η = innovation_cov(best_model)
println("Factor innovation covariance Σ_η (diagonal):")
println("  ", round.(diag(Σ_η), digits=4))
println()

# AR error coefficients (if applicable)
δ = ar_coefficients(best_model)
if !isempty(δ) && !all(δ .== 0)
    println("AR error coefficients δ:")
    println("  ", round.(δ, digits=4))
    println()
end

# Factor loadings summary
Λ = loadings(best_model)
println("Factor loadings Λ₀ summary:")
Λ0 = Λ[1]  # First element is Λ₀ (contemporaneous loadings)
for k in 1:n_factors
    loadings_k = Λ0[:, k]
    println("  Factor $k: mean=$(round(mean(loadings_k), digits=3)), " *
            "std=$(round(std(loadings_k), digits=3)), " *
            "max|λ|=$(round(maximum(abs, loadings_k), digits=3))")
end
println()

# Factor summary statistics
smoothed_factors = factors(best_model)
println("Smoothed factor summary:")
for k in 1:n_factors
    f_k = smoothed_factors[k, :]
    println("  Factor $k: mean=$(round(mean(f_k), digits=3)), " *
            "std=$(round(std(f_k), digits=3)), " *
            "min=$(round(minimum(f_k), digits=2)), " *
            "max=$(round(maximum(f_k), digits=2))")
end
println()

println("=" ^ 70)
println("Estimation complete!")
println("=" ^ 70)
