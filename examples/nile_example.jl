"""
Nile River Example: Local Level Model with DSL

This example demonstrates how to:
1. Specify a local level model using the DSL
2. Build the likelihood function
3. Evaluate and optimize the likelihood
"""

using Siphon
using CSV
using DataFrames

# Load the Nile data
nile = CSV.read(joinpath(@__DIR__, "..", "test", "Nile.csv"), DataFrame; header = false)
y = permutedims(Matrix(nile))  # Convert to 1×n matrix for the filter

println("Nile data: $(length(nile)) observations")
println("Range: $(minimum(nile)) - $(maximum(nile))")

# ============================================
# Method 1: Using the local_level template
# ============================================

println("\n=== Method 1: Using local_level() template ===")

spec = local_level()
println("Parameters: ", param_names(spec))
println("Initial values: ", initial_values(spec))

# Get negative log-likelihood function
negloglik = objective_function(spec, y)

# Evaluate at initial values
θ0 = initial_values(spec)
nll0 = negloglik(θ0)
println("Neg log-likelihood at init: ", nll0)

# ============================================
# Method 2: Using custom_ssm for more control
# ============================================

println("\n=== Method 2: Using custom_ssm() ===")

spec2 = custom_ssm(
    Z = [1.0],
    H = [FreeParam(:var_obs, init = 10000.0, lower = 0.0)],
    T = [1.0],
    R = [1.0],
    Q = [FreeParam(:var_level, init = 2500.0, lower = 0.0)],
    a1 = [0.0],
    P1 = [1e7]
)

println("Parameters: ", param_names(spec2))
println("Initial values: ", initial_values(spec2))

negloglik2 = objective_function(spec2, y)
nll2 = negloglik2(initial_values(spec2))
println("Neg log-likelihood at init: ", nll2)

# ============================================
# Method 3: Using SSMLogDensity (unconstrained)
# ============================================

println("\n=== Method 3: Using SSMLogDensity (unconstrained space) ===")

ld = SSMLogDensity(spec2, y)

# Transform to unconstrained space
θ_init = initial_values(spec2)
θ_u = transform_to_unconstrained(spec2, θ_init)

println("Initial (constrained): ", θ_init)
println("Initial (unconstrained): ", θ_u)

ll = logdensity(ld, θ_u)
println("Log-density at init: ", ll)

# ============================================
# Simple grid search optimization
# ============================================

println("\n=== Simple Grid Search ===")

function grid_search(negloglik)
    best_nll = Inf
    best_θ = [1.0, 1.0]

    # Grid search over variance values directly
    for var_obs in 10000:1000:20000
        for var_level in 500:100:3000
            θ = [Float64(var_obs), Float64(var_level)]
            nll = negloglik(θ)
            if nll < best_nll
                best_nll = nll
                best_θ = θ
            end
        end
    end
    return best_θ, best_nll
end

best_θ, best_nll = grid_search(negloglik)

println("Best parameters from grid search:")
println("  var_obs   = ", best_θ[1])
println("  var_level = ", best_θ[2])
println("  Neg log-likelihood = ", best_nll)
println("  Log-likelihood = ", -best_nll)

# Compare with known MLE values from Durbin & Koopman
println("\n=== Reference values (Durbin & Koopman) ===")
println("  var_obs ≈ 15099")
println("  var_level ≈ 1469")

# Build and filter with estimated parameters
ss = build_linear_state_space(spec, best_θ, y)
ll = kalman_loglik(ss.p, y, ss.a1, ss.P1)
println("\nFiltered log-likelihood: ", ll)
