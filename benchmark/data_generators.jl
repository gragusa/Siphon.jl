"""
    data_generators.jl

Synthetic data generation for Kalman filter benchmarks.
Generates state-space models and observations for various scenarios.
"""

using LinearAlgebra
using Random
using StaticArrays

"""
Generate a stable random transition matrix T.
Eigenvalues are scaled to have magnitude < 0.99.
"""
function random_stable_T(m::Int; rng = Random.GLOBAL_RNG)
    T = randn(rng, m, m)
    # Scale to ensure stability
    λmax = maximum(abs.(eigvals(T)))
    if λmax > 0.99
        T = T * (0.95 / λmax)
    end
    return T
end

"""
Generate a random positive definite matrix.
"""
function random_posdef(n::Int; scale = 1.0, rng = Random.GLOBAL_RNG)
    A = randn(rng, n, n)
    return scale * (A * A') + 0.01 * I
end

# ============================================
# Scenario 1: Scalar local level model
# ============================================

"""
Generate scalar local level model parameters.
    yₜ = μₜ + εₜ,  εₜ ~ N(0, σ²_ε)
    μₜ₊₁ = μₜ + ηₜ,  ηₜ ~ N(0, σ²_η)
"""
function scalar_local_level(; σ_ε = 15.0, σ_η = 10.0)
    Z = 1.0
    H = σ_ε^2
    T = 1.0
    R = 1.0
    Q = σ_η^2
    a1 = 0.0
    P1 = 1e4  # Large but not too diffuse for numerical stability
    return (; Z, H, T, R, Q, a1, P1)
end

"""
Simulate observations from scalar local level model.
"""
function simulate_scalar(n::Int; σ_ε = 15.0, σ_η = 10.0, rng = Random.GLOBAL_RNG)
    μ = zeros(n + 1)
    y = zeros(n)
    μ[1] = randn(rng) * 100  # Initial state
    for t = 1:n
        y[t] = μ[t] + σ_ε * randn(rng)
        μ[t+1] = μ[t] + σ_η * randn(rng)
    end
    return y
end

# ============================================
# Scenario 2: Small matrix (3 states, 1 obs)
# ============================================

"""
Generate small matrix model parameters (local linear trend + cycle).
State: [level, slope, cycle]
"""
function small_matrix(;
    σ_ε = 1.0,
    σ_level = 0.5,
    σ_slope = 0.1,
    σ_cycle = 0.3,
    ρ = 0.9,
    λ = 0.1,
)
    m = 3  # states
    p = 1  # observations

    Z = [1.0 0.0 1.0]  # 1×3
    H = fill(σ_ε^2, 1, 1)

    T = [
        1.0 1.0 0.0;
        0.0 1.0 0.0;
        0.0 0.0 ρ*cos(λ)
    ]

    R = Matrix(1.0I, m, m)
    Q = diagm([σ_level^2, σ_slope^2, σ_cycle^2])

    a1 = zeros(m)
    P1 = 1e4 * Matrix(1.0I, m, m)  # Large but stable

    return (; Z, H, T, R, Q, a1, P1)
end

"""
Simulate observations from small matrix model.
"""
function simulate_small_matrix(n::Int; rng = Random.GLOBAL_RNG)
    parms = small_matrix()
    m = 3

    α = zeros(m, n + 1)
    y = zeros(1, n)

    # Initial state from prior
    α[:, 1] = randn(rng, m) .* 10

    R_chol = cholesky(parms.Q).L
    H_chol = sqrt(parms.H[1, 1])

    for t = 1:n
        y[:, t] = parms.Z * α[:, t] .+ H_chol * randn(rng)
        α[:, t+1] = parms.T * α[:, t] + R_chol * randn(rng, m)
    end

    return y
end

# ============================================
# Scenario 3: Medium matrix (10 states, 3 obs)
# ============================================

"""
Generate medium matrix model parameters.
VAR(1)-like structure with 10 states, 3 observables.
"""
function medium_matrix(; rng = Random.GLOBAL_RNG)
    m = 10  # states
    p = 3   # observations
    r = 5   # shocks

    # Observation equation
    Z = randn(rng, p, m) ./ sqrt(m)
    H = random_posdef(p; scale = 0.1, rng = rng)

    # State equation
    T = random_stable_T(m; rng = rng)
    R = randn(rng, m, r)
    Q = random_posdef(r; scale = 0.5, rng = rng)

    a1 = zeros(m)
    P1 = 1e4 * Matrix(1.0I, m, m)  # Large but stable

    return (; Z, H, T, R, Q, a1, P1)
end

"""
Simulate observations from medium matrix model.
"""
function simulate_medium_matrix(n::Int; rng = Random.GLOBAL_RNG)
    parms = medium_matrix(; rng = rng)
    m, p, r = 10, 3, 5

    α = zeros(m, n + 1)
    y = zeros(p, n)

    α[:, 1] = randn(rng, m)

    R_chol = cholesky(parms.Q).L
    H_chol = cholesky(parms.H).L

    for t = 1:n
        y[:, t] = parms.Z * α[:, t] + H_chol * randn(rng, p)
        α[:, t+1] = parms.T * α[:, t] + parms.R * R_chol * randn(rng, r)
    end

    return y, parms
end

# ============================================
# Scenario 4: StaticArrays (small fixed-size)
# ============================================

"""
Generate StaticArrays version of small matrix model.
"""
function small_static()
    parms = small_matrix()

    Z = SMatrix{1,3}(parms.Z)
    H = SMatrix{1,1}(parms.H)
    T = SMatrix{3,3}(parms.T)
    R = SMatrix{3,3}(parms.R)
    Q = SMatrix{3,3}(parms.Q)
    a1 = SVector{3}(parms.a1)
    P1 = SMatrix{3,3}(parms.P1)

    return (; Z, H, T, R, Q, a1, P1)
end

# ============================================
# Utility functions
# ============================================

"""
Create a reproducible RNG for benchmarks.
"""
benchmark_rng(seed::Int = 42) = Random.MersenneTwister(seed)

"""
Pre-generate all benchmark data.
"""
function generate_all_benchmark_data(; seed = 42)
    rng = benchmark_rng(seed)

    data = Dict{Symbol,Any}()

    # Sequence lengths
    lengths = [100, 1000, 10000]

    for n in lengths
        data[Symbol("scalar_$n")] = simulate_scalar(n; rng = rng)
        data[Symbol("small_matrix_$n")] = simulate_small_matrix(n; rng = rng)

        y, parms = simulate_medium_matrix(n; rng = rng)
        data[Symbol("medium_matrix_$n")] = y
        data[Symbol("medium_parms_$n")] = parms
    end

    return data
end
