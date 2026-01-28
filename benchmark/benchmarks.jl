"""
Benchmarks for Siphon.jl

Benchmark suite measuring:
1. Kalman filter operations (scalar and matrix models)
2. Kalman smoother operations
3. Log-likelihood computation
4. MLE estimation
5. EM estimation
6. DSL model construction
"""

using BenchmarkTools
using LinearAlgebra
using Random
using StableRNGs

using Siphon
using Siphon: local_level, local_linear_trend, ar1, dynamic_factor
using Siphon: StateSpaceModel, kalman_filter!, kalman_smoother!, kalman_loglik
using Siphon: MLE, EM, fit!

# ============================================================================
# Data Generation
# ============================================================================

const DEFAULT_SEED = 20240612

"""
Generate local level model data.
"""
function generate_local_level_data(rng::AbstractRNG, n::Int; σ_obs = 15.0, σ_level = 10.0)
    μ = zeros(n + 1)
    y = zeros(1, n)
    μ[1] = randn(rng) * 100
    for t in 1:n
        y[1, t] = μ[t] + σ_obs * randn(rng)
        μ[t + 1] = μ[t] + σ_level * randn(rng)
    end
    return y
end

"""
Generate local linear trend model data.
"""
function generate_trend_data(rng::AbstractRNG, n::Int)
    level = zeros(n + 1)
    slope = zeros(n + 1)
    y = zeros(1, n)

    level[1] = randn(rng) * 10
    slope[1] = randn(rng) * 0.5

    σ_obs = 5.0
    σ_level = 2.0
    σ_slope = 0.5

    for t in 1:n
        y[1, t] = level[t] + σ_obs * randn(rng)
        level[t + 1] = level[t] + slope[t] + σ_level * randn(rng)
        slope[t + 1] = slope[t] + σ_slope * randn(rng)
    end
    return y
end

"""
Generate multivariate factor model data.
"""
function generate_factor_data(rng::AbstractRNG, n::Int, n_obs::Int, n_factors::Int)
    # Generate factors
    F = zeros(n_factors, n + 1)
    F[:, 1] = randn(rng, n_factors)

    for t in 1:n
        F[:, t + 1] = 0.8 * F[:, t] + 0.3 * randn(rng, n_factors)
    end

    # Generate loadings and observations
    Lambda = randn(rng, n_obs, n_factors) ./ sqrt(n_factors)
    y = zeros(n_obs, n)

    for t in 1:n
        y[:, t] = Lambda * F[:, t] + 0.5 * randn(rng, n_obs)
    end

    return y
end

# ============================================================================
# Benchmark Suite
# ============================================================================

const SUITE = BenchmarkGroup()

# ----------------------------------------------------------------------------
# DSL Model Construction Benchmarks
# ----------------------------------------------------------------------------

SUITE["dsl"] = BenchmarkGroup()

SUITE["dsl"]["local_level"] = @benchmarkable local_level(var_obs_init = 225.0)
SUITE["dsl"]["local_linear_trend"] = @benchmarkable local_linear_trend()
SUITE["dsl"]["ar1"] = @benchmarkable ar1(rho_init = 0.9)
SUITE["dsl"]["dynamic_factor_10obs_3factors"] = @benchmarkable dynamic_factor(10, 3; factor_lags = 2)

# ----------------------------------------------------------------------------
# Log-likelihood Computation Benchmarks
# ----------------------------------------------------------------------------

SUITE["loglik"] = BenchmarkGroup()

# Local level model
let rng = StableRNG(DEFAULT_SEED)
    y = generate_local_level_data(rng, 500)
    spec = local_level(var_obs_init = 225.0, var_level_init = 100.0)
    θ = (var_obs = 225.0, var_level = 100.0)
    model = StateSpaceModel(spec, θ, size(y, 2))

    SUITE["loglik"]["local_level_n500"] = @benchmarkable kalman_loglik($model, $y)
end

let rng = StableRNG(DEFAULT_SEED + 1)
    y = generate_local_level_data(rng, 2000)
    spec = local_level(var_obs_init = 225.0, var_level_init = 100.0)
    θ = (var_obs = 225.0, var_level = 100.0)
    model = StateSpaceModel(spec, θ, size(y, 2))

    SUITE["loglik"]["local_level_n2000"] = @benchmarkable kalman_loglik($model, $y)
end

# Local linear trend model
let rng = StableRNG(DEFAULT_SEED + 2)
    y = generate_trend_data(rng, 500)
    spec = local_linear_trend()
    θ = (var_obs = 25.0, var_level = 4.0, var_slope = 0.25)
    model = StateSpaceModel(spec, θ, size(y, 2))

    SUITE["loglik"]["trend_n500"] = @benchmarkable kalman_loglik($model, $y)
end

# ----------------------------------------------------------------------------
# Filter + Smoother Benchmarks
# ----------------------------------------------------------------------------

SUITE["filter"] = BenchmarkGroup()

let rng = StableRNG(DEFAULT_SEED + 10)
    y = generate_local_level_data(rng, 1000)
    spec = local_level(var_obs_init = 225.0, var_level_init = 100.0)
    θ = (var_obs = 225.0, var_level = 100.0)
    model = StateSpaceModel(spec, θ, size(y, 2))

    SUITE["filter"]["local_level_n1000"] = @benchmarkable kalman_filter!($model, $y)
end

let rng = StableRNG(DEFAULT_SEED + 11)
    y = generate_trend_data(rng, 1000)
    spec = local_linear_trend()
    θ = (var_obs = 25.0, var_level = 4.0, var_slope = 0.25)
    model = StateSpaceModel(spec, θ, size(y, 2))

    SUITE["filter"]["trend_n1000"] = @benchmarkable kalman_filter!($model, $y)
end

SUITE["smoother"] = BenchmarkGroup()

let rng = StableRNG(DEFAULT_SEED + 20)
    y = generate_local_level_data(rng, 1000)
    spec = local_level(var_obs_init = 225.0, var_level_init = 100.0)
    θ = (var_obs = 225.0, var_level = 100.0)
    model = StateSpaceModel(spec, θ, size(y, 2))
    kalman_filter!(model, y)

    SUITE["smoother"]["local_level_n1000"] = @benchmarkable kalman_smoother!($model)
end

let rng = StableRNG(DEFAULT_SEED + 21)
    y = generate_trend_data(rng, 1000)
    spec = local_linear_trend()
    θ = (var_obs = 25.0, var_level = 4.0, var_slope = 0.25)
    model = StateSpaceModel(spec, θ, size(y, 2))
    kalman_filter!(model, y)

    SUITE["smoother"]["trend_n1000"] = @benchmarkable kalman_smoother!($model)
end

# ----------------------------------------------------------------------------
# MLE Estimation Benchmarks
# ----------------------------------------------------------------------------

SUITE["mle"] = BenchmarkGroup()

let rng = StableRNG(DEFAULT_SEED + 30)
    y = generate_local_level_data(rng, 200)
    spec = local_level(var_obs_init = 100.0, var_level_init = 50.0)

    SUITE["mle"]["local_level_n200"] = @benchmarkable begin
        m = StateSpaceModel($spec, size($y, 2))
        fit!(MLE(), m, $y)
    end
end

let rng = StableRNG(DEFAULT_SEED + 31)
    y = generate_trend_data(rng, 200)
    spec = local_linear_trend()

    SUITE["mle"]["trend_n200"] = @benchmarkable begin
        m = StateSpaceModel($spec, size($y, 2))
        fit!(MLE(), m, $y)
    end
end

# ----------------------------------------------------------------------------
# EM Estimation Benchmarks
# ----------------------------------------------------------------------------

SUITE["em"] = BenchmarkGroup()

let rng = StableRNG(DEFAULT_SEED + 40)
    y = generate_local_level_data(rng, 200)
    spec = local_level(var_obs_init = 100.0, var_level_init = 50.0)

    SUITE["em"]["local_level_n200_iter50"] = @benchmarkable begin
        m = StateSpaceModel($spec, size($y, 2))
        fit!(EM(), m, $y; maxiter = 50, verbose = false)
    end
end

let rng = StableRNG(DEFAULT_SEED + 41)
    y = generate_trend_data(rng, 200)
    spec = local_linear_trend()

    SUITE["em"]["trend_n200_iter50"] = @benchmarkable begin
        m = StateSpaceModel($spec, size($y, 2))
        fit!(EM(), m, $y; maxiter = 50, verbose = false)
    end
end

# ----------------------------------------------------------------------------
# Dynamic Factor Model Benchmarks
# ----------------------------------------------------------------------------

SUITE["dfm"] = BenchmarkGroup()

let rng = StableRNG(DEFAULT_SEED + 50)
    y = generate_factor_data(rng, 200, 10, 2)
    spec = dynamic_factor(10, 2; factor_lags = 1)

    # Note: DFM estimation can be slow, so we just benchmark a few EM iterations
    SUITE["dfm"]["10obs_2factors_em20"] = @benchmarkable begin
        m = StateSpaceModel($spec, size($y, 2))
        fit!(EM(), m, $y; maxiter = 20, verbose = false)
    end
end

let rng = StableRNG(DEFAULT_SEED + 51)
    y = generate_factor_data(rng, 200, 20, 3)
    spec = dynamic_factor(20, 3; factor_lags = 1)

    SUITE["dfm"]["20obs_3factors_em20"] = @benchmarkable begin
        m = StateSpaceModel($spec, size($y, 2))
        fit!(EM(), m, $y; maxiter = 20, verbose = false)
    end
end
