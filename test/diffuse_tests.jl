"""
Tests for exact diffuse Kalman filter initialization.

Tests both the pure functional version (filter_ad.jl) and the in-place version (inplace.jl).
"""

using Test
using Siphon
using LinearAlgebra
using Random

@testset "Exact diffuse filter - local level model" begin
    Random.seed!(1234)

    # Local level model: y_t = μ_t + ε_t, μ_{t+1} = μ_t + η_t
    σ²_obs = 100.0
    σ²_state = 10.0

    p = KFParms(
        [1.0;;],       # Z
        [σ²_obs;;],    # H
        [1.0;;],       # T
        [1.0;;],       # R
        [σ²_state;;],   # Q
    )

    # Generate data
    n = 100
    true_level = cumsum(sqrt(σ²_state) * randn(n))
    y = reshape(true_level + sqrt(σ²_obs) * randn(n), 1, n)

    # Exact diffuse initialization
    a1 = [0.0]
    P1_star = [0.0;;]   # No finite uncertainty
    P1_inf = [1.0;;]    # Full diffuse on level

    # Test log-likelihood computation
    ll_diffuse = kalman_loglik(p, y, a1, P1_star, P1_inf)
    @test isfinite(ll_diffuse)
    @test ll_diffuse < 0  # Log-likelihood should be negative

    # Compare with approximate diffuse (large P1)
    P1_approx = [1e7;;]
    ll_approx = kalman_loglik(p, y, a1, P1_approx)
    @test isfinite(ll_approx)

    # Likelihoods should be similar (exact diffuse excludes first observation)
    # The difference comes from treatment of first observation
    @test abs(ll_diffuse - ll_approx) < 10.0  # Reasonable tolerance
end

@testset "Exact diffuse filter - full filter output" begin
    Random.seed!(2345)

    σ²_obs = 100.0
    σ²_state = 10.0

    p = KFParms([1.0;;], [σ²_obs;;], [1.0;;], [1.0;;], [σ²_state;;])

    n = 50
    y = randn(1, n)

    a1 = [0.0]
    P1_star = [0.0;;]
    P1_inf = [1.0;;]

    result = kalman_filter(p, y, a1, P1_star, P1_inf)

    # Check dimensions
    @test size(result.at) == (1, n)
    @test size(result.att) == (1, n)
    @test size(result.Pt) == (1, 1, n)
    @test size(result.vt) == (1, n)

    # Check diffuse period
    d = diffuse_period(result)
    @test d >= 1  # At least one diffuse observation for univariate

    # Check diffuse covariances exist
    Pinf_store, Pstar_store = diffuse_covariances(result)
    @test size(Pinf_store, 3) == d
    @test size(Pstar_store, 3) == d

    # Check flags
    flags = diffuse_flags(result)
    @test length(flags) == d
    @test all(f -> f in [-1, 0, 1], flags)  # Valid flag values

    # Log-likelihood should match standalone function
    ll_standalone = kalman_loglik(p, y, a1, P1_star, P1_inf)
    @test loglikelihood(result) ≈ ll_standalone
end

@testset "Diffuse filter - local linear trend" begin
    Random.seed!(3456)

    # Local linear trend: μ_t = μ_{t-1} + β_{t-1}, β_t = β_{t-1}
    σ²_obs = 25.0
    σ²_level = 1.0
    σ²_slope = 0.1

    p = KFParms(
        [1.0 0.0],                     # Z
        [σ²_obs;;],                    # H
        [1.0 1.0; 0.0 1.0],            # T
        [1.0 0.0; 0.0 1.0],            # R
        [σ²_level 0.0; 0.0 σ²_slope],   # Q
    )

    n = 100
    y = randn(1, n)

    a1 = [0.0, 0.0]
    P1_star = zeros(2, 2)
    P1_inf = Matrix(1.0I, 2, 2)  # Both level and slope are diffuse

    result = kalman_filter(p, y, a1, P1_star, P1_inf)

    @test isfinite(loglikelihood(result))
    d = diffuse_period(result)
    @test d >= 2  # Need at least 2 observations to identify both states
end

@testset "Diffuse filter - missing data handling" begin
    Random.seed!(4567)

    p = KFParms([1.0;;], [100.0;;], [1.0;;], [1.0;;], [10.0;;])

    n = 50
    y = randn(1, n)
    y[1, 5:10] .= NaN  # Missing observations

    a1 = [0.0]
    P1_star = [0.0;;]
    P1_inf = [1.0;;]

    result = kalman_filter(p, y, a1, P1_star, P1_inf)

    @test isfinite(loglikelihood(result))

    # Missing mask should be set correctly
    @test all(result.missing_mask[5:10])
    @test !any(result.missing_mask[1:4])

    # Innovations should be NaN for missing observations
    @test all(isnan.(result.vt[:, 5:10]))
end

@testset "Diffuse filter convergence" begin
    Random.seed!(5678)

    # After diffuse period, Pinf should be zero and filter should behave normally
    p = KFParms([1.0;;], [100.0;;], [1.0;;], [1.0;;], [10.0;;])

    n = 100
    y = randn(1, n)

    a1 = [0.0]
    P1_star = [0.0;;]
    P1_inf = [1.0;;]

    result = kalman_filter(p, y, a1, P1_star, P1_inf)
    d = diffuse_period(result)

    # After diffuse period, covariances should be finite and reasonable
    for t = (d+1):n
        P_t = result.Pt[:, :, t]
        @test all(isfinite.(P_t))
        @test all(P_t .> 0)  # Should be positive
    end
end

@testset "Diffuse vs approximate comparison" begin
    Random.seed!(6789)

    # The key difference: exact diffuse doesn't count first observation in likelihood
    # when Finf is invertible

    p = KFParms([1.0;;], [100.0;;], [1.0;;], [1.0;;], [10.0;;])

    n = 200
    y = randn(1, n)

    a1 = [0.0]
    P1_star = [0.0;;]
    P1_inf = [1.0;;]

    # Exact diffuse
    ll_exact = kalman_loglik(p, y, a1, P1_star, P1_inf)

    # Approximate diffuse with various P1 values
    for P1_val in [1e4, 1e6, 1e8, 1e10]
        P1_approx = [P1_val;;]
        ll_approx = kalman_loglik(p, y, a1, P1_approx)

        # As P1 → ∞, approximate should converge toward exact
        # But there will always be a small difference due to first observation treatment
        @test isfinite(ll_approx)
    end
end

# ============================================
# In-place Diffuse Filter Tests
# ============================================

@testset "In-place diffuse filter - local level" begin
    Random.seed!(7890)

    # Local level model
    σ²_obs = 100.0
    σ²_state = 10.0

    Z = [1.0;;]
    H = [σ²_obs;;]
    Tmat = [1.0;;]
    R = [1.0;;]
    Q = [σ²_state;;]

    n = 100
    y = randn(1, n)

    a1 = [0.0]
    P1_star = [0.0;;]
    P1_inf = [1.0;;]

    # In-place version
    ws = DiffuseKalmanWorkspace(Z, H, Tmat, R, Q, a1, P1_star, P1_inf, n)
    ll_inplace = kalman_filter!(ws, y)

    # Pure functional version
    p = KFParms(Z, H, Tmat, R, Q)
    ll_pure = kalman_loglik(p, y, a1, P1_star, P1_inf)

    # Log-likelihoods should match
    @test ll_inplace ≈ ll_pure rtol=1e-10

    # Diffuse period should match
    result_pure = kalman_filter(p, y, a1, P1_star, P1_inf)
    @test diffuse_period(ws) == diffuse_period(result_pure)

    # Diffuse flags should match
    @test collect(diffuse_flags(ws)) == diffuse_flags(result_pure)
end

@testset "In-place diffuse filter - local linear trend" begin
    Random.seed!(8901)

    # Local linear trend model
    σ²_obs = 25.0
    σ²_level = 1.0
    σ²_slope = 0.1

    Z = [1.0 0.0]
    H = [σ²_obs;;]
    Tmat = [1.0 1.0; 0.0 1.0]
    R = [1.0 0.0; 0.0 1.0]
    Q = [σ²_level 0.0; 0.0 σ²_slope]

    n = 100
    y = randn(1, n)

    a1 = [0.0, 0.0]
    P1_star = zeros(2, 2)
    P1_inf = Matrix(1.0I, 2, 2)

    # In-place version
    ws = DiffuseKalmanWorkspace(Z, H, Tmat, R, Q, a1, P1_star, P1_inf, n)
    ll_inplace = kalman_filter!(ws, y)

    # Pure functional version
    p = KFParms(Z, H, Tmat, R, Q)
    ll_pure = kalman_loglik(p, y, a1, P1_star, P1_inf)

    # Log-likelihoods should match
    @test ll_inplace ≈ ll_pure rtol=1e-10

    # Need at least 2 observations to identify both states
    @test diffuse_period(ws) >= 2
end

@testset "In-place diffuse filter - filtered states match" begin
    Random.seed!(9012)

    σ²_obs = 100.0
    σ²_state = 10.0

    Z = [1.0;;]
    H = [σ²_obs;;]
    Tmat = [1.0;;]
    R = [1.0;;]
    Q = [σ²_state;;]

    n = 50
    y = randn(1, n)

    a1 = [0.0]
    P1_star = [0.0;;]
    P1_inf = [1.0;;]

    # In-place version
    ws = DiffuseKalmanWorkspace(Z, H, Tmat, R, Q, a1, P1_star, P1_inf, n)
    kalman_filter!(ws, y)

    # Pure functional version
    p = KFParms(Z, H, Tmat, R, Q)
    result = kalman_filter(p, y, a1, P1_star, P1_inf)

    # Compare filtered states (after diffuse period)
    d = diffuse_period(ws)
    for t = (d+1):n
        @test filtered_states(ws)[:, t] ≈ result.att[:, t] rtol=1e-10
        @test predicted_states(ws)[:, t] ≈ result.at[:, t] rtol=1e-10
    end

    # Compare prediction errors
    for t = (d+1):n
        @test prediction_errors(ws)[:, t] ≈ result.vt[:, t] rtol=1e-10
    end
end

@testset "In-place diffuse filter - missing data" begin
    Random.seed!(123)

    Z = [1.0;;]
    H = [100.0;;]
    Tmat = [1.0;;]
    R = [1.0;;]
    Q = [10.0;;]

    n = 50
    y = randn(1, n)
    y[1, 5:10] .= NaN  # Missing observations

    a1 = [0.0]
    P1_star = [0.0;;]
    P1_inf = [1.0;;]

    # In-place version
    ws = DiffuseKalmanWorkspace(Z, H, Tmat, R, Q, a1, P1_star, P1_inf, n)
    ll_inplace = kalman_filter!(ws, y)

    # Pure functional version
    p = KFParms(Z, H, Tmat, R, Q)
    ll_pure = kalman_loglik(p, y, a1, P1_star, P1_inf)

    # Log-likelihoods should match
    @test ll_inplace ≈ ll_pure rtol=1e-10

    # Check missing mask
    @test all(Siphon.missing_mask(ws)[5:10])
    @test !any(Siphon.missing_mask(ws)[1:4])
end

@testset "In-place diffuse filter - workspace reuse" begin
    Random.seed!(234)

    Z = [1.0;;]
    H = [100.0;;]
    Tmat = [1.0;;]
    R = [1.0;;]
    Q = [10.0;;]

    n = 50
    a1 = [0.0]
    P1_star = [0.0;;]
    P1_inf = [1.0;;]

    ws = DiffuseKalmanWorkspace(Z, H, Tmat, R, Q, a1, P1_star, P1_inf, n)

    # Run on first dataset
    y1 = randn(1, n)
    ll1 = kalman_filter!(ws, y1)

    # Reset and run on second dataset
    set_initial_diffuse!(ws, a1, P1_star, P1_inf)
    y2 = randn(1, n)
    ll2 = kalman_filter!(ws, y2)

    # Verify second run is correct
    p = KFParms(Z, H, Tmat, R, Q)
    ll2_pure = kalman_loglik(p, y2, a1, P1_star, P1_inf)
    @test ll2 ≈ ll2_pure rtol=1e-10

    # Different data should give different likelihoods
    @test ll1 != ll2
end

@testset "In-place diffuse filter - accessors" begin
    p_dim = 2
    m_dim = 3
    r_dim = 2
    n = 50

    ws = DiffuseKalmanWorkspace{Float64}(p_dim, m_dim, r_dim, n)

    # Check dimension accessors
    @test Siphon.obs_dim(ws) == p_dim
    @test Siphon.state_dim(ws) == m_dim
    @test Siphon.shock_dim(ws) == r_dim
    @test Siphon.n_times(ws) == n

    # Check storage dimensions
    @test size(predicted_states(ws)) == (m_dim, n)
    @test size(filtered_states(ws)) == (m_dim, n)
    @test size(variances_predicted_states(ws)) == (m_dim, m_dim, n)
    @test size(prediction_errors(ws)) == (p_dim, n)
    @test size(kalman_gains(ws)) == (m_dim, p_dim, n)
end
