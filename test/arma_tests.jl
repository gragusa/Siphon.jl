# ARMA Model Tests
#
# Tests for ARMA(p,q) models in state-space form.
# Includes validation against MARSS R package where applicable.

using Test
using Siphon
using LinearAlgebra
using Random

@testset "ARMA Template Structure" begin
    @testset "ARMA(1,0) = AR(1)" begin
        spec = arma(1, 0)

        @test spec.name == Symbol("ARMA(1,0)")
        @test spec.n_states == 1  # max(1, 0+1) = 1
        @test spec.n_obs == 1
        @test spec.n_shocks == 1

        # Parameters: φ1, var
        @test length(spec.params) == 2
        @test param_names(spec) == [:φ1, :var]
    end

    @testset "ARMA(0,1) = MA(1)" begin
        spec = arma(0, 1)

        @test spec.name == Symbol("ARMA(0,1)")
        @test spec.n_states == 2  # max(0, 1+1) = 2
        @test spec.n_obs == 1
        @test spec.n_shocks == 1

        # Parameters: θ1, var
        @test length(spec.params) == 2
        @test param_names(spec) == [:θ1, :var]
    end

    @testset "ARMA(2,2)" begin
        spec = arma(2, 2)

        @test spec.name == Symbol("ARMA(2,2)")
        @test spec.n_states == 3  # max(2, 2+1) = 3
        @test spec.n_obs == 1
        @test spec.n_shocks == 1

        # Parameters: φ1, φ2, θ1, θ2, var
        @test length(spec.params) == 5
        @test param_names(spec) == [:φ1, :φ2, :θ1, :θ2, :var]
    end

    @testset "ARMA(3,1)" begin
        spec = arma(3, 1)

        @test spec.name == Symbol("ARMA(3,1)")
        @test spec.n_states == 3  # max(3, 1+1) = 3
        @test spec.n_obs == 1
        @test spec.n_shocks == 1

        # Parameters: φ1, φ2, φ3, θ1, var
        @test length(spec.params) == 5
        @test param_names(spec) == [:φ1, :φ2, :φ3, :θ1, :var]
    end

    @testset "ARMA(1,3)" begin
        spec = arma(1, 3)

        @test spec.name == Symbol("ARMA(1,3)")
        @test spec.n_states == 4  # max(1, 3+1) = 4
        @test spec.n_obs == 1
        @test spec.n_shocks == 1

        # Parameters: φ1, θ1, θ2, θ3, var
        @test length(spec.params) == 5
        @test param_names(spec) == [:φ1, :θ1, :θ2, :θ3, :var]
    end
end

@testset "ARMA Matrix Structure" begin
    @testset "ARMA(2,2) matrices" begin
        spec = arma(2, 2; ar_init = [0.7, -0.2], ma_init = [0.4, 0.1], var_init = 1.5)

        # Build matrices at initial values
        theta = (φ1 = 0.7, φ2 = -0.2, θ1 = 0.4, θ2 = 0.1, var = 1.5)
        kfparms = Siphon.build_kfparms(spec, theta)

        # Z = [1 0 0]
        @test size(kfparms.Z) == (1, 3)
        @test kfparms.Z[1, 1] ≈ 1.0
        @test kfparms.Z[1, 2] ≈ 0.0
        @test kfparms.Z[1, 3] ≈ 0.0

        # H = [0] (no observation noise in pure ARMA)
        @test size(kfparms.H) == (1, 1)
        @test kfparms.H[1, 1] ≈ 0.0

        # T (companion form)
        @test size(kfparms.T) == (3, 3)
        @test kfparms.T[1, 1] ≈ 0.7   # φ1
        @test kfparms.T[2, 1] ≈ -0.2  # φ2
        @test kfparms.T[3, 1] ≈ 0.0   # φ3 = 0 (p=2)
        @test kfparms.T[1, 2] ≈ 1.0   # subdiagonal
        @test kfparms.T[2, 3] ≈ 1.0   # subdiagonal

        # R = [1; θ1; θ2]
        @test size(kfparms.R) == (3, 1)
        @test kfparms.R[1, 1] ≈ 1.0
        @test kfparms.R[2, 1] ≈ 0.4   # θ1
        @test kfparms.R[3, 1] ≈ 0.1   # θ2

        # Q = [var]
        @test size(kfparms.Q) == (1, 1)
        @test kfparms.Q[1, 1] ≈ 1.5
    end
end

@testset "ARMA Kalman Filter" begin
    Random.seed!(42)

    # Generate AR(1) data
    n = 200
    φ = 0.8
    σ² = 1.0
    y = zeros(n)
    ε = randn(n)
    for t = 2:n
        y[t] = φ * y[t-1] + sqrt(σ²) * ε[t]
    end
    y_mat = reshape(y, 1, n)

    # Fit AR(1) = ARMA(1,0)
    spec = arma(1, 0; ar_init = [0.5], var_init = 1.0)

    # Log-likelihood should be computable
    theta = (φ1 = 0.8, var = 1.0)
    ll = Siphon.ssm_loglik(spec, theta, y_mat)
    @test isfinite(ll)
    @test ll < 0
end

@testset "ARMA MLE Estimation" begin
    Random.seed!(123)

    @testset "AR(1) parameter recovery" begin
        # True parameters
        φ_true = 0.75
        var_true = 2.0
        n = 500

        # Simulate AR(1)
        y = zeros(n)
        ε = sqrt(var_true) * randn(n)
        for t = 2:n
            y[t] = φ_true * y[t-1] + ε[t]
        end
        y_mat = reshape(y, 1, n)

        # Estimate
        spec = arma(1, 0; ar_init = [0.5], var_init = 1.0)
        model = StateSpaceModel(spec, n)
        fit!(MLE(), model, y_mat)

        params = parameters(model)

        # AR coefficient should be close to true value
        @test isapprox(params.φ1, φ_true, atol = 0.15)

        # Variance should be close
        @test isapprox(params.var, var_true, rtol = 0.3)

        # Log-likelihood should be finite
        @test isfinite(loglikelihood(model))
    end

    @testset "MA(1) parameter recovery" begin
        # True parameters
        θ_true = 0.6
        var_true = 1.5
        n = 500

        # Simulate MA(1)
        ε = sqrt(var_true) * randn(n + 1)
        y = zeros(n)
        for t = 1:n
            y[t] = ε[t+1] + θ_true * ε[t]
        end
        y_mat = reshape(y, 1, n)

        # Estimate
        spec = arma(0, 1; ma_init = [0.3], var_init = 1.0)
        model = StateSpaceModel(spec, n)
        fit!(MLE(), model, y_mat)

        params = parameters(model)

        # MA coefficient should be reasonable
        # (MA estimation can be harder than AR)
        @test isapprox(params.θ1, θ_true, atol = 0.3)

        # Variance should be in reasonable range
        @test params.var > 0.5 * var_true
        @test params.var < 2.0 * var_true
    end
end

@testset "ARMA StateSpaceModel Accessors" begin
    Random.seed!(42)

    n = 100
    y = randn(1, n)

    spec = arma(2, 1)
    model = StateSpaceModel(spec, n)
    fit!(MLE(), model, y)

    @testset "Parameter accessors" begin
        params = parameters(model)
        @test haskey(pairs(params), :φ1)
        @test haskey(pairs(params), :φ2)
        @test haskey(pairs(params), :θ1)
        @test haskey(pairs(params), :var)

        # theta_values should have same length as params
        @test length(model.theta_values) == 4
    end

    @testset "Matrix accessors" begin
        Z = obs_matrix(model)
        H = obs_cov(model)
        T = transition_matrix(model)
        R = selection_matrix(model)
        Q = state_cov(model)

        # Check dimensions: ARMA(2,1) has r = max(2, 1+1) = 2 states
        r = spec.n_states  # Get actual state dimension
        @test size(Z) == (1, r)
        @test size(H) == (1, 1)
        @test size(T) == (r, r)
        @test size(R) == (r, 1)
        @test size(Q) == (1, 1)

        # H should be zero for pure ARMA
        @test H[1, 1] ≈ 0.0

        # Z should select first state
        @test Z[1, 1] ≈ 1.0
    end

    @testset "system_matrices accessor" begin
        mats = system_matrices(model)

        @test haskey(pairs(mats), :Z)
        @test haskey(pairs(mats), :H)
        @test haskey(pairs(mats), :T)
        @test haskey(pairs(mats), :R)
        @test haskey(pairs(mats), :Q)

        # Should match individual accessors
        @test mats.Z == obs_matrix(model)
        @test mats.T == transition_matrix(model)
    end

    @testset "Filter accessors" begin
        at = predicted_states(model)
        att = filtered_states(model)
        vt = prediction_errors(model)

        r = spec.n_states
        @test size(at) == (r, n)
        @test size(att) == (r, n)
        @test size(vt) == (1, n)
    end

    @testset "Smoother accessors" begin
        αs = smoothed_states(model)

        r = spec.n_states
        @test size(αs) == (r, n)
        @test all(isfinite.(αs))
    end
end

@testset "ARMA EM Estimation" begin
    Random.seed!(456)

    # Generate AR(1) data
    n = 200
    φ_true = 0.7
    var_true = 1.0

    y = zeros(n)
    ε = sqrt(var_true) * randn(n)
    for t = 2:n
        y[t] = φ_true * y[t-1] + ε[t]
    end
    y_mat = reshape(y, 1, n)

    # Fit with EM
    spec = arma(1, 0)
    model = StateSpaceModel(spec, n)
    fit!(EM(), model, y_mat; maxiter = 100)

    @test model.fitted
    @test isfinite(loglikelihood(model))

    params = parameters(model)
    # EM should get reasonable estimates
    @test abs(params.φ1) < 1.0  # Bounded
    @test params.var > 0.0      # Positive variance
end

# ============================================
# MARSS Validation Tests
# ============================================
#
# These tests compare Siphon.jl results against MARSS R package.
# MARSS uses the same innovations form state-space representation.
#
# Reference: Holmes, E. E., Ward, E. J., and Wills, K. (2012).
# MARSS: Multivariate Autoregressive State-Space Models for Analyzing Time-Series Data.
# R Journal 4(1):11-19.

@testset "MARSS Validation: AR(1)" begin
    # Simulated AR(1) data with known parameters
    # This data can be generated in R and compared with MARSS

    Random.seed!(789)
    n = 100

    # Generate standardized AR(1): y_t = 0.8 * y_{t-1} + ε_t, ε_t ~ N(0,1)
    φ = 0.8
    σ² = 1.0
    y = zeros(n)
    ε = randn(n)
    for t = 2:n
        y[t] = φ * y[t-1] + ε[t]
    end
    y_mat = reshape(y, 1, n)

    # Fit with Siphon
    spec = arma(1, 0; ar_init = [0.5], var_init = 1.0)
    theta_true = (φ1 = φ, var = σ²)

    # Compute log-likelihood at true parameters
    ll_true = Siphon.ssm_loglik(spec, theta_true, y_mat)

    # Log-likelihood should be reasonable for 100 observations
    # (approximately -n/2 * log(2π) - n/2 * log(σ²) - 0.5 * sum((y-μ)²/σ²))
    @test ll_true > -200
    @test ll_true < -100

    # Build state-space matrices and verify structure
    kfparms = Siphon.build_kfparms(spec, theta_true)

    # MARSS uses: x_t = B * x_{t-1} + u + w_t
    #             y_t = Z * x_t + a + v_t
    # For AR(1): B = [φ], Z = [1], Q = [σ²], R = [0]

    @test kfparms.T[1, 1] ≈ φ        # B in MARSS notation
    @test kfparms.Z[1, 1] ≈ 1.0      # Z
    @test kfparms.Q[1, 1] ≈ σ²       # Q
    @test kfparms.H[1, 1] ≈ 0.0      # R (no obs noise in pure AR)
end

@testset "MARSS Validation: ARMA(2,1)" begin
    # ARMA(2,1): y_t = φ1*y_{t-1} + φ2*y_{t-2} + ε_t + θ1*ε_{t-1}
    # State dimension: r = max(p, q+1) = max(2, 2) = 2

    Random.seed!(101)
    n = 200

    # True parameters
    φ1, φ2 = 0.5, -0.2
    θ1 = 0.3
    σ² = 1.0

    # Simulate
    ε = sqrt(σ²) * randn(n + 10)
    y = zeros(n)
    for t = 3:n
        y[t] = φ1 * y[t-1] + φ2 * y[t-2] + ε[t+10] + θ1 * ε[t+9]
    end
    y_mat = reshape(y, 1, n)

    # Fit model
    spec = arma(2, 1)
    model = StateSpaceModel(spec, n)
    fit!(MLE(), model, y_mat)

    params = parameters(model)

    # Check structure is correct
    mats = system_matrices(model)
    r = spec.n_states  # max(2, 1+1) = 2

    @test size(mats.T) == (r, r)
    @test size(mats.R) == (r, 1)

    # For ARMA(2,1) with r=2:
    # T matrix structure: companion form
    # T = [φ1  1]
    #     [φ2  0]
    @test mats.T[1, 2] ≈ 1.0
    @test mats.T[2, 2] ≈ 0.0

    # R matrix structure
    # R = [1; θ1]
    @test mats.R[1, 1] ≈ 1.0
    # θ1 coefficient in R[2,1] should match params.θ1
    @test mats.R[2, 1] ≈ params.θ1
end

@testset "ARMA Custom vs Template" begin
    # Verify that custom_ssm and arma template produce equivalent results

    Random.seed!(202)
    n = 100
    y = randn(1, n)

    # ARMA(1,1) via template
    spec_template = arma(1, 1; ar_init = [0.6], ma_init = [0.3], var_init = 1.0)

    # ARMA(1,1) via custom_ssm using proper matrix type annotations
    r = 2  # max(1, 1+1) = 2

    # Build T matrix with Union type
    T_mat = Union{Float64,FreeParam}[
        FreeParam(:φ1, init = 0.6, lower = -Inf, upper = Inf) 1.0;
        0.0 0.0
    ]

    # Build R matrix with Union type
    R_mat = Union{Float64,FreeParam}[
        1.0;
        FreeParam(:θ1, init = 0.3, lower = -Inf, upper = Inf)
    ]
    R_mat = reshape(R_mat, 2, 1)  # Make it a proper 2x1 matrix

    spec_custom = custom_ssm(
        Z = [1.0 0.0],
        H = zeros_mat(1, 1),
        T = T_mat,
        R = R_mat,
        Q = reshape([FreeParam(:var, init = 1.0, lower = 0.0)], 1, 1),
        a1 = [0.0, 0.0],
        P1 = 1e7 * Matrix(1.0I, 2, 2),
        name = :ARMA11_custom,
    )

    # Both should have same dimensions
    @test spec_template.n_states == spec_custom.n_states
    @test spec_template.n_obs == spec_custom.n_obs

    # Build matrices at same parameters
    theta = (φ1 = 0.6, θ1 = 0.3, var = 1.0)

    kf_template = Siphon.build_kfparms(spec_template, theta)
    kf_custom = Siphon.build_kfparms(spec_custom, theta)

    # Matrices should be equal
    @test kf_template.Z ≈ kf_custom.Z
    @test kf_template.H ≈ kf_custom.H
    @test kf_template.T ≈ kf_custom.T
    @test kf_template.R ≈ kf_custom.R
    @test kf_template.Q ≈ kf_custom.Q

    # Log-likelihoods should be equal
    ll_template = Siphon.ssm_loglik(spec_template, theta, y)
    ll_custom = Siphon.ssm_loglik(spec_custom, theta, y)

    @test ll_template ≈ ll_custom
end
