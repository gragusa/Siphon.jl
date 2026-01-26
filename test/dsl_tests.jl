"""
Tests for the DSL module: templates, custom_ssm, matrix helpers, MatrixExpr, and SSMLogDensity.
"""

using Test
using Siphon
using LinearAlgebra
using Statistics: cor

# ============================================
# SSMParameter and SSMSpec Tests
# ============================================

@testset "SSMParameter" begin
    # Basic construction
    p = SSMParameter(:σ; init = 1.0, lower = 0.0, upper = Inf)
    @test p.name == :σ
    @test p.init == 1.0
    @test p.lower == 0.0
    @test p.upper == Inf

    # Convenience constructor
    p2 = SSMParameter(:μ, 5.0)
    @test p2.init == 5.0
    @test p2.lower == -Inf
    @test p2.upper == Inf

    # Invalid init
    @test_throws ArgumentError SSMParameter(:x; init = 10.0, lower = 0.0, upper = 5.0)
end

# ============================================
# Template Models Tests
# ============================================

@testset "local_level template" begin
    spec = local_level()
    @test spec.name == :LocalLevel
    @test spec.n_states == 1
    @test spec.n_obs == 1
    @test spec.n_shocks == 1
    @test length(spec.params) == 2
    @test param_names(spec) == [:var_obs, :var_level]

    # Check bounds
    lower, upper = param_bounds(spec)
    @test all(lower .>= 0)  # Variance params should be non-negative
    @test all(upper .== Inf)

    # Fixed parameter
    spec_fixed = local_level(var_obs = 225.0)
    @test length(spec_fixed.params) == 1
    @test param_names(spec_fixed) == [:var_level]
end

@testset "local_linear_trend template" begin
    spec = local_linear_trend()
    @test spec.name == :LocalLinearTrend
    @test spec.n_states == 2
    @test spec.n_obs == 1
    @test length(spec.params) == 3
    @test :var_obs in param_names(spec)
    @test :var_level in param_names(spec)
    @test :var_slope in param_names(spec)
end

@testset "ar1 template" begin
    spec = ar1()
    @test spec.n_states == 1
    @test :ρ in param_names(spec)
    @test :var_obs in param_names(spec)
    @test :var_state in param_names(spec)

    # Check ρ bounds
    lower, upper = param_bounds(spec)
    ρ_idx = findfirst(p -> p.name == :ρ, spec.params)
    @test -1 < spec.params[ρ_idx].lower
    @test spec.params[ρ_idx].upper < 1
end

@testset "arma template" begin
    spec = arma(2, 1)
    @test spec.n_states == 2
    # ARMA uses φ1, φ2, θ1 naming (no underscore)
    @test :φ1 in param_names(spec)
    @test :φ2 in param_names(spec)
    @test :θ1 in param_names(spec)
end

# ============================================
# custom_ssm Tests
# ============================================

@testset "custom_ssm basic" begin
    spec = custom_ssm(
        Z = [1.0],
        H = [FreeParam(:var_obs, init = 100.0, lower = 0.0)],
        T = [1.0],
        R = [1.0],
        Q = [FreeParam(:var_level, init = 25.0, lower = 0.0)],
        a1 = [0.0],
        P1 = [1e7],
    )

    @test spec.n_states == 1
    @test spec.n_obs == 1
    @test param_names(spec) == [:var_obs, :var_level]
    @test initial_values(spec) == [100.0, 25.0]
end

@testset "custom_ssm dimension validation" begin
    # Mismatched dimensions should error
    @test_throws DimensionMismatch custom_ssm(
        Z = [1.0 0.0],  # 1x2
        H = [1.0;;],     # 1x1 - OK
        T = [1.0;;],     # 1x1 - wrong! should be 2x2
        R = [1.0;;],
        Q = [1.0;;],
        a1 = [0.0],      # wrong length
        P1 = [1e7;;],
    )
end

@testset "custom_ssm with mixed matrices" begin
    spec = custom_ssm(
        Z = [1.0 0.0],
        H = [FreeParam(:var_obs, init = 1.0, lower = 0.0);;],
        T = [1.0 1.0; 0.0 1.0],
        R = Matrix(1.0I, 2, 2),
        Q = [
            FreeParam(:var_level, init = 0.01, lower = 0.0) 0.0;
            0.0 FreeParam(:var_slope, init = 0.0001, lower = 0.0)
        ],
        a1 = [0.0, 0.0],
        P1 = 1e7 * Matrix(1.0I, 2, 2),
    )

    @test spec.n_states == 2
    @test length(spec.params) == 3
end

# ============================================
# Matrix Helpers Tests
# ============================================

@testset "diag_free" begin
    mat = diag_free([:a, :b, :c], init = 1.0)
    @test size(mat) == (3, 3)
    @test mat[1, 1] isa FreeParam
    @test mat[1, 1].name == :a
    @test mat[1, 2] == 0.0
    @test mat[2, 2].name == :b

    # With prefix
    mat2 = diag_free(3, :σ, init = 2.0)
    @test mat2[1, 1].name == :σ_1
    @test mat2[2, 2].name == :σ_2
    @test mat2[3, 3].name == :σ_3
end

@testset "scalar_free" begin
    mat = scalar_free(:σ, init = 5.0)
    @test length(mat) == 1  # Returns a 1-element vector
    @test mat[1].name == :σ
    @test mat[1].init == 5.0
end

@testset "identity_mat and zeros_mat" begin
    I3 = identity_mat(3)
    @test I3 == Matrix(1.0I, 3, 3)

    Z = zeros_mat(2, 3)
    @test size(Z) == (2, 3)
    @test all(Z .== 0.0)
end

@testset "selection_mat" begin
    R = selection_mat(3, 2)
    @test size(R) == (3, 2)
    @test R[1, 1] == 1.0
    @test R[2, 2] == 1.0
    @test R[3, 1] == 0.0
    @test R[3, 2] == 0.0
end

@testset "companion_mat" begin
    T = companion_mat(3, :φ)
    @test size(T) == (3, 3)
    @test T[1, 1] isa FreeParam
    @test T[1, 1].name == :φ_1
    @test T[1, 2] == 1.0
    @test T[2, 3] == 1.0
    @test T[3, 3] == 0.0
end

@testset "lower_triangular_free" begin
    L = lower_triangular_free(3, :L)
    @test L[1, 1] isa FreeParam
    @test L[2, 1] isa FreeParam
    @test L[3, 1] isa FreeParam
    @test L[1, 2] == 0.0
    @test L[1, 3] == 0.0
    @test L[2, 2] isa FreeParam
end

@testset "symmetric_free" begin
    S = symmetric_free(2, :Σ)
    @test S[1, 2] === S[2, 1]  # Same reference
    @test S[1, 1] isa FreeParam
    @test S[1, 1].name == :Σ_1_1
    @test S[1, 2].name == :Σ_2_1
end

# ============================================
# MatrixExpr Tests
# ============================================

@testset "MatrixExpr basic" begin
    maturities = [3, 6, 12, 24, 60, 120]
    Z = build_dns_loadings(maturities)

    @test Z.dims == (6, 3)
    @test length(Z.params) == 1
    @test Z.params[1].name == :λ
end

@testset "MatrixExpr in custom_ssm" begin
    maturities = [3, 6, 12]
    Z = build_dns_loadings(maturities)

    spec = custom_ssm(
        Z = Z,
        H = diag_free(3, :var_obs, init = 0.01),
        T = [
            FreeParam(:φ_L, init = 0.9, lower = 0.0, upper = 0.9999) 0.0 0.0;
            0.0 FreeParam(:φ_S, init = 0.9, lower = 0.0, upper = 0.9999) 0.0;
            0.0 0.0 FreeParam(:φ_C, init = 0.9, lower = 0.0, upper = 0.9999)
        ],
        R = identity_mat(3),
        Q = diag_free([:var_L, :var_S, :var_C], init = 0.01),
        a1 = [0.0, 0.0, 0.0],
        P1 = 1e6 * identity_mat(3),
    )

    @test :λ in param_names(spec)
    @test haskey(spec.matrix_exprs, :Z)

    # Test building model
    θ = initial_values(spec)
    y = randn(3, 10)
    ss = build_linear_state_space(spec, θ, y)

    # Z matrix should have DNS structure
    @test size(ss.p.Z) == (3, 3)
    @test all(ss.p.Z[:, 1] .≈ 1.0)  # Level column is all 1s
end

@testset "dns_loading functions" begin
    λ = 0.0609
    τ = 12.0

    l1 = dns_loading1(λ, τ)
    l2 = dns_loading2(λ, τ)

    @test l1 > 0
    @test l1 < 1
    @test l2 > 0
    @test l2 < l1

    # Edge case: very small λτ
    @test dns_loading1(1e-12, 1.0) ≈ 1.0 atol=1e-6
end

@testset "build_svensson_loadings" begin
    maturities = [3, 6, 12, 24]
    Z = build_svensson_loadings(maturities)

    @test Z.dims == (4, 4)
    @test length(Z.params) == 2
    @test Z.params[1].name == :λ1
    @test Z.params[2].name == :λ2
end

# ============================================
# Build and Filter Tests
# ============================================

@testset "build_linear_state_space" begin
    spec = local_level()
    θ = initial_values(spec)
    y = randn(1, 100)

    ss = build_linear_state_space(spec, θ, y)
    @test haskey(ss, :p)
    @test haskey(ss, :a1)
    @test haskey(ss, :P1)
    @test ss.p isa KFParms

    # Should be usable with kalman_loglik
    ll = kalman_loglik(ss.p, y, ss.a1, ss.P1)
    @test isfinite(ll)
end

@testset "objective_function" begin
    spec = local_level()
    y = randn(1, 100) .* 10 .+ 100

    negloglik = objective_function(spec, y)
    θ = initial_values(spec)

    val = negloglik(θ)
    @test isfinite(val)
    @test val > 0  # Negative log-likelihood should be positive
end

# ============================================
# SSMLogDensity Tests
# ============================================

@testset "transform_to_constrained" begin
    spec = local_level()

    # Initial values (vector)
    θ_c_vec = initial_values(spec)
    θ_u = transform_to_unconstrained(spec, θ_c_vec)

    # Round-trip - transform_to_constrained returns NamedTuple
    θ_c_nt, logjac = transform_to_constrained(spec, θ_u)
    @test collect(values(θ_c_nt)) ≈ θ_c_vec atol=1e-10

    # Unconstrained [0,0] should give positive constrained values
    θ_test, _ = transform_to_constrained(spec, [0.0, 0.0])
    @test all(v -> v > 0, values(θ_test))
end

@testset "SSMLogDensity" begin
    spec = local_level()
    y = randn(1, 100) .* 10 .+ 100

    ld = SSMLogDensity(spec, y)

    θ_u = transform_to_unconstrained(spec, initial_values(spec))
    ll = logdensity(ld, θ_u)

    @test isfinite(ll)
end

@testset "SSMLogDensity with prior" begin
    spec = local_level()
    y = randn(1, 100) .* 10 .+ 100

    # NormalPrior now takes NamedTuple arguments (variance parameters)
    prior = NormalPrior(
        (var_obs = 25.0, var_level = 25.0),
        (var_obs = 100.0, var_level = 100.0),
    )
    ld = SSMLogDensity(spec, y; prior = prior)

    θ_u = transform_to_unconstrained(spec, initial_values(spec))
    ll_with_prior = logdensity(ld, θ_u)

    ld_no_prior = SSMLogDensity(spec, y)
    ll_no_prior = logdensity(ld_no_prior, θ_u)

    @test ll_with_prior != ll_no_prior
end

@testset "FlatPrior" begin
    prior = FlatPrior()
    @test prior([1.0, 2.0, 3.0]) == 0.0
end

@testset "NormalPrior" begin
    # NormalPrior takes NamedTuple arguments
    prior = NormalPrior((a = 0.0, b = 0.0), (a = 1.0, b = 1.0))

    # At mean, should be maximum
    ll_at_mean = prior((a = 0.0, b = 0.0))
    ll_away = prior((a = 2.0, b = 2.0))
    @test ll_at_mean > ll_away
end

@testset "NormalPriorVec" begin
    # NormalPriorVec uses vectors for backwards compatibility
    prior = NormalPriorVec([0.0, 0.0], [1.0, 1.0])

    # At mean, should be maximum
    ll_at_mean = prior([0.0, 0.0])
    ll_away = prior([2.0, 2.0])
    @test ll_at_mean > ll_away
end

@testset "CompositePrior" begin
    # Using NamedTuple-based priors
    p1 = NormalPrior((x = 0.0,), (x = 1.0,))
    p2 = NormalPrior((x = 0.0,), (x = 2.0,))
    composite = CompositePrior(p1, p2)

    θ = (x = 1.0,)
    @test composite(θ) ≈ p1(θ) + p2(θ)
end

# ============================================
# LogDensityProblems Interface Tests
# ============================================

@testset "LogDensityProblems interface" begin
    using LogDensityProblems

    spec = local_level()
    y = randn(1, 100)
    ld = SSMLogDensity(spec, y)

    @test LogDensityProblems.dimension(ld) == 2
    @test LogDensityProblems.capabilities(typeof(ld)) ==
          LogDensityProblems.LogDensityOrder{0}()

    θ_u = transform_to_unconstrained(spec, initial_values(spec))
    ll = LogDensityProblems.logdensity(ld, θ_u)
    @test isfinite(ll)
end

# ============================================
# AD-Compatible Filter Tests
# ============================================

@testset "kalman_loglik basic" begin
    # Create simple local level model parameters
    Z = [1.0;;]
    H = [100.0;;]
    T = [1.0;;]
    R = [1.0;;]
    Q = [50.0;;]
    a1 = [0.0]
    P1 = [1e7;;]

    p = Siphon.KFParms(Z, H, T, R, Q)
    y = randn(1, 50) .* 10 .+ 100

    ll = kalman_loglik(p, y, a1, P1)
    @test isfinite(ll)
    @test ll < 0  # Log-likelihood should be negative
end

@testset "kalman_loglik AD compatibility" begin
    using ForwardDiff

    spec = local_level()
    y = randn(1, 50)
    ld = SSMLogDensity(spec, y)

    θ_u = transform_to_unconstrained(spec, initial_values(spec))

    # Test that gradient computation works
    g = ForwardDiff.gradient(θ -> logdensity(ld, θ), θ_u)
    @test length(g) == 2
    @test all(isfinite.(g))

    # Test that Hessian computation works
    H = ForwardDiff.hessian(θ -> logdensity(ld, θ), θ_u)
    @test size(H) == (2, 2)
    @test all(isfinite.(H))
end

@testset "kalman_loglik matrix vs scalar" begin
    # Verify matrix and scalar versions give same results
    Z = [1.0;;]
    H = [100.0;;]
    T = [1.0;;]
    R = [1.0;;]
    Q = [50.0;;]
    a1 = [100.0]
    P1 = [1000.0;;]

    y = randn(1, 50) .* 10 .+ 100

    # Matrix version
    kfparms = Siphon.KFParms(Z, H, T, R, Q)
    ll_matrix = kalman_loglik(kfparms, y, a1, P1)

    # Scalar version
    ll_scalar = kalman_loglik_scalar(1.0, 100.0, 1.0, 1.0, 50.0, 100.0, 1000.0, vec(y))

    @test ll_matrix ≈ ll_scalar rtol=1e-10
end

@testset "kalman_loglik_scalar" begin
    Z = 1.0
    H = 100.0
    T = 1.0
    R = 1.0
    Q = 50.0
    a1 = 0.0
    P1 = 1e7

    y = randn(50) .* 10 .+ 100

    ll = kalman_loglik_scalar(Z, H, T, R, Q, a1, P1, y)
    @test isfinite(ll)
    @test ll < 0
end

@testset "kalman_filter" begin
    spec = local_level()
    θ = initial_values(spec)

    kfparms = Siphon.DSL.build_kfparms(spec, θ)
    a1, P1 = Siphon.DSL.build_initial_state(spec, θ)
    y = randn(1, 50)

    result = kalman_filter(kfparms, y, a1, P1)

    @test hasproperty(result, :loglik)
    @test hasproperty(result, :at)
    @test hasproperty(result, :Pt)
    @test hasproperty(result, :vt)
    @test hasproperty(result, :Ft)
    @test hasproperty(result, :Kt)

    @test size(result.at) == (1, 50)      # n states: one per observation
    @test size(result.Pt) == (1, 1, 50)
    @test isfinite(result.loglik)
end

# =============================================================================
# Smoother tests
# =============================================================================

@testset "kalman_smoother basic" begin
    # Simple local level model
    Z = [1.0;;]
    H = [100.0;;]
    T = [1.0;;]
    R = [1.0;;]
    Q = [50.0;;]
    a1 = [100.0]
    P1 = [1000.0;;]

    p = Siphon.KFParms(Z, H, T, R, Q)
    y = randn(1, 30) .* 10 .+ 100

    # Run filter
    result = kalman_filter(p, y, a1, P1)

    # Run smoother
    smooth = kalman_smoother(Z, T, result.at, result.Pt, result.vt, result.Ft)
    alpha_smooth = smooth.alpha
    V_smooth = smooth.V

    @test size(alpha_smooth) == (1, 30)
    @test size(V_smooth) == (1, 1, 30)

    # Smoothed variances should be positive
    @test all(V_smooth[1, 1, :] .> 0)

    # Smoothed variance should be <= predicted variance (smoothing uses more info)
    for t = 1:30
        @test V_smooth[1, 1, t] <= result.Pt[1, 1, t] + 1e-10
    end
end

@testset "kalman_smoother Nile reference" begin
    using DelimitedFiles

    # Load Nile data
    nile = readdlm(joinpath(@__DIR__, "Nile.csv"), ',', Float64)
    y = reshape(nile[:, 1], 1, :)

    # MLE estimates from Durbin & Koopman (2012)
    # σ²_ε (observation) = 15099
    # σ²_η (state) = 1469.1
    Z = [1.0;;]
    H = [15099.0;;]
    T = [1.0;;]
    R = [1.0;;]
    Q = [1469.1;;]
    a1 = [0.0]
    P1 = [1e7;;]  # Large initial variance (approximate diffuse)

    p = Siphon.KFParms(Z, H, T, R, Q)

    # Run filter and smoother
    result = kalman_filter(p, y, a1, P1)
    smooth = kalman_smoother(Z, T, result.at, result.Pt, result.vt, result.Ft)
    alpha_smooth = smooth.alpha
    V_smooth = smooth.V

    # The last smoothed value should match D&K closely (t=100: 798.4)
    # since by then the initialization effect has dissipated
    @test isapprox(alpha_smooth[1, 100], 798.4, rtol = 0.01)

    # Smoothed states should be positively correlated with observations
    @test cor(vec(alpha_smooth), vec(y)) > 0.7

    # Smoothed variances should all be positive
    @test all(V_smooth[1, 1, :] .> 0)

    # Variance should be smaller than predicted variance (smoothing uses more info)
    for t = 1:size(y, 2)
        @test V_smooth[1, 1, t] <= result.Pt[1, 1, t] + 1e-10
    end
end

@testset "kalman_smoother AD compatibility" begin
    using ForwardDiff

    # Test that smoother is AD-compatible
    Z = [1.0;;]
    H = [100.0;;]
    T = [1.0;;]
    R = [1.0;;]
    Q = [50.0;;]
    a1 = [100.0]
    P1 = [1000.0;;]

    p = Siphon.KFParms(Z, H, T, R, Q)
    y = randn(1, 20) .* 10 .+ 100

    # Define a function that computes sum of smoothed states
    function smooth_sum(θ)
        H_new = [θ[1];;]
        Q_new = [θ[2];;]
        p_new = Siphon.KFParms(Z, H_new, T, R, Q_new)
        result = kalman_filter(p_new, y, a1, P1)
        smooth = kalman_smoother(Z, T, result.at, result.Pt, result.vt, result.Ft)
        return sum(smooth.alpha)
    end

    θ = [100.0, 50.0]
    g = ForwardDiff.gradient(smooth_sum, θ)

    @test length(g) == 2
    @test all(isfinite.(g))
end

@testset "kalman_filter_and_smooth" begin
    Z = [1.0;;]
    H = [100.0;;]
    T = [1.0;;]
    R = [1.0;;]
    Q = [50.0;;]
    a1 = [100.0]
    P1 = [1000.0;;]

    p = Siphon.KFParms(Z, H, T, R, Q)
    y = randn(1, 30) .* 10 .+ 100

    result = kalman_filter_and_smooth(p, y, a1, P1)

    @test haskey(result, :loglik)
    @test haskey(result, :a_filtered)
    @test haskey(result, :P_filtered)
    @test haskey(result, :alpha_smooth)
    @test haskey(result, :V_smooth)

    @test isfinite(result.loglik)
    @test size(result.alpha_smooth) == (1, 30)
    @test all(result.V_smooth[1, 1, :] .> 0)
end

@testset "kalman_smoother_scalar" begin
    Z = 1.0
    H = 100.0
    T = 1.0
    R = 1.0
    Q = 50.0
    a1 = 100.0
    P1 = 1000.0

    y = randn(30) .* 10 .+ 100

    # Run scalar filter
    result_scalar = Siphon.kalman_filter_scalar(Z, H, T, R, Q, a1, P1, y)

    # Run scalar smoother
    alpha_smooth, V_smooth = kalman_smoother_scalar(
        Z,
        T,
        result_scalar.at,
        result_scalar.Pt,
        result_scalar.vt,
        result_scalar.Ft,
    )

    @test length(alpha_smooth) == 30
    @test length(V_smooth) == 30
    @test all(V_smooth .> 0)

    # Compare with matrix version
    Z_m = [1.0;;]
    H_m = [100.0;;]
    T_m = [1.0;;]
    R_m = [1.0;;]
    Q_m = [50.0;;]
    a1_m = [100.0]
    P1_m = [1000.0;;]
    y_m = reshape(y, 1, :)

    p = Siphon.KFParms(Z_m, H_m, T_m, R_m, Q_m)
    result = kalman_filter(p, y_m, a1_m, P1_m)
    smooth_m = kalman_smoother(Z_m, T_m, result.at, result.Pt, result.vt, result.Ft)
    alpha_smooth_m = smooth_m.alpha
    V_smooth_m = smooth_m.V

    @test isapprox(alpha_smooth, vec(alpha_smooth_m), rtol = 1e-10)
    @test isapprox(V_smooth, vec(V_smooth_m[1, 1, :]), rtol = 1e-10)
end

# ============================================
# Unified StateSpaceModel API Tests
# ============================================

@testset "StateSpaceModel with known parameters (NamedTuple)" begin
    spec = local_level()
    θ = (var_obs = 100.0, var_level = 50.0)
    n = 100

    model = StateSpaceModel(spec, θ, n)

    @test model.fitted == true
    @test model.theta_fitted == true
    @test model.converged == true
    @test model.backend == :external
    @test parameters(model) == θ
end

@testset "StateSpaceModel with known parameters (Vector)" begin
    spec = local_level()
    θ_vec = [100.0, 50.0]  # [var_obs, var_level]
    n = 100

    model = StateSpaceModel(spec, θ_vec, n)

    @test model.fitted == true
    @test model.theta_fitted == true
    @test parameters(model).var_obs == 100.0
    @test parameters(model).var_level == 50.0
end

@testset "kalman_loglik(model, y)" begin
    spec = local_level()
    θ = (var_obs = 100.0, var_level = 50.0)
    n = 100
    y = randn(1, n)

    model = StateSpaceModel(spec, θ, n)

    # New unified API
    ll_new = kalman_loglik(model, y)
    @test isfinite(ll_new)

    # Compare with old API (should match)
    ss = build_linear_state_space(spec, initial_values(spec), y)
    # Replace initial values with θ
    θ_vec = [θ.var_obs, θ.var_level]
    ss_θ = build_linear_state_space(spec, θ_vec, y)
    ll_old = kalman_loglik(ss_θ.p, y, ss_θ.a1, ss_θ.P1)

    @test ll_new ≈ ll_old rtol=1e-10
end

@testset "kalman_filter!(model, y)" begin
    spec = local_level()
    θ = (var_obs = 100.0, var_level = 50.0)
    n = 100
    y = randn(1, n)

    model = StateSpaceModel(spec, θ, n)

    # Filter should not be valid before calling kalman_filter!
    @test model.filter_valid == false

    ll = kalman_filter!(model, y)

    @test isfinite(ll)
    @test model.filter_valid == true
    @test loglikelihood(model) ≈ ll

    # Access filtered states
    att = filtered_states(model)
    @test size(att) == (1, n)

    # Predicted states should also be available
    at = predicted_states(model)
    @test size(at) == (1, n)
end

@testset "kalman_smoother!(model)" begin
    spec = local_level()
    θ = (var_obs = 100.0, var_level = 50.0)
    n = 100
    y = randn(1, n)

    model = StateSpaceModel(spec, θ, n)

    # Smoother should fail before filter
    @test_throws ArgumentError kalman_smoother!(model)

    # Run filter first
    kalman_filter!(model, y)
    @test model.smoother_computed == false

    # Now run smoother
    kalman_smoother!(model)
    @test model.smoother_computed == true

    # Access smoothed states
    alpha = smoothed_states(model)
    @test size(alpha) == (1, n)

    V = smoothed_states_cov(model)
    @test size(V) == (1, 1, n)
end

@testset "Unified API equivalence with old API" begin
    spec = local_level()
    θ = (var_obs = 100.0, var_level = 50.0)
    n = 100
    y = randn(1, n)

    # New unified way
    model = StateSpaceModel(spec, θ, n)
    ll_new = kalman_filter!(model, y)
    kalman_smoother!(model)
    alpha_new = smoothed_states(model)

    # Old way with build_linear_state_space
    θ_vec = [θ.var_obs, θ.var_level]
    ss = build_linear_state_space(spec, θ_vec, y)
    filt = kalman_filter(ss.p, y, ss.a1, ss.P1)
    smooth = kalman_smoother(ss.p.Z, ss.p.T, filt.at, filt.Pt, filt.vt, filt.Ft)

    @test ll_new ≈ filt.loglik rtol=1e-10
    @test alpha_new ≈ smooth.alpha rtol=1e-10
end

@testset "StateSpaceModel with local_linear_trend" begin
    spec = local_linear_trend()
    θ = (var_obs = 100.0, var_level = 50.0, var_slope = 10.0)
    n = 100
    y = randn(1, n)

    model = StateSpaceModel(spec, θ, n)
    ll = kalman_filter!(model, y)
    kalman_smoother!(model)

    @test isfinite(ll)
    @test size(smoothed_states(model)) == (2, n)
end

@testset "StateSpaceModel dimension validation" begin
    spec = local_level()
    θ = (var_obs = 100.0, var_level = 50.0)
    n = 100
    y = randn(1, n)

    model = StateSpaceModel(spec, θ, n)

    # Wrong observation dimension
    y_wrong = randn(2, n)
    @test_throws AssertionError kalman_filter!(model, y_wrong)

    # Wrong time dimension
    y_wrong_t = randn(1, 50)
    @test_throws AssertionError kalman_filter!(model, y_wrong_t)
end

@testset "StateSpaceModel parameter vector length validation" begin
    spec = local_level()
    θ_wrong = [100.0]  # Too few parameters
    n = 100

    @test_throws AssertionError StateSpaceModel(spec, θ_wrong, n)
end
