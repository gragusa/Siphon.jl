"""
Tests for model specification: templates, custom_ssm, matrix helpers, MatrixExpr, and SSMLogDensity.
"""

using Test
using Siphon
using LinearAlgebra
using Statistics: cor, var
using SpecialFunctions: gamma
using Random
using ForwardDiff
using CSV
using DataFrames
using Distributions: Distributions

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

@testset "template keywords: a number fixes, a FreeParam estimates" begin
    bounds(spec, name) = (
        p = spec.params[Siphon.param_index(spec, name)]; (p.lower, p.upper, p.init))

    # Unset FreeParam fields take the template's defaults
    @test bounds(local_level(var_obs = FreeParam(init = 225.0)), :var_obs) ==
          (0.0, Inf, 225.0)
    @test bounds(ar1(ρ = FreeParam(upper = 0.9)), :ρ) == (-0.9999, 0.9, 0.8)
    # A template default outside the user's bounds falls back to their midpoint
    @test bounds(ar1(ρ = FreeParam(upper = 0.5)), :ρ)[3] ≈ (-0.9999 + 0.5) / 2
    @test bounds(local_level(var_obs = FreeParam(:var_obs)), :var_obs) == (0.0, Inf, 1.0)

    # A number fixes the parameter
    spec = local_linear_trend(var_slope = 0.0)
    @test param_names(spec) == [:var_obs, :var_level]
    @test Siphon.build_kfparms(spec, (var_obs = 1.0, var_level = 2.0)).Q ==
          [2.0 0.0; 0.0 0.0]

    # Groups take one value for all members or a vector
    spec = arma(2, 1; ar = [FreeParam(init = 0.4), 0.0], ma = FreeParam(init = 0.2))
    @test param_names(spec) == [:φ1, :θ1, :var]
    @test initial_values(spec)[1:2] == [0.4, 0.2]
    @test Siphon.build_kfparms(spec, (φ1 = 0.4, θ1 = 0.2, var = 1.0)).T ==
          [0.4 1.0; 0.0 0.0]
    dfm = dynamic_factor(3, 1; ar = FreeParam(init = 0.3), var_factor = 2.0)
    @test bounds(dfm, :φ_1_1) == (-0.9999, 0.9999, 0.3)
    @test :var_factor_1 ∉ param_names(dfm)
    @test param_names(dynamic_factor(3, 1; correlated_errors = true,
        var_obs = FreeParam(init = 4.0)))[3:5] == [:H_sd_1, :H_sd_2, :H_sd_3]
    @test bounds(
        dynamic_factor(3, 1; correlated_errors = true,
            var_obs = FreeParam(init = 4.0)), :H_sd_1)[3] == 2.0

    # A fixed DNS decay gives a fixed loading matrix
    mats = [3, 12, 60]
    fixed = dns_model(mats; λ = 0.0609, Q_structure = :diagonal)
    free = dns_model(mats; Q_structure = :diagonal)
    @test :λ ∉ param_names(fixed) && !haskey(fixed.matrix_exprs, :Z)
    θ = NamedTuple{Tuple(param_names(fixed))}(Tuple(initial_values(fixed)))
    @test Siphon.build_kfparms(fixed, θ).Z ≈
          Siphon.build_kfparms(free, merge(θ, (λ = 0.0609,))).Z
    @test param_names(dns_model(mats; T_structure = :full, T = fill(FreeParam(), 9)))[2:10] ==
          [Symbol("T_$(i)_$(j)") for i in 1:3 for j in 1:3]

    # Old forms and misuse fail with a message that says what to pass
    @test_throws "must be a number (to fix the parameter) or a FreeParam" local_level(var_obs = :free)
    @test_throws "or a FreeParam" local_level(var_obs = (init = 1.0,))
    @test_throws "pass a FreeParam without a name" local_level(var_obs = FreeParam(:x))
    @test_throws "covers 2 parameters" arma(2, 0; ar = [0.1])
    @test_throws "cannot carry a name" dynamic_factor(3, 1; factor_lags = 2, ar = FreeParam(:φ))
    @test_throws "sets only starting variances" dns_model(mats; Q = FreeParam(lower = 0.0))
    @test_throws "sets only starting variances" dynamic_factor(
        3, 1; correlated_errors = true,
        var_obs = 1.0)
    @test_throws MethodError arma(1, 0; ar_init = [0.5])
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
        P1 = [1e7]
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
        P1 = [1e7;;]
    )
end

@testset "custom_ssm with mixed matrices" begin
    spec = custom_ssm(
        Z = [1.0 0.0],
        H = [FreeParam(:var_obs, init = 1.0, lower = 0.0);;],
        T = [1.0 1.0; 0.0 1.0],
        R = Matrix(1.0I, 2, 2),
        Q = [FreeParam(:var_level, init = 0.01, lower = 0.0) 0.0;
             0.0 FreeParam(:var_slope, init = 0.0001, lower = 0.0)],
        a1 = [0.0, 0.0],
        P1 = 1e7 * Matrix(1.0I, 2, 2)
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

@testset "custom_ssm accepts I and Diagonal" begin
    spec = custom_ssm(Z = [1.0 0.0], H = Diagonal([2.0]), T = [1.0 1.0; 0.0 1.0],
        R = I, Q = diag_free([:var_level, :var_slope]), a1 = zeros(2), P1 = 1e7I)
    @test spec.n_shocks == 2
    θ = (var_level = 1.0, var_slope = 2.0)
    kf = Siphon.build_kfparms(spec, θ)
    @test kf.R == Matrix(1.0I, 2, 2)
    @test kf.H == [2.0;;]
    @test Siphon.build_initial_state(spec, θ)[2] == 1e7 * Matrix(1.0I, 2, 2)
    # Q = I takes its size from the columns of R
    spec = custom_ssm(Z = [1.0 0.0], H = [1.0], T = I, R = [1.0; 0.0;;], Q = I,
        a1 = zeros(2), P1 = I)
    @test spec.n_shocks == 1
end

@testset "FreeParam fields and defaults" begin
    # Positional order matches SSMParameter(name, lower, upper, init)
    fp = FreeParam(:ρ, -1, 1, 0.5)
    @test (fp.name, fp.lower, fp.upper, fp.init) == (:ρ, -1.0, 1.0, 0.5)
    @test FreeParam().name === nothing && FreeParam().init === nothing
    @test !isdefined(Siphon, :P) && !isdefined(Siphon, Symbol("@P"))

    # An unset init lies strictly inside the bounds and is flagged as a default
    p = SSMParameter(FreeParam(:v; lower = 0.0))
    @test p.init == 1.0 && p.default_init
    @test SSMParameter(FreeParam(:ρ; lower = -1, upper = 1)).init == 0.0
    @test SSMParameter(FreeParam(:u; upper = 2.0)).init == 1.0
    @test SSMParameter(FreeParam(:x)).init == 0.0
    @test !SSMParameter(FreeParam(:v; lower = 0.0, init = 3.0)).default_init

    @test_throws "strictly inside the bounds" FreeParam(:v; lower = 0.0, init = 0.0)
    @test_throws "strictly inside the bounds" SSMParameter(:v; lower = 0.0, init = 0.0)
    @test_throws "must be below upper bound" FreeParam(:v; lower = 1.0, upper = 1.0)

    base = (H = [1.0], T = [1.0], R = [1.0], Q = [1.0], a1 = [0.0], P1 = [1.0])
    @test_throws "needs a name" custom_ssm(; Z = [FreeParam(; init = 1.0)], base...)

    # A repeated name is one shared parameter, and its settings must agree
    shared = custom_ssm(;
        base..., Z = [FreeParam(:a; init = 1.0) FreeParam(:a; init = 1.0)],
        T = I, R = I, Q = I, a1 = zeros(2), P1 = I)
    @test param_names(shared) == [:a]
    @test_throws "declared twice with different settings" custom_ssm(; base...,
        Z = [FreeParam(:a; init = 1.0) FreeParam(:a; init = 2.0)],
        T = I, R = I, Q = I, a1 = zeros(2), P1 = I)
    @test_throws "declared twice" custom_ssm(; base...,
        Z = [FreeParam(:Q_sd_1; init = 1.0)], Q = cov_free(1, :Q; init_sd = 2.0))
end

@testset "cov_free names and init_sd" begin
    spec = custom_ssm(
        Z = [1.0 0.0], H = [1.0], T = I, R = I, Q = cov_free(2, :Q; init_sd = 2),
        a1 = zeros(2), P1 = I)
    @test param_names(spec) == [:Q_sd_1, :Q_sd_2, :Q_corr_1]
    @test initial_values(spec)[1:2] == [2.0, 2.0]
    @test_throws "init_sd must be positive" cov_free(2, :Q; init_sd = 0)
end

@testset "spec supertype" begin
    @test local_level() isa AbstractSSMSpec
    @test hasmethod(param_names, Tuple{AbstractSSMSpec})
    @test dns_model([3, 12, 60]; Q_structure = :diagonal, H_structure = :scalar) |>
          param_names == [:λ, :T_L, :T_S, :T_C, :var_obs, :var_L, :var_S, :var_C]
end

@testset "selection_mat" begin
    R = selection_mat(3, 2)
    @test size(R) == (3, 2)
    @test R[1, 1] == 1.0
    @test R[2, 2] == 1.0
    @test R[3, 1] == 0.0
    @test R[3, 2] == 0.0
end

@testset "companion_free" begin
    T = companion_free(3, :φ)
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

@testset "block_diag basic shapes" begin
    # Two diagonal blocks of free params
    B = block_diag(diag_free([:a, :b]), diag_free([:c]))
    @test size(B) == (3, 3)
    @test B[1, 1] isa FreeParam && B[1, 1].name == :a
    @test B[2, 2] isa FreeParam && B[2, 2].name == :b
    @test B[3, 3] isa FreeParam && B[3, 3].name == :c
    # Off-block entries are zero (not FreeParam)
    @test B[1, 2] == 0.0 && B[1, 3] == 0.0
    @test B[2, 1] == 0.0 && B[2, 3] == 0.0
    @test B[3, 1] == 0.0 && B[3, 2] == 0.0

    # Free + fixed mix
    B2 = block_diag(diag_free([:q1, :q2]), [3.0;;])
    @test size(B2) == (3, 3)
    @test B2[3, 3] == 3.0
    @test B2[1, 3] == 0.0
end

@testset "block_diag rectangular blocks" begin
    # Independent column-vector loadings for two factor groups
    Z = block_diag([FreeParam(:λ1); FreeParam(:λ2)],
        [FreeParam(:λ3); FreeParam(:λ4); FreeParam(:λ5)])
    @test size(Z) == (5, 2)
    @test Z[1, 1].name == :λ1
    @test Z[2, 1].name == :λ2
    @test Z[3, 2].name == :λ3
    @test Z[4, 2].name == :λ4
    @test Z[5, 2].name == :λ5
    @test Z[1, 2] == 0.0
    @test Z[3, 1] == 0.0
end

@testset "block_diag scalar and FreeParam blocks" begin
    # Scalars and a single FreeParam are treated as 1×1 blocks
    B = block_diag(2.0, FreeParam(:ρ; init = 0.5),
        diag_free([:σ]))
    @test size(B) == (3, 3)
    @test B[1, 1] == 2.0
    @test B[2, 2] isa FreeParam && B[2, 2].name == :ρ
    @test B[3, 3] isa FreeParam && B[3, 3].name == :σ
end

@testset "block_diag preserves shared FreeParam refs" begin
    # symmetric_free ties the (i,j)/(j,i) cells via identical FreeParam objects;
    # block_diag must keep that identity intact.
    S = symmetric_free(2, :Σ)
    B = block_diag([1.0;;], S)
    @test B[2, 3] === B[3, 2]
    @test B[2, 3] === S[1, 2]
end

@testset "block_diag with custom_ssm" begin
    # Uses block_diag for both T and Q in a 2-state, 1-obs SSM made of two
    # independent AR(1) components.
    T = block_diag(FreeParam(:ρ1; init = 0.5, lower = -0.99, upper = 0.99),
        FreeParam(:ρ2; init = 0.3, lower = -0.99, upper = 0.99))
    Q = block_diag(diag_free([:q1, :q2], init = 1.0))
    spec = custom_ssm(
        Z = [1.0 1.0],
        H = [FreeParam(:h; init = 1.0, lower = 0.0)],
        T = T,
        R = I,
        Q = Q,
        a1 = [0.0, 0.0],
        P1 = [10.0 0.0; 0.0 10.0],
        name = :TwoAR1
    )
    @test n_params(spec) == 5  # ρ1, ρ2, h, q1, q2
    names = param_names(spec)
    @test :ρ1 in names && :ρ2 in names
    @test :q1 in names && :q2 in names
    @test :h in names
    @test spec.n_states == 2
    @test spec.n_obs == 1
end

@testset "block_diag errors" begin
    @test_throws ArgumentError block_diag()
    @test_throws ArgumentError block_diag("not a matrix")
    # CovFree / MatrixExpr explicitly unsupported (clearer error)
    @test_throws ArgumentError block_diag(cov_free(2, :Σ), [1.0;;])
end

# ============================================
# MatrixExpr Tests
# ============================================

@testset "MatrixExpr basic" begin
    maturities = [3, 6, 12, 24, 60, 120]
    Z = Siphon.build_dns_loadings(maturities)

    @test Z.dims == (6, 3)
    @test length(Z.params) == 1
    @test Z.params[1].name == :λ
end

@testset "MatrixExpr in custom_ssm" begin
    maturities = [3, 6, 12]
    Z = Siphon.build_dns_loadings(maturities)

    spec = custom_ssm(
        Z = Z,
        H = diag_free(3, :var_obs, init = 0.01),
        T = [FreeParam(:φ_L, init = 0.9, lower = 0.0, upper = 0.9999) 0.0 0.0;
             0.0 FreeParam(:φ_S, init = 0.9, lower = 0.0, upper = 0.9999) 0.0;
             0.0 0.0 FreeParam(:φ_C, init = 0.9, lower = 0.0, upper = 0.9999)],
        R = I,
        Q = diag_free([:var_L, :var_S, :var_C], init = 0.01),
        a1 = [0.0, 0.0, 0.0],
        P1 = 1e6I
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

    l1 = Siphon.dns_loading1(λ, τ)
    l2 = Siphon.dns_loading2(λ, τ)

    @test l1 > 0
    @test l1 < 1
    @test l2 > 0
    @test l2 < l1

    # Edge case: very small λτ
    @test Siphon.dns_loading1(1e-12, 1.0) ≈ 1.0 atol=1e-6
end

@testset "build_svensson_loadings" begin
    maturities = [3, 6, 12, 24]
    Z = Siphon.build_svensson_loadings(maturities)

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
    prior = ParameterPriors(var_obs = InverseGammaPrior(2.0, 100.0),
        var_level = NormalPrior(25.0, 100.0))
    ld = SSMLogDensity(spec, y; prior)
    ld_flat = SSMLogDensity(spec, y)
    @test ld_flat.prior isa FlatPrior

    θ_u = transform_to_unconstrained(spec, (var_obs = 80.0, var_level = 30.0))
    θ, _ = transform_to_constrained(spec, θ_u)
    @test logdensity(ld, θ_u) - logdensity(ld_flat, θ_u) ≈ prior(θ)
    @test all(isfinite, ForwardDiff.gradient(u -> logdensity(ld, u), θ_u))

    # A plain function θ -> log p(θ) is accepted
    ld_fun = SSMLogDensity(spec, y; prior = θ -> -θ.var_obs)
    @test logdensity(ld_fun, θ_u) - logdensity(ld_flat, θ_u) ≈ -80.0

    @test_throws "which the spec does not estimate; its parameters are var_obs, var_level" SSMLogDensity(
        spec, y; prior = ParameterPriors(σ_obs = NormalPrior(0.0, 1.0)))
    @test_throws "which the spec does not estimate" SSMLogDensity(spec, y;
        prior = CompositePrior(FlatPrior(), ParameterPriors(σ_obs = NormalPrior(0.0, 1.0))))
    @test_throws "use FlatPrior() for no prior" SSMLogDensity(spec, y; prior = nothing)
end

@testset "FlatPrior" begin
    prior = FlatPrior()
    @test prior isa AbstractPrior
    @test prior((a = 1.0, b = 2.0)) == 0
end

@testset "NormalPrior" begin
    p = NormalPrior(1, 2)
    @test p isa NormalPrior{Float64}
    @test p(0.5) ≈ -0.5 * (0.25)^2 - log(2.0) - 0.5 * log(2π)
    @test p(1.0) > p(3.0)
    @test_throws "σ must be positive" NormalPrior(0.0, 0.0)
end

@testset "InverseGammaPrior" begin
    # Density of the parameter itself (a variance), no change of variables
    p = InverseGammaPrior(3, 2.0)
    x = 0.7
    @test p(x) ≈ 3 * log(2.0) - log(gamma(3.0)) - 4 * log(x) - 2.0 / x
    @test p(0.0) == -Inf
    @test p(-1.0) == -Inf
    @test_throws "α and β must be positive" InverseGammaPrior(0.0, 1.0)
end

@testset "ParameterPriors" begin
    prior = ParameterPriors(a = NormalPrior(0.0, 1.0), b = InverseGammaPrior(2.0, 1.0))
    θ = (b = 0.5, a = 0.3, c = 10.0)  # order-free; unnamed `c` is flat
    @test prior(θ) ≈ NormalPrior(0.0, 1.0)(0.3) + InverseGammaPrior(2.0, 1.0)(0.5)
    @test (@inferred prior(θ)) isa Float64
    @test ParameterPriors((a = NormalPrior(0.0, 1.0),))((a = 0.0,)) ≈ -0.5 * log(2π)
    @test_throws "use FlatPrior() for no prior" ParameterPriors()
    @test_throws "the prior for `a` is a Float64" ParameterPriors(a = 1.0)
end

@testset "ParameterPriors with Distributions" begin
    prior = ParameterPriors(a = Distributions.Normal(0.0, 2.0),
        b = Distributions.InverseGamma(2.0, 1.0), c = NormalPrior(1.0, 1.0))
    θ = (a = 0.3, b = 0.5, c = 0.0)
    @test prior(θ) ≈
          Distributions.logpdf(Distributions.Normal(0.0, 2.0), 0.3) +
          Distributions.logpdf(Distributions.InverseGamma(2.0, 1.0), 0.5) +
          NormalPrior(1.0, 1.0)(0.0)
    @test InverseGammaPrior(2.0, 1.0)(0.5) ≈
          Distributions.logpdf(Distributions.InverseGamma(2.0, 1.0), 0.5)

    # A prior with narrower support than the parameter's bounds gives -Inf outside it
    spec = local_level()
    ld = SSMLogDensity(spec, randn(1, 50);
        prior = ParameterPriors(var_level = Distributions.Beta(2, 2)))
    @test logdensity(ld, transform_to_unconstrained(spec, (
        var_obs = 1.0, var_level = 2.0))) ==
          -Inf
    @test isfinite(logdensity(ld,
        transform_to_unconstrained(spec, (var_obs = 1.0, var_level = 0.5))))
end

@testset "CompositePrior" begin
    p1 = ParameterPriors(x = NormalPrior(0.0, 1.0))
    p2 = ParameterPriors(x = NormalPrior(0.0, 2.0))
    composite = CompositePrior(p1, p2, θ -> -θ.x)
    @test composite isa AbstractPrior

    θ = (x = 1.0,)
    @test composite(θ) ≈ p1(θ) + p2(θ) - 1.0
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
    ll_scalar = kalman_loglik(KFParms(1.0, 100.0, 1.0, 1.0, 50.0), vec(y), 100.0, 1000.0)

    @test ll_matrix ≈ ll_scalar rtol=1e-10
end

@testset "kalman_loglik on a scalar KFParms" begin
    Z = 1.0
    H = 100.0
    T = 1.0
    R = 1.0
    Q = 50.0
    a1 = 0.0
    P1 = 1e7

    y = randn(50) .* 10 .+ 100

    ll = kalman_loglik(KFParms(Z, H, T, R, Q), y, a1, P1)
    @test isfinite(ll)
    @test ll < 0
end

@testset "kalman_filter" begin
    spec = local_level()
    θ = initial_values(spec)

    kfparms = Siphon.build_kfparms(spec, θ)
    a1, P1 = Siphon.build_initial_state(spec, θ)
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
    smooth = kalman_smoother(result)
    alpha_smooth = smoothed_states(smooth)
    V_smooth = smoothed_states_cov(smooth)

    @test size(alpha_smooth) == (1, 30)
    @test size(V_smooth) == (1, 1, 30)

    # Smoothed variances should be positive
    @test all(V_smooth[1, 1, :] .> 0)

    # Smoothed variance should be <= predicted variance (smoothing uses more info)
    for t in 1:30
        @test V_smooth[1, 1, t] <= result.Pt[1, 1, t] + 1e-10
    end
end

@testset "kalman_smoother Nile reference" begin
    using CSV
    using DataFrames

    # Load Nile data
    nile = CSV.read(joinpath(@__DIR__, "Nile.csv"), DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)

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
    smooth = kalman_smoother(result)
    alpha_smooth = smoothed_states(smooth)
    V_smooth = smoothed_states_cov(smooth)

    # The last smoothed value should match D&K closely (t=100: 798.4)
    # since by then the initialization effect has dissipated
    @test isapprox(alpha_smooth[1, 100], 798.4, rtol = 0.01)

    # Smoothed states should be positively correlated with observations
    @test cor(vec(alpha_smooth), vec(y)) > 0.7

    # Smoothed variances should all be positive
    @test all(V_smooth[1, 1, :] .> 0)

    # Variance should be smaller than predicted variance (smoothing uses more info)
    for t in 1:size(y, 2)
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
        return sum(smoothed_states(kalman_smoother(result)))
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

    filt, smooth = kalman_filter_and_smooth(p, y, a1, P1)

    @test isfinite(loglikelihood(filt))
    @test size(smoothed_states(smooth)) == (1, 30)
    @test all(smoothed_states_cov(smooth)[1, 1, :] .> 0)
    @test crosslag_covs(last(kalman_filter_and_smooth(p, y, a1, P1; crosscov = true))) ≈
          crosslag_covs(kalman_smoother(filt; crosscov = true))
end

@testset "kalman_smoother on a scalar KFParms" begin
    Z = 1.0
    H = 100.0
    T = 1.0
    R = 1.0
    Q = 50.0
    a1 = 100.0
    P1 = 1000.0

    y = randn(30) .* 10 .+ 100

    # Run scalar filter
    result_scalar = kalman_filter(KFParms(Z, H, T, R, Q), y, a1, P1)

    # Run smoother on the scalar result
    smooth_scalar = kalman_smoother(result_scalar)
    alpha_smooth = vec(smoothed_states(smooth_scalar))
    V_smooth = vec(smoothed_states_cov(smooth_scalar))

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
    smooth_m = kalman_smoother(result)
    alpha_smooth_m = smoothed_states(smooth_m)
    V_smooth_m = smoothed_states_cov(smooth_m)

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

    # The filter accessors throw before kalman_filter! has run
    @test_throws "kalman_filter!(model, y)" filtered_states(model)

    ll = kalman_filter!(model, y)

    @test isfinite(ll)
    @test size(filtered_states(model)) == (1, n)
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
    smooth = kalman_smoother(filt)

    @test ll_new ≈ filt.loglik rtol=1e-10
    @test alpha_new ≈ smoothed_states(smooth) rtol=1e-10
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
    @test_throws DimensionMismatch kalman_filter!(model, y_wrong)

    # Wrong time dimension
    y_wrong_t = randn(1, 50)
    @test_throws DimensionMismatch kalman_filter!(model, y_wrong_t)
end

@testset "StateSpaceModel parameter vector length validation" begin
    spec = local_level()
    θ_wrong = [100.0]  # Too few parameters
    n = 100

    @test_throws ArgumentError StateSpaceModel(spec, θ_wrong, n)
end

# ============================================
# Maximum likelihood and parameter bounds
# ============================================

@testset "optimize_ssm maximises the log-likelihood" begin
    nile = CSV.read(joinpath(@__DIR__, "Nile.csv"), DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)
    spec = local_level()

    res = optimize_ssm(spec, y; θ0 = (var_obs = 1e4, var_level = 1e4))
    # Durbin & Koopman (2012), Nile data
    @test res.θ.var_obs ≈ 15099.0 rtol = 1e-3
    @test res.θ.var_level ≈ 1469.1 rtol = 1e-3
    @test res.loglik ≈ ssm_loglik(spec, res.θ, y)
    @test res.loglik ≈ -641.5856 atol = 1e-3

    # θ0 as a NamedTuple is matched by name, in any order, and may be partial
    res2 = optimize_ssm(spec, y; θ0 = (var_level = 1e4, var_obs = 1e4))
    @test res2.θ.var_obs ≈ res.θ.var_obs rtol = 1e-4
    res3 = optimize_ssm(spec, y; θ0 = (var_obs = 1e4,))
    @test res3.loglik ≈ res.loglik atol = 1e-4
    @test_throws ArgumentError optimize_ssm(spec, y; θ0 = (σ_obs = 1.0,))
end

@testset "default starting values match the data's scale" begin
    nile = CSV.read(joinpath(@__DIR__, "Nile.csv"), DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)

    # Template variances start at 1.0. Without rescaling, L-BFGS stops at a
    # degenerate optimum on the Nile data (var_obs → 0, loglik ≈ −656).
    res = optimize_ssm(local_level(), y)
    @test res.loglik ≈ -641.5856 atol = 1e-3
    model = fit!(MLE(), StateSpaceModel(local_level(), size(y, 2)), y)
    @test model.loglik ≈ -641.5856 atol = 1e-3

    # Only template defaults are flagged; inits the user gives are not
    @test all(p -> p.default_init, local_level().params)
    @test [p.default_init for p in local_level(var_obs = FreeParam(init = 5.0)).params] ==
          [false, true]
    @test local_level(var_obs = FreeParam(lower = 1.0)).params[1].default_init
    @test last(arma(1, 0).params).default_init
    @test !last(arma(1, 0; var = FreeParam(init = 3.0)).params).default_init
    @test !SSMParameter(:x; init = 1.0, lower = 0.0).default_init

    # A user-given init is used as given; a defaulted one is rescaled
    init = Siphon._initial_constrained(local_level(var_obs = FreeParam(init = 5.0)), nothing, y)
    @test init[1] == 5.0
    @test init[2] ≈ var(diff(vec(y)))
    # Defaults keep the template's ratios between variances
    init = Siphon._initial_constrained(local_linear_trend(), (var_slope = 2.0,), y)
    @test init[2] / init[1] ≈ 0.01
    @test init[3] == 2.0

    # On a trending series the scale comes from differences, not levels
    rng = Random.Xoshiro(1)
    n = 200
    μ = cumsum(cumsum(0.5 .+ 0.1 .* randn(rng, n)) .+ 2 .* randn(rng, n))
    yt = reshape(1000 .+ μ .+ 5 .* randn(rng, n), 1, :)
    spec = local_linear_trend()
    ref = optimize_ssm(spec, yt; θ0 = (var_obs = 25.0, var_level = 4.0, var_slope = 0.01))
    @test optimize_ssm(spec, yt).loglik ≈ ref.loglik atol = 1e-3
end

@testset "exact diffuse initialisation through a spec" begin
    nile = CSV.read(joinpath(@__DIR__, "Nile.csv"), DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)
    spec = local_level(diffuse = :exact)
    θ = (var_obs = 15099.0, var_level = 1469.1)
    p = KFParms([1.0;;], [15099.0;;], [1.0;;], [1.0;;], [1469.1;;])
    exact = kalman_loglik(p, y, [0.0], zeros(1, 1), ones(1, 1))

    ss = build_linear_state_space(spec, θ, y)
    @test ss.P1_inf == ones(1, 1)
    @test build_linear_state_space(local_level(), θ, y).P1_inf === nothing

    # Every spec-based path uses the diffuse filter. KFAS gives −632.55.
    @test ssm_loglik(spec, θ, y) ≈ exact
    @test ssm_loglik(spec, θ, y) ≈ -632.55 atol = 0.01
    @test Siphon._spec_loglik(spec, θ, y, :auto) ≈ exact
    @test objective_function(spec, y)([15099.0, 1469.1]) ≈ -exact
    filt = kalman_filter(p, y, [0.0], zeros(1, 1), ones(1, 1))
    @test predict(spec, θ, y) ≈ predicted_observations(filt)
    @test forecast(spec, θ, y, 2).yhat ≈ fill(filt.att[1, end], 1, 2)
    @test size(forecast_paths(spec, θ, y, 3, 4)) == (1, 3, 4)
    @test optimize_ssm(spec, y).loglik ≈ -632.5456 atol = 1e-3
    @test logdensity(SSMLogDensity(spec, y), transform_to_unconstrained(spec, collect(θ))) ≈
          exact +
          transform_to_constrained(spec, transform_to_unconstrained(spec, collect(θ)))[2]

    # StateSpaceModel: filtering, smoothing and MLE run; EM rejects it
    model = StateSpaceModel(spec, θ, size(y, 2))
    @test kalman_loglik(model, y) ≈ exact
    @test kalman_filter!(model, y) ≈ exact
    @test smoothed_states(model) ≈ smoothed_states(kalman_smoother(filt))
    @test fit!(MLE(), StateSpaceModel(spec, size(y, 2)), y).loglik ≈ -632.5456 atol = 1e-3
    @test_throws ArgumentError fit!(EM(), StateSpaceModel(spec, size(y, 2)), y)

    # arma accepts the same diffuse options as the other templates
    @test uses_exact_diffuse(arma(1, 1; diffuse = :exact))
    @test !uses_exact_diffuse(arma(1, 1))
    θa = (φ1 = 0.5, θ1 = 0.2, var = 1.0)
    @test Siphon.build_initial_state(arma(1, 1; diffuse = false), θa)[2] == 1e4 * I(2)

    # A cov_free initial covariance is built from its parameters
    cs = custom_ssm(Z = [1.0], H = [1.0], T = [1.0], R = [1.0], Q = [1.0],
        a1 = [0.0], P1 = cov_free(1, :P1; init_sd = 30.0))
    @test Siphon.build_initial_state(cs, (P1_sd_1 = 30.0,))[2] ≈ [900.0;;]
end

@testset "optimize_ssm_with_stderr uses the delta method" begin
    nile = CSV.read(joinpath(@__DIR__, "Nile.csv"), DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)
    spec = local_level()
    res = optimize_ssm_with_stderr(spec, y; θ0 = (var_obs = 1e4, var_level = 1e4))

    # At an interior optimum the delta-method covariance equals the inverse
    # Hessian taken directly in the constrained parameters.
    names = Tuple(param_names(spec))
    Hc = ForwardDiff.hessian(
        v -> -ssm_loglik(spec, NamedTuple{names}(Tuple(v)), y), collect(res.θ))
    @test res.vcov ≈ inv(Hc) rtol = 1e-3
    @test res.stderr ≈ sqrt.(diag(inv(Hc))) rtol = 1e-3
end

@testset "parameter bounds survive the transform" begin
    spec = custom_ssm(
        Z = [1.0],
        H = [FreeParam(:x, init = 2.0, lower = 1.0)],
        T = [FreeParam(:z, init = 0.5, lower = -1.0)],
        R = [1.0],
        Q = [FreeParam(:w, init = 0.5, upper = 3.0)],
        a1 = [0.0],
        P1 = [1e7]
    )
    for u in ([-20.0, -50.0, 50.0], [20.0, 5.0, -5.0])
        θc, _ = transform_to_constrained(spec, u)
        @test θc.x >= 1.0
        @test θc.z >= -1.0
        @test θc.w <= 3.0
    end
    # Round trip
    θ0 = initial_values(spec)
    θc, _ = transform_to_constrained(spec, transform_to_unconstrained(spec, θ0))
    @test collect(θc) ≈ θ0
end

@testset "matrix expressions share a supertype" begin
    @test MatrixExpr <: Siphon.AbstractMatrixExpr
    @test CovMatrixExpr <: Siphon.AbstractMatrixExpr
    @test !(Expr <: Siphon.MatrixElement)

    spec = dns_model([3, 12, 60])
    @test spec.matrix_exprs isa Dict{Symbol, Siphon.AbstractMatrixExpr}
    @test spec.matrix_exprs[:Z] isa MatrixExpr
    spec = custom_ssm(Z = [1.0], H = [FreeParam(:h, init = 1.0, lower = 0.0)],
        T = [1.0], R = [1.0], Q = cov_free(1, :Q), a1 = [0.0], P1 = [1e7])
    @test spec.matrix_exprs[:Q] isa CovMatrixExpr

    # Only a MatrixExpr is read as one; a look-alike is an ordinary input.
    lookalike = (params = SSMParameter{Float64}[], builder = (θ, d) -> ones(1, 1),
        dims = (1, 1))
    @test_throws MethodError custom_ssm(Z = lookalike, H = [1.0], T = [1.0],
        R = [1.0], Q = [1.0], a1 = [0.0], P1 = [1e7])
end

@testset "single export list" begin
    @test !isdefined(Siphon, :DSL)
    @test all(n -> isdefined(Siphon, n), names(Siphon))
end
