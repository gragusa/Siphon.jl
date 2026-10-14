"""
Tests for EM algorithm implementation.
"""

using Test
using Siphon
using LinearAlgebra
using CSV
using DataFrames
using ForwardDiff
using Random

include("em_helpers.jl")

@testset "kalman_smoother basic" begin
    # Simple local level model
    Z = [1.0;;]
    H = [100.0;;]
    T = [1.0;;]
    R = [1.0;;]
    Q = [50.0;;]
    a1 = [0.0]
    P1 = [1e7;;]

    p = KFParms(Z, H, T, R, Q)

    # Generate some data
    Random.seed!(42)
    y = randn(1, 30) .* 10 .+ 100

    filt = kalman_filter(p, y, a1, P1)

    # Without cross-covariances
    result1 = kalman_smoother(filt)
    @test size(smoothed_states(result1)) == (1, 30)
    @test size(smoothed_states_cov(result1)) == (1, 1, 30)
    @test result1.Pcross === nothing

    # With cross-covariances
    result2 = kalman_smoother(filt; crosscov = true)
    @test size(crosslag_covs(result2)) == (1, 1, 29)  # n-1 cross-covariances

    # Cross-covariances should be positive for local level
    @test all(crosslag_covs(result2) .> 0)
end

@testset "fit!(EM(), ...) local_level basic" begin
    spec = local_level()

    # Generate data from known parameters
    true_var_obs = 100.0
    true_var_level = 25.0
    n = 200

    # Simulate local level model
    Random.seed!(123)
    states = zeros(n)
    y_data = zeros(1, n)
    states[1] = 0.0
    for t in 1:n
        if t > 1
            states[t] = states[t - 1] + sqrt(true_var_level) * randn()
        end
        y_data[1, t] = states[t] + sqrt(true_var_obs) * randn()
    end

    # Use unified fit!(EM(), ...) API
    model = StateSpaceModel(spec, n)
    fit!(EM(), model, y_data; maxiter = 100, verbose = false)

    @test isfinite(loglikelihood(model))
end

@testset "fit!(EM(), ...) Nile data" begin
    # Load Nile data
    nile = CSV.read(joinpath(@__DIR__, "Nile.csv"), DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)

    spec = local_level()
    model = StateSpaceModel(spec, size(y, 2))
    fit!(EM(), model, y; maxiter = 300, tol = 1e-8)

    # Get parameter names and values
    names = param_names(spec)
    var_obs_idx = findfirst(==(:var_obs), names)
    var_level_idx = findfirst(==(:var_level), names)

    # Compare with known MLE from Durbin & Koopman
    # var_obs ≈ 15099, var_level ≈ 1469.1
    # Note: fit!(EM(), ...) uses a different code path than _em_local_level
    # so we use slightly higher tolerance
    @test isapprox(model.theta_values[var_obs_idx], 15099.0, rtol = 0.10)
    @test isapprox(model.theta_values[var_level_idx], 1469.1, rtol = 0.10)
end

@testset "EM log-likelihood is monotone" begin
    Random.seed!(456)
    y = randn(1, 50) .* 10 .+ 100
    one_ = ones(1, 1)
    result = em_matrices(one_, one_, one_, fill(100.0, 1, 1), fill(50.0, 1, 1), y,
        [0.0], fill(1e7, 1, 1); Z_free = falses(1, 1), T_free = falses(1, 1),
        maxiter = 50)
    @test all(diff(result.loglik_history) .>= -1e-8)
end

@testset "fit!(EM()) finds the local level MLE" begin
    nile = CSV.read(joinpath(@__DIR__, "Nile.csv"), DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)
    spec = local_level()
    model = StateSpaceModel(spec, size(y, 2))
    fit!(EM(), model, y; maxiter = 5000, tol = 1e-10)

    @test isconverged(model)
    # Durbin & Koopman (2012): var_obs ≈ 15099, var_level ≈ 1469.1, ll ≈ -641.59
    @test isapprox(parameters(model).var_obs, 15099.0, rtol = 0.01)
    @test isapprox(parameters(model).var_level, 1469.1, rtol = 0.01)
    @test isapprox(loglikelihood(model), -641.59, rtol = 0.001)
    @test ssm_loglik(spec, parameters(model), y) ≈ loglikelihood(model)
end

@testset "fit!(EM()) keeps fixed variances" begin
    nile = CSV.read(joinpath(@__DIR__, "Nile.csv"), DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)

    spec = local_level(var_obs = 15099.0)
    @test param_names(spec) == [:var_level]
    model = StateSpaceModel(spec, size(y, 2))
    fit!(EM(), model, y; maxiter = 1000, tol = 1e-8)
    @test isconverged(model)
    @test system_matrices(model).H == fill(15099.0, 1, 1)
    @test isapprox(parameters(model).var_level, 1469.1, rtol = 0.1)
    @test ssm_loglik(spec, parameters(model), y) ≈ loglikelihood(model)

    spec = local_level(var_level = 1469.0)
    @test param_names(spec) == [:var_obs]
    model = StateSpaceModel(spec, size(y, 2))
    fit!(EM(), model, y; maxiter = 1000, tol = 1e-8)
    @test isconverged(model)
    @test system_matrices(model).Q == fill(1469.0, 1, 1)
    @test isapprox(parameters(model).var_obs, 15099.0, rtol = 0.1)
    @test ssm_loglik(spec, parameters(model), y) ≈ loglikelihood(model)
end

@testset "fit!(EM()) with missing periods matches the MLE" begin
    nile = CSV.read(joinpath(@__DIR__, "Nile.csv"), DataFrame; header = false)
    y = reshape(Float64.(nile[!, 1]), 1, :)
    y[1, 21:40] .= NaN
    spec = local_level()
    em = fit!(EM(), StateSpaceModel(spec, size(y, 2)), y; maxiter = 5000, tol = 1e-12)
    mle = fit!(MLE(), StateSpaceModel(spec, size(y, 2)), y)
    @test loglikelihood(em) ≈ loglikelihood(mle) rtol = 1e-6
    @test parameters(em).var_obs ≈ parameters(mle).var_obs rtol = 1e-2
    @test parameters(em).var_level ≈ parameters(mle).var_level rtol = 1e-2
end

@testset "EMResult fields" begin
    Random.seed!(111)
    y = randn(1, 50) .* 10 .+ 100
    one_ = ones(1, 1)
    result = em_matrices(one_, one_, one_, fill(100.0, 1, 1), fill(50.0, 1, 1), y,
        [0.0], fill(1e7, 1, 1); Z_free = falses(1, 1), T_free = falses(1, 1),
        maxiter = 50)

    @test result isa EMResult{Float64}
    @test result.converged isa Bool
    @test result.iterations isa Int
    @test result.loglik isa Float64
    @test length(result.loglik_history) == result.iterations
    @test (result.Z, result.T, result.R) == (one_, one_, one_)
    @test result.a1 == [0.0]
    @test result.P1 == fill(1e7, 1, 1)
    @test occursin("EMResult(p = 1, m = 1", sprint(show, result))
end

@testset "kalman_smoother AD compatibility" begin
    Z = [1.0;;]
    T = [1.0;;]
    a1 = [0.0]
    P1 = [1e7;;]
    Random.seed!(222)
    y = randn(1, 20) .* 10 .+ 100

    function test_func(theta)
        H = [theta[1];;]
        Q = [theta[2];;]
        R = [1.0;;]
        p = KFParms(Z, H, T, R, Q)

        filt = kalman_filter(p, y, a1, P1)
        smooth = kalman_smoother(filt; crosscov = true)

        # Sum of smoothed states and cross-covariances
        return sum(smoothed_states(smooth)) + sum(crosslag_covs(smooth))
    end

    theta = [100.0, 50.0]
    g = ForwardDiff.gradient(test_func, theta)

    @test length(g) == 2
    @test all(isfinite.(g))
end

# ============================================
# Full Covariance EM Tests
# ============================================

@testset "EM with full covariances" begin
    Random.seed!(42)
    Z_true = [1.0 0.0; 0.0 1.0]
    T_true = [0.9 0.0; 0.0 0.8]
    R = [1.0 0.0; 0.0 1.0]
    H_true = [1.0 0.3; 0.3 1.5]
    Q_true = [0.5 0.1; 0.1 0.4]
    n, m, p = 200, 2, 2
    L_Q = cholesky(Symmetric(Q_true)).L
    L_H = cholesky(Symmetric(H_true)).L
    states = zeros(m, n)
    y = zeros(p, n)
    for t in 1:n
        t > 1 && (states[:, t] = T_true * states[:, t - 1] + L_Q * randn(m))
        y[:, t] = Z_true * states[:, t] + L_H * randn(p)
    end

    result = em_matrices(Z_true, [0.5 0.0; 0.0 0.5], R, [2.0 0.0; 0.0 2.0],
        [1.0 0.0; 0.0 1.0], y, zeros(m), 10.0 * Matrix{Float64}(I, m, m);
        Z_free = falses(p, m), maxiter = 500, tol = 1e-5)

    @test result.iterations > 1
    @test isfinite(result.loglik)
    @test all(diff(result.loglik_history) .>= -1e-6)
    @test isposdef(Symmetric(result.H))
    @test isposdef(Symmetric(result.Q))
    @test result.H ≈ result.H' atol = 1e-10
    @test result.Q ≈ result.Q' atol = 1e-10
    @test 0.5 < result.T[1, 1] < 1.0
    @test 0.5 < result.T[2, 2] < 1.0
end

@testset "EM keeps fixed cells" begin
    Random.seed!(456)
    Z = [1.0 0.0; 0.0 1.0]
    T = [0.9 0.0; 0.0 0.8]
    H_init = [1.0 0.0; 0.0 1.0]
    Q_init = [0.5 0.0; 0.0 0.5]
    y = randn(2, 100)

    result = em_matrices(Z, T, Z, H_init, Q_init, y, zeros(2), 10.0 * Matrix(1.0I, 2, 2);
        Z_free = falses(2, 2), T_free = falses(2, 2), maxiter = 100)
    @test result.Z == Z
    @test result.T == T
    @test !(result.H ≈ H_init)
    @test !(result.Q ≈ Q_init)

    # Diagonal masks keep the off-diagonal cells at zero.
    result = em_matrices(Z, T, Z, H_init, Q_init, y, zeros(2), 10.0 * Matrix(1.0I, 2, 2);
        H_free = trues(2), Q_free = trues(2), maxiter = 100)
    @test result.H[1, 2] == result.H[2, 1] == 0.0
    @test result.Q[1, 2] == result.Q[2, 1] == 0.0
    @test all(diff(result.loglik_history) .>= -1e-6)
end

@testset "EM with tied and restricted cells" begin
    Random.seed!(7)
    n = 400
    Q_true = [0.5 0.3; 0.3 0.4]
    L_Q = cholesky(Q_true).L
    α = zeros(2, n)
    for t in 2:n
        α[:, t] = 0.7 * α[:, t - 1] + L_Q * randn(2)
    end
    y = α + 0.3 * randn(2, n)
    I2 = Matrix(1.0I, 2, 2)
    diagmask = Matrix(I2 .== 1)

    # T = φ I with full Q: the two diagonal cells stay equal.
    tied = em_matrices(I2, 0.5 * I2, I2, 0.2 * I2, I2, y, zeros(2), 10 * I2;
        Z_free = falses(2, 2), T_free = diagmask,
        T_ties = [[CartesianIndex(1, 1), CartesianIndex(2, 2)]], maxiter = 300)
    @test tied.T[1, 1] == tied.T[2, 2]
    @test tied.T[1, 2] == tied.T[2, 1] == 0.0
    @test all(diff(tied.loglik_history) .>= -1e-8)
    @test isapprox(tied.T[1, 1], 0.7; atol = 0.1)

    # Diagonal T with full Q: the restricted M-step weights the rows by Q⁻¹,
    # so EM stays monotone.
    diagT = em_matrices(I2, 0.5 * I2, I2, 0.2 * I2, I2, y, zeros(2), 10 * I2;
        Z_free = falses(2, 2), T_free = diagmask, maxiter = 300)
    @test diagT.T[1, 2] == diagT.T[2, 1] == 0.0
    @test all(diff(diagT.loglik_history) .>= -1e-8)
    @test diagT.loglik >= tied.loglik - 1e-8

    # A tied T needs a nonsingular R Q Rᵀ.
    @test_throws "EM cannot update the tied cells of T" em_matrices(
        I2, 0.5 * I2, [1.0 0.0; 0.0 0.0], 0.2 * I2, I2, y, zeros(2), 10 * I2;
        Z_free = falses(2, 2), T_free = diagmask, Q_free = falses(2, 2),
        T_ties = [[CartesianIndex(1, 1), CartesianIndex(2, 2)]], maxiter = 3)
end

@testset "fit!(EM()) with tied parameters" begin
    Random.seed!(11)
    n = 300
    α = zeros(2, n)
    for t in 2:n
        α[:, t] = 0.8 * α[:, t - 1] + [0.7, 0.5] .* randn(2)
    end
    y = α + 0.4 * randn(2, n)
    φ = FreeParam(:φ, init = 0.5, lower = -1.0, upper = 1.0)
    h = FreeParam(:h, init = 1.0, lower = 0.0)
    spec = custom_ssm(
        Z = [1.0 0.0; 0.0 1.0],
        H = [h 0.0; 0.0 h],
        T = [φ 0.0; 0.0 φ],
        R = [1.0 0.0; 0.0 1.0],
        Q = diag_free(2, :q),
        a1 = [0.0, 0.0],
        P1 = [10.0 0.0; 0.0 10.0]
    )
    model = StateSpaceModel(spec, n)
    fit!(EM(), model, y; maxiter = 2000, tol = 1e-10)
    mats = system_matrices(model)
    @test mats.T[1, 1] == mats.T[2, 2] == parameters(model).φ
    @test mats.H[1, 1] == mats.H[2, 2] == parameters(model).h
    @test ssm_loglik(spec, parameters(model), y) ≈ loglikelihood(model)
    mle = fit!(MLE(), StateSpaceModel(spec, n), y)
    @test loglikelihood(model) ≈ loglikelihood(mle) rtol = 1e-5
end

@testset "fit!(EM()) with a symmetric covariance cell" begin
    Random.seed!(12)
    n = 300
    L_H = cholesky([1.0 0.4; 0.4 0.8]).L
    α = zeros(2, n)
    for t in 2:n
        α[:, t] = 0.6 * α[:, t - 1] + 0.5 * randn(2)
    end
    y = α + L_H * randn(2, n)
    h12 = FreeParam(:h12, init = 0.0, lower = -Inf)
    spec = custom_ssm(
        Z = [1.0 0.0; 0.0 1.0],
        H = [FreeParam(:h11, init = 1.0, lower = 0.0) h12;
             h12 FreeParam(:h22, init = 1.0, lower = 0.0)],
        T = [0.6 0.0; 0.0 0.6],
        R = [1.0 0.0; 0.0 1.0],
        Q = [0.25 0.0; 0.0 0.25],
        a1 = [0.0, 0.0],
        P1 = [10.0 0.0; 0.0 10.0]
    )
    model = StateSpaceModel(spec, n)
    fit!(EM(), model, y; maxiter = 1000, tol = 1e-10)
    H = system_matrices(model).H
    @test H[1, 2] == H[2, 1] == parameters(model).h12
    @test parameters(model).h12 > 0.1
    @test ssm_loglik(spec, parameters(model), y) ≈ loglikelihood(model)
end

@testset "fit!(EM()) rejects parameters it cannot update" begin
    spec = dns_model([3, 12, 24, 60, 120]; diffuse = false)
    @test_throws "fit!(EM(), ...) cannot estimate λ (it enters Z through a matrix expression)" fit!(
        EM(), StateSpaceModel(spec, 20), randn(5, 20))
end

@testset "block_diag matches manual block-diagonal spec under EM" begin
    # Build the same 3-state, 2-observation, 2-shock state-space model two
    # ways: once by writing every cell of T/H/Q by hand, and once via
    # block_diag. The model has an AR(2) block and an independent AR(1)
    # block. Running EM with the same data, starting values, and iteration
    # count should produce identical parameter estimates and log-likelihood.
    # Every state has its own shock: EM cannot move a T entry in a row whose
    # state has none.

    # ---- DSL spec via block_diag ----
    T_dsl = block_diag(
        companion_free(2, :φ),
        FreeParam(:ρ; init = 0.5, lower = -0.99, upper = 0.99)
    )
    H_dsl = block_diag(
        FreeParam(:h1; init = 1.0, lower = 0.0),
        FreeParam(:h2; init = 1.0, lower = 0.0)
    )
    Q_dsl = block_diag(
        FreeParam(:q1; init = 1.0, lower = 0.0),
        FreeParam(:q_lag; init = 1.0, lower = 0.0),
        FreeParam(:q2; init = 1.0, lower = 0.0)
    )

    # ---- Manual spec, cell by cell ----
    # companion_free(2, :φ) puts FreeParam(:φ_1, init=0.5) at (1,1) and
    # FreeParam(:φ_2, init=0.25) at (2,1) (init/i convention), with a 1.0
    # on the superdiagonal and zeros everywhere else.
    T_manual = Any[FreeParam(:φ_1, init = 0.5, lower = -Inf, upper = Inf) 1.0 0.0;
                   FreeParam(:φ_2, init = 0.25, lower = -Inf, upper = Inf) 0.0 0.0;
                   0.0 0.0 FreeParam(:ρ, init = 0.5, lower = -0.99, upper = 0.99)]
    H_manual = Any[FreeParam(:h1, init = 1.0, lower = 0.0) 0.0;
                   0.0 FreeParam(:h2, init = 1.0, lower = 0.0)]
    Q_manual = Any[FreeParam(:q1, init = 1.0, lower = 0.0) 0.0 0.0;
                   0.0 FreeParam(:q_lag, init = 1.0, lower = 0.0) 0.0;
                   0.0 0.0 FreeParam(:q2, init = 1.0, lower = 0.0)]

    # Observation, selection, initial state are identical and fixed.
    Z = [1.0 0.0 0.0;
         0.0 0.0 1.0]            # obs 1 ← state 1, obs 2 ← state 3
    R = Matrix(1.0I, 3, 3)       # one shock per state
    a1 = [0.0, 0.0, 0.0]
    P1 = Matrix(1e6I, 3, 3)

    spec_dsl = custom_ssm(Z = Z, H = H_dsl, T = T_dsl, R = R, Q = Q_dsl,
        a1 = a1, P1 = P1, name = :BlockDiagDSL)
    spec_manual = custom_ssm(Z = Z, H = H_manual, T = T_manual, R = R,
        Q = Q_manual, a1 = a1, P1 = P1, name = :BlockDiagManual)

    # The two specs should expose the same parameter list in the same order
    # with the same initial values and bounds — that's the precondition for
    # EM giving identical answers.
    @test param_names(spec_dsl) == param_names(spec_manual)
    @test initial_values(spec_dsl) == initial_values(spec_manual)
    @test param_bounds(spec_dsl) == param_bounds(spec_manual)

    # ---- Generate synthetic data ----
    n = 120
    Random.seed!(20260425)
    # Truth: AR(2) with φ = (0.6, -0.2), AR(1) with ρ = 0.7
    φ_true = (0.6, -0.2)
    ρ_true = 0.7
    σ_q1, σ_q2, σ_lag = sqrt(0.5), sqrt(0.3), sqrt(0.1)
    σ_h1, σ_h2 = sqrt(0.4), sqrt(0.2)

    α = zeros(3, n)
    for t in 2:n
        α[1, t] = φ_true[1] * α[1, t - 1] + φ_true[2] * α[2, t - 1] + σ_q1 * randn()
        α[2, t] = α[1, t - 1] + σ_lag * randn()
        α[3, t] = ρ_true * α[3, t - 1] + σ_q2 * randn()
    end
    y = Matrix{Float64}(undef, 2, n)
    for t in 1:n
        y[1, t] = α[1, t] + σ_h1 * randn()
        y[2, t] = α[3, t] + σ_h2 * randn()
    end

    # ---- Run EM on both ----
    model_dsl = StateSpaceModel(spec_dsl, n)
    model_manual = StateSpaceModel(spec_manual, n)

    fit!(EM(), model_dsl, y; maxiter = 60, verbose = false, warn = false)
    fit!(EM(), model_manual, y; maxiter = 60, verbose = false, warn = false)

    # ---- Compare ----
    # EM is deterministic given the same starting values; equivalent specs
    # should match to machine precision.
    @test loglikelihood(model_dsl) ≈ loglikelihood(model_manual) atol = 1e-10
    @test model_dsl.theta_values ≈ model_manual.theta_values atol = 1e-10
    # Loose check against the truth for the AR(1) block, which is well
    # identified; the AR(2) block with a shock on its lag state is not at n = 120.
    @test isapprox(parameters(model_dsl).ρ, ρ_true, atol = 0.30)
end

# ============================================
# In-place EM honours spec-fixed elements
# ============================================

@testset "fit!(EM) preserves spec-fixed Z and T" begin
    # local_level fixes Z = 1, T = 1 structurally (only var_obs and var_level
    # are free). Before mask propagation was wired up, the in-place EM would
    # update Z and T as if they were free, drifting away from the unit values
    # while still hitting a similar log-likelihood. Verify the fix.
    Random.seed!(20260425)
    n = 200
    states = zeros(n)
    y = zeros(1, n)
    for t in 2:n
        states[t] = states[t - 1] + 5 * randn()
    end
    for t in 1:n
        y[1, t] = states[t] + 10 * randn()
    end

    spec = local_level()
    model = StateSpaceModel(spec, n)
    fit!(EM(), model, y; maxiter = 100, tol = 1e-8)

    # The workspace's Z and T must remain at their spec-fixed values.
    kf_ws = model.workspaces.kf_ws
    @test kf_ws.Z[1, 1] ≈ 1.0 atol = 0
    @test kf_ws.Tmat[1, 1] ≈ 1.0 atol = 0

    # And EM should still find a sensible likelihood.
    @test isfinite(loglikelihood(model))
    @test loglikelihood(model) > -10_000  # sanity bound
end

@testset "fit!(EM) preserves zero columns in DFM-style Z" begin
    # Custom 3-state, 2-obs spec where the second column of Z is structurally
    # zero. The mask must keep that column at zero across EM iterations.
    Random.seed!(20260425)
    n = 150
    Z = Any[FreeParam(:λ_1, init = 0.5) 0.0 FreeParam(:λ_3, init = 0.5);
            FreeParam(:λ_4, init = 0.5) 0.0 FreeParam(:λ_6, init = 0.5)]
    T = [0.7 0.0 0.0;
         0.0 0.5 0.0;
         0.0 0.0 0.6]
    R = Matrix(I, 3, 3)
    Q = Any[FreeParam(:q1, init = 1.0, lower = 0.0) 0.0 0.0;
            0.0 FreeParam(:q2, init = 1.0, lower = 0.0) 0.0;
            0.0 0.0 FreeParam(:q3, init = 1.0, lower = 0.0)]
    H = Any[FreeParam(:h1, init = 1.0, lower = 0.0) 0.0;
            0.0 FreeParam(:h2, init = 1.0, lower = 0.0)]
    spec = custom_ssm(Z = Z, H = H, T = T, R = R, Q = Q,
        a1 = zeros(3), P1 = Matrix(I, 3, 3) * 10.0,
        name = :ZeroColumnZ)
    y = randn(2, n)

    model = StateSpaceModel(spec, n)
    fit!(EM(), model, y; maxiter = 30, tol = 0.0)

    kf_ws = model.workspaces.kf_ws
    @test kf_ws.Z[1, 2] == 0.0   # column 2 of Z stays zero
    @test kf_ws.Z[2, 2] == 0.0
    # And the structurally-fixed T elements stay put.
    @test kf_ws.Tmat[1, 1] ≈ 0.7
    @test kf_ws.Tmat[2, 2] ≈ 0.5
    @test kf_ws.Tmat[3, 3] ≈ 0.6
end

# ============================================
# static knob
# ============================================

@testset "fit!(MLE) static and dynamic paths agree" begin
    Random.seed!(20260425)
    n = 80
    y = reshape(cumsum(randn(n)) .+ randn(n), 1, n)
    spec = local_level()

    m_static = StateSpaceModel(spec, n)
    fit!(MLE(), m_static, y)
    m_dyn = StateSpaceModel(spec, n)
    fit!(MLE(), m_dyn, y; static = :off)
    m_on = StateSpaceModel(spec, n)
    fit!(MLE(), m_on, y; static = :on)
    @test loglikelihood(m_dyn) ≈ loglikelihood(m_static) atol = 1e-4
    @test loglikelihood(m_on) ≈ loglikelihood(m_static)
    @test filtered_states(m_dyn) ≈ filtered_states(m_static) rtol = 1e-3

    θ = parameters(m_static)
    m = StateSpaceModel(spec, θ, n)
    @test kalman_loglik(m, y) ≈ kalman_loglik(m, y; static = :off)
    @test kalman_filter!(m, y; static = :off) ≈ kalman_filter!(m, y)
end

@testset "fit!(MLE) static knob validation" begin
    spec = local_level()
    n = 50
    y = randn(1, n)
    model = StateSpaceModel(spec, n)
    @test_throws "static must be :auto, :on or :off" fit!(
        MLE(), model, y; static = :nonsense)
end

# ============================================
# EM updates a1 / P1 only when spec marks them free
# ============================================

@testset "fit!(EM) leaves spec-fixed (a1, P1) untouched" begin
    # local_level uses FixedValue everywhere in a1 / P1, so EM must not
    # change them — even after the M-step now calls update_initial_state!.
    Random.seed!(20260425)
    n = 200
    y = zeros(1, n)
    states = zeros(n)
    for t in 2:n
        states[t] = states[t - 1] + 5 * randn()
    end
    for t in 1:n
        y[1, t] = states[t] + 10 * randn()
    end

    spec = local_level()                 # diffuse=true → P1 = 1e7
    model = StateSpaceModel(spec, n)
    fit!(EM(), model, y; maxiter = 50, tol = 1e-8)

    kf_ws = model.workspaces.kf_ws
    @test kf_ws.a1[1] == 0.0
    @test kf_ws.P1[1, 1] == 1e7
end

@testset "fit!(EM) estimates FreeParam a1 / P1" begin
    # Local-level model where the user marks both the initial mean and
    # variance as free. EM should converge and produce values consistent
    # with the smoothed t=1 distribution.
    Random.seed!(20260425)
    n = 250
    states = zeros(n)
    states[1] = 5.0
    for t in 2:n
        states[t] = states[t - 1] + 1.0 * randn()
    end
    y = Matrix{Float64}(undef, 1, n)
    for t in 1:n
        y[1, t] = states[t] + 2.0 * randn()
    end

    spec = custom_ssm(
        Z = [1.0;;],
        H = [FreeParam(:var_obs, init = 4.0, lower = 0.0);;],
        T = [1.0;;], R = [1.0;;],
        Q = [FreeParam(:var_level, init = 1.0, lower = 0.0);;],
        a1 = [FreeParam(:mu0, init = 0.0)],
        P1 = [FreeParam(:p0, init = 1.0, lower = 0.0);;]
    )
    model = StateSpaceModel(spec, n)
    fit!(EM(), model, y; maxiter = 200, tol = 1e-10)

    nm = param_names(spec)
    θ = parameters(model)
    @test isfinite(loglikelihood(model))
    @test :mu0 in nm && :p0 in nm

    # mu0 is the smoothed mean at t=1; with a long enough series it should
    # be in the neighbourhood of the true initial state (5.0). Loose check
    # because p0 → small inflates posterior precision.
    @test abs(θ.mu0 - 5.0) < 3.0
    @test θ.p0 ≥ 0.0   # variance non-negative

    # The fitted (a1, P1) in the workspace match the parameters that
    # fit!(EM()) stores on the model.
    kf_ws = model.workspaces.kf_ws
    @test kf_ws.a1[1] ≈ θ.mu0
    @test kf_ws.P1[1, 1] ≈ θ.p0

    # At an exact EM fixed point the closed-form M-step gives
    # a1 = αs[:, 1] and P1 = Vs[:, :, 1]; with finite maxiter the two are
    # close but not bit-equal because em_estimate! refreshes the smoother
    # after the last M-step. Verify they're close in relative terms.
    @test kf_ws.a1[1] ≈ kf_ws.αs[1, 1] rtol = 1e-3
    @test kf_ws.P1[1, 1] ≈ kf_ws.Vs[1, 1, 1] rtol = 1e-2
end

@testset "fit!(EM) preserves loglik monotonicity with free a1 / P1" begin
    # Closed-form M-step including a1/P1 should still produce a non-
    # decreasing loglik trajectory.
    Random.seed!(123)
    n = 80
    y = reshape(cumsum(0.5 .* randn(n)) .+ 0.3 .* randn(n), 1, n)

    spec = custom_ssm(
        Z = [1.0;;],
        H = [FreeParam(:h, init = 0.5, lower = 0.0);;],
        T = [1.0;;], R = [1.0;;],
        Q = [FreeParam(:q, init = 0.5, lower = 0.0);;],
        a1 = [FreeParam(:mu0, init = 0.0)],
        P1 = [FreeParam(:p0, init = 1.0, lower = 0.0);;]
    )

    # Drive em_estimate! directly to capture the loglik history.
    theta_init = [p.init for p in spec.params]
    names_t = Tuple(p.name for p in spec.params)
    theta_nt = NamedTuple{names_t}(Tuple(theta_init))
    kfp = Siphon.build_kfparms(spec, theta_nt)
    a1_init, P1_init = Siphon.build_initial_state(spec, theta_nt)
    kf_ws = KalmanWorkspace(kfp.Z, kfp.H, kfp.T, kfp.R, kfp.Q, a1_init, P1_init, n)
    em_ws = Siphon.EMWorkspace(kf_ws)
    Siphon.set_em_masks!(em_ws, spec)
    result = Siphon.em_estimate!(kf_ws, em_ws, y; maxiter = 50, tol = 0.0, verbose = false)

    @test all(diff(result.loglik_history) .>= -1e-8)
end

@testset "fit!(EM) partial-free P1 stays symmetric" begin
    # 2-state model with one FreeParam in P1[1,1] and FixedValues elsewhere.
    # The update should respect the fixed entries.
    Random.seed!(0)
    n = 100
    α = zeros(2, n)
    for t in 2:n
        α[1, t] = α[1, t - 1] + 0.5 * randn()
        α[2, t] = α[2, t - 1] + 0.5 * randn()
    end
    y = α[1:1, :] + 0.5 .* randn(1, n)

    spec = custom_ssm(
        Z = [1.0 0.0],
        H = [FreeParam(:h, init = 0.25, lower = 0.0);;],
        T = [1.0 0.0; 0.0 1.0],
        R = Matrix(I, 2, 2),
        Q = [FreeParam(:q1, init = 0.25, lower = 0.0) 0.0;
             0.0 FreeParam(:q2, init = 0.25, lower = 0.0)],
        # P1[1,1] is free; off-diagonals and P1[2,2] are fixed.
        a1 = [0.0, 0.0],
        P1 = [FreeParam(:p11, init = 1.0, lower = 0.0) 0.0;
              0.0 10.0]
    )
    model = StateSpaceModel(spec, n)
    fit!(EM(), model, y; maxiter = 30, tol = 0.0)

    kf_ws = model.workspaces.kf_ws
    # Fixed cells unchanged
    @test kf_ws.P1[2, 2] == 10.0
    @test kf_ws.P1[1, 2] == 0.0
    @test kf_ws.P1[2, 1] == 0.0
    # Free cell estimated (positive, finite)
    @test kf_ws.P1[1, 1] > 0
    @test isfinite(kf_ws.P1[1, 1])
end

@testset "Cross-lag covariances with a singular predicted covariance" begin
    # AR(2) in companion form with H = 0: the first state is observed exactly, so
    # P_{t|t-1} is singular from the second period on.
    Z = [1.0 0.0]
    H = [0.0;;]
    Tm = [0.6 1.0; -0.15 0.0]
    R = [1.0; 0.0;;]
    Q = [1.0;;]
    a1 = zeros(2)
    P1 = [1.6 -0.1; -0.1 0.03]
    rng = MersenneTwister(7)
    n = 80
    y = zeros(1, n)
    for t in 3:n
        y[1, t] = 0.6 * y[1, t - 1] - 0.15 * y[1, t - 2] + randn(rng)
    end

    ws = KalmanWorkspace(Z, H, Tm, R, Q, a1, P1, n)
    kalman_filter!(ws, y)
    kalman_smoother!(ws; crosscov = true)
    Pc_ws = crosslag_covs(ws)
    @test all(isfinite, Pc_ws)

    sm = kalman_smoother(kalman_filter(KFParms(Z, H, Tm, R, Q), y, a1, P1);
        crosscov = true)
    @test crosslag_covs(sm) ≈ Pc_ws atol = 1e-10

    # The first state equals y exactly, so its cross-lag covariance with any
    # state is zero.
    @test all(abs.(Pc_ws[1, :, :]) .< 1e-10)
    @test all(abs.(Pc_ws[:, 1, :]) .< 1e-10)
end

@testset "fit!(EM) on autoregressive templates" begin
    rng = MersenneTwister(1)
    n = 300
    y = zeros(1, n)
    for t in 3:n
        y[1, t] = 0.6 * y[1, t - 1] - 0.15 * y[1, t - 2] + randn(rng)
    end

    # φ2 sits in a row of T whose state has no shock; EM would leave it at its
    # starting value.
    @test_throws "cannot estimate φ2 (it enters row 2 of T, whose state has no shock" fit!(
        EM(), StateSpaceModel(arma(2, 0), n), y)

    m_em = fit!(EM(), StateSpaceModel(arma(1, 0), n), y; maxiter = 2000)
    m_mle = fit!(MLE(), StateSpaceModel(arma(1, 0), n), y)
    @test isconverged(m_em)
    @test parameters(m_em).φ1 ≈ parameters(m_mle).φ1 rtol = 1e-6
    @test loglikelihood(m_em) ≈ loglikelihood(m_mle) rtol = 1e-8
end
