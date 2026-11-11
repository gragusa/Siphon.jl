"""
Tests for DynamicFactorModel: constructor validation, EM recovery of the
idiosyncratic variances, loadings and AR error coefficients, monotonicity of the
EM log-likelihood, and forecasting with AR errors.
"""

using Test
using Siphon
using LinearAlgebra
using Random
using ForwardDiff

include("dfm_dense_helpers.jl")

# One-factor DFM with AR(q) factor and AR(1) idiosyncratic errors (δ = 0 gives
# white noise errors).
function _simulate_dfm(; λ, σ², φ = (0.7,), δ = 0.0, n = 500, seed = 1)
    rng = MersenneTwister(seed)
    N = length(λ)
    f = zeros(n)
    e = zeros(N, n)
    for t in 1:n
        f[t] = sum(φ[j] * (t > j ? f[t - j] : 0.0) for j in eachindex(φ)) + randn(rng)
        e[:, t] .= (t > 1 ? δ .* e[:, t - 1] : 0.0) .+ sqrt.(σ²) .* randn(rng, N)
    end
    return λ * f' .+ e, f
end

function _abscor(a, b)
    abs(dot(a .- sum(a) / length(a), b .- sum(b) / length(b))) /
    (norm(a .- sum(a) / length(a)) * norm(b .- sum(b) / length(b)))
end

_monotone(h) = all(diff(h) .>= -1e-6 * abs(h[end]))

const DFM_λ = [1.0, 0.8, 0.6, 1.2, 0.9, 0.7]
const DFM_σ² = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]

@testset "DynamicFactorModel" begin
    @testset "constructor validation" begin
        @test_throws ArgumentError DynamicFactorModel(0, 1, 100)
        @test_throws ArgumentError DynamicFactorModel(5, 0, 100)
        @test_throws ArgumentError DynamicFactorModel(5, 1, 1)
        @test_throws ArgumentError DynamicFactorModel(5, 1, 100; loading_lags = -1)
        @test_throws ArgumentError DynamicFactorModel(5, 1, 100; factor_lags = -1)
        @test_throws ArgumentError DynamicFactorModel(5, 1, 100; error_lags = -1)
        @test DynamicFactorModel(5, 1, 100; factor_lags = 0) isa DynamicFactorModel
    end

    @testset "EM recovers heterogeneous idiosyncratic variances" begin
        y, f = _simulate_dfm(; λ = DFM_λ, σ² = DFM_σ²)
        model = DynamicFactorModel(size(y, 1), 1, size(y, 2); rng = Xoshiro(3))
        fit!(EM(), model, y; maxiter = 1000)

        σ²_hat = idiosyncratic_variances(model)
        @test all(abs.(σ²_hat .- DFM_σ²) .<= 0.15 .* DFM_σ² .+ 0.02)
        Λ = loadings(model)[1]
        @test maximum(abs.(vec(Λ) ./ Λ[1] .- DFM_λ ./ DFM_λ[1])) < 0.2
        @test _abscor(vec(factors(model)), f) > 0.95
        @test _monotone(model.loglik_history)
        @test isfinite(loglikelihood(model))
    end

    @testset "EM is monotone with fixed lagged-factor loadings" begin
        # factor_lags = 2 puts fₜ₋₁ in the state with its loadings fixed at 0,
        # so the loadings update must be the restricted M-step.
        y, f = _simulate_dfm(; λ = DFM_λ, σ² = DFM_σ², φ = (0.5, 0.3))
        model = DynamicFactorModel(
            size(y, 1), 1, size(y, 2); factor_lags = 2, rng = Xoshiro(3))
        fit!(EM(), model, y; maxiter = 300, tol = 1e-12)
        @test _monotone(model.loglik_history)
        @test all(iszero, model.kf_ws.Z[:, 2])
        @test all(abs.(idiosyncratic_variances(model) .- DFM_σ²) .<= 0.15 .* DFM_σ² .+ 0.02)
        @test _abscor(vec(factors(model)), f) > 0.95
    end

    @testset "EM with AR(1) errors recovers loadings, δ and σ²" begin
        y, f = _simulate_dfm(; λ = DFM_λ, σ² = DFM_σ², δ = 0.6, n = 2000)
        model = DynamicFactorModel(
            size(y, 1), 1, size(y, 2); error_lags = 1, rng = Xoshiro(3))
        fit!(EM(), model, y; maxiter = 1000)
        @test isconverged(model)
        @test _monotone(model.loglik_history)

        Λ = vec(loadings(model)[1])
        @test maximum(abs.(Λ ./ Λ[1] .- DFM_λ ./ DFM_λ[1])) < 0.1
        @test abs(ar_coefficients(model)[1] - 0.6) < 0.05
        @test all(abs.(idiosyncratic_variances(model) .- DFM_σ²) .<= 0.15 .* DFM_σ² .+ 0.03)
        @test _abscor(vec(factors(model)), f) > 0.95

        # The errors are states: Z = [Λ₀ 0 I] on [fₜ; fₜ₋₁; eₜ] with H = 0, the
        # error block of T and Q holds δ and σ², and the errors start stationary.
        δ = ar_coefficients(model)[1]
        σ² = idiosyncratic_variances(model)
        kf, N = model.kf_ws, length(Λ)
        @test kf.Z ≈ [Λ zeros(N) Matrix(1.0I, N, N)]
        @test iszero(kf.H)
        @test kf.Tmat[3:end, 3:end] ≈ δ * I
        @test kf.Q[2:end, 2:end] ≈ Diagonal(σ²)
        @test kf.P1[3:end, 3:end] ≈ Diagonal(σ² ./ (1 - δ^2))
    end

    @testset "EM with AR(2) errors and dynamic loadings" begin
        rng = MersenneTwister(4)
        n, N = 3000, length(DFM_λ)
        λ1 = [0.5, 0.0, -0.3, 0.2, 0.4, 0.0]
        f = zeros(n)
        e = zeros(N, n)
        for t in 3:n
            f[t] = 0.7 * f[t - 1] + randn(rng)
            e[:, t] .= 0.5 .* e[:, t - 1] .+ 0.2 .* e[:, t - 2] .+
                       sqrt.(DFM_σ²) .* randn(rng, N)
        end
        y = DFM_λ * f' .+ λ1 * circshift(f, 1)' .+ e
        model = DynamicFactorModel(
            N, 1, n; error_lags = 2, loading_lags = 1, rng = Xoshiro(3))
        fit!(EM(), model, y; maxiter = 2000)
        @test _monotone(model.loglik_history)
        @test ar_coefficients(model) ≈ [0.5, 0.2] atol = 0.05
        L = loadings(model)
        s = L[1][1]
        @test vec(L[1]) ./ s ≈ DFM_λ ./ DFM_λ[1] atol = 0.1
        @test vec(L[2]) ./ s ≈ λ1 ./ DFM_λ[1] atol = 0.1
    end

    @testset "AR errors with missing and partially observed periods" begin
        y, _ = _simulate_dfm(; λ = DFM_λ, σ² = DFM_σ², δ = 0.6, n = 400)
        y[:, 100:102] .= NaN
        y[2, 150:180] .= NaN
        y[5, 1:20] .= NaN
        model = DynamicFactorModel(
            size(y, 1), 1, size(y, 2); error_lags = 1, rng = Xoshiro(3))
        fit!(EM(), model, y; maxiter = 300, warn = false)
        @test _monotone(model.loglik_history)
        @test isfinite(loglikelihood(model))
        @test abs(ar_coefficients(model)[1] - 0.6) < 0.1
    end

    @testset "AR errors: exact likelihood and M-step against dense conditioning" begin
        rng = Xoshiro(21)
        N, n = 4, 40
        y = randn(rng, N, n) .+ cumsum(randn(rng, 1, n); dims = 2) .* [1.0, 0.5, -0.3, 0.8]
        y[2, 5:9] .= NaN   # partially observed periods
        y[4, 20] = NaN
        y[:, 30] .= NaN    # a fully missing period
        y[1, 1:2] .= NaN   # missing at the start
        model = DynamicFactorModel(N, 2, n; error_lags = 2, loading_lags = 1,
            factor_lags = 2, rng = Xoshiro(4))
        fit!(EM(), model, y; maxiter = 4, warn = false)
        @test _monotone(model.loglik_history)
        d = model.dfm_ws
        s = d.factor_dim ÷ 2
        θ0 = (deepcopy(d.Λ), deepcopy(d.Φ), copy(d.Σ_η), d.δ[1:2], copy(d.σ²_v))
        @test loglikelihood(model) ≈ dfm_loglik_dense(θ0..., y; s) rtol = 1e-12

        # The M-step for the errors, from the smoother at θ0, maximises the
        # expected complete-data log-likelihood block by block: each gradient,
        # computed from the dense posterior at θ0, vanishes at its update.
        post = dfm_posterior_dense(θ0, y; s)
        d1 = deepcopy(d)
        Siphon._update_dfm_ar!(d1, model.kf_ws)
        Qe(Λ, δ, σ²) = dfm_Q_errors(Λ, δ, σ², post; N, n)
        unvec(v) = [reshape(v[1:8], N, 2), reshape(v[9:16], N, 2)]
        Λ1 = vcat(vec(d1.Λ[1]), vec(d1.Λ[2]))
        @test norm(ForwardDiff.gradient(v -> Qe(unvec(v), θ0[4], θ0[5]), Λ1)) < 1e-8
        @test norm(ForwardDiff.gradient(v -> Qe(d1.Λ, v, θ0[5]), d1.δ[1:2])) < 1e-6
        @test norm(ForwardDiff.gradient(v -> Qe(d1.Λ, d1.δ[1:2], v), d1.σ²_v)) < 1e-8
    end

    @testset "forecast with AR errors" begin
        y, _ = _simulate_dfm(; λ = DFM_λ, σ² = DFM_σ², δ = 0.6, n = 300)
        model = DynamicFactorModel(
            size(y, 1), 1, size(y, 2); error_lags = 1, rng = Xoshiro(3))
        fit!(EM(), model, y; maxiter = 300)
        kf, (N, n) = model.kf_ws, size(y)
        δ = ar_coefficients(model)[1]
        Λ = vec(loadings(model)[1])
        Z, Tm = kf.Z, kf.Tmat
        fc = forecast(model, 2)

        # With H = 0 the filtered errors of an observed period are Xₙ - Λ fₙ|ₙ,
        # and their forecast decays at rate δ.
        a1 = Tm * kf.att[:, n]
        P1 = Tm * kf.Ptt[:, :, n] * Tm' + kf.RQR
        @test kf.att[3:end, n] ≈ y[:, n] .- Λ .* kf.att[1, n]
        @test a1[3:end] ≈ δ .* kf.att[3:end, n]
        @test fc isa ForecastResult
        @test forecast_periods(fc) == (n + 1):(n + 2)
        @test forecast_observations(fc)[:, 1] ≈ Z * a1
        @test forecast_observations_cov(fc)[:, :, 1] ≈ Z * P1 * Z'
        @test forecast_observations(fc)[:, 2] ≈ Z * Tm * a1
        @test forecast_states(fc)[:, 1] ≈ a1

        # A missing last period is forecast from the filtered errors.
        y[:, end] .= NaN
        model = DynamicFactorModel(N, 1, n; error_lags = 1, rng = Xoshiro(3))
        fit!(EM(), model, y; maxiter = 5, warn = false)
        @test all(isfinite, forecast_observations(forecast(model, 1)))
    end

    @testset "rng keyword" begin
        Z0(rng) = copy(DynamicFactorModel(6, 2, 50; rng).kf_ws.Z)
        @test Z0(Xoshiro(5)) == Z0(Xoshiro(5))
        @test Z0(Xoshiro(5)) != Z0(Xoshiro(6))
    end

    @testset "shared accessors" begin
        y, _ = _simulate_dfm(; λ = DFM_λ, σ² = DFM_σ², n = 200)
        N, n = size(y)
        model = DynamicFactorModel(N, 1, n; factor_lags = 2, rng = Xoshiro(3))

        for f in (filtered_states, smoothed_states, factors, factors_cov, filtered_factors,
            filtered_factors_cov, prediction_errors, system_matrices, initial_state,
            parameters, loadings, var_coefficients, factor_innovation_cov,
            ar_coefficients, idiosyncratic_variances)
            @test_throws "fit!(EM(), model, y)" f(model)
        end
        @test_throws "fit!(EM(), model, y)" forecast(model, 1)

        fit!(EM(), model, y; maxiter = 50, warn = false)
        @test model isa AbstractStateSpaceModel

        # The factors are the first k states.
        @test factors(model) == smoothed_states(model)[1:1, :]
        @test filtered_factors(model) == filtered_states(model)[1:1, :]
        @test factors_cov(model) == smoothed_states_cov(model)[1:1, 1:1, :]
        @test filtered_factors_cov(model) == filtered_states_cov(model)[1:1, 1:1, :]
        @test size(filtered_states(model)) == (2, n)
        @test size(crosslag_covs(model)) == (2, 2, n - 1)
        @test size(prediction_errors(model)) == (N, n)
        @test Siphon.observed_mask(model) == trues(N, n)

        # The system matrices hold the natural parameters and reproduce the loglik.
        p = system_matrices(model)
        @test state_cov(model) == factor_innovation_cov(model)
        @test obs_matrix(model)[:, 1] == loadings(model)[1][:, 1]
        @test transition_matrix(model)[1, :] == [var_coefficients(model)[1][1, 1],
            var_coefficients(model)[2][1, 1]]
        @test diag(obs_cov(model)) == idiosyncratic_variances(model)
        filt = kalman_filter(p, y, initial_state(model), initial_state_cov(model))
        @test loglikelihood(filt) ≈ loglikelihood(model)
        @test filtered_states(filt) ≈ filtered_states(model)

        θ = parameters(model)
        @test keys(θ) == (:loadings, :var_coefficients, :factor_innovation_cov,
            :ar_coefficients, :idiosyncratic_variances)
        @test θ.loadings == loadings(model)
        @test θ.factor_innovation_cov == factor_innovation_cov(model)

        # White-noise errors have no AR coefficients.
        @test ar_coefficients(model) == Float64[]

        # Parameter accessors return copies.
        loadings(model)[1] .= 0
        var_coefficients(model)[1] .= 0
        factor_innovation_cov(model) .= 0
        idiosyncratic_variances(model) .= 0
        system_matrices(model).Z .= 0
        @test loadings(model) == θ.loadings
        @test var_coefficients(model) == θ.var_coefficients
        @test factor_innovation_cov(model) == θ.factor_innovation_cov
        @test idiosyncratic_variances(model) == θ.idiosyncratic_variances
        @test obs_matrix(model) == p.Z
    end
end

@testset "dynamic_factor template keywords" begin
    spec = dynamic_factor(5, 2; factor_lags = 3, loading_lags = 1)
    @test spec.n_states == 6
    @test :λ1_1_1 in param_names(spec)
    @test_throws MethodError dynamic_factor(5, 2; obs_lags = 1)
end

@testset "update_Z! restricted M-step" begin
    # With some Z cells fixed, each row solves
    # Z[i, J] = (S_yα[i, J] - Z[i, F] S_αα[F, J]) / S_αα[J, J].
    rng = MersenneTwister(11)
    p, m, n = 3, 4, 50
    kf = KalmanWorkspace(randn(rng, p, m), Matrix(1.0I, p, p), Matrix(0.5I, m, m),
        Matrix(1.0I, m, m), Matrix(1.0I, m, m), zeros(m), Matrix(1.0I, m, m), n)
    em = Siphon.EMWorkspace(kf)
    A = randn(rng, m, 2m)
    em.S_αα .= A * A'
    em.S_yα .= randn(rng, p, m)
    fill!(em.Z_free, true)
    em.Z_free[1, 3:4] .= false      # rows 1 and 2 share a pattern
    em.Z_free[2, 3:4] .= false
    em.Z_free[3, 1] = false         # row 3 has its own
    Z0 = copy(kf.Z)

    Siphon.update_Z!(kf, em, n)

    for i in 1:p
        J = findall(em.Z_free[i, :])
        F = findall(.!em.Z_free[i, :])
        expected = (em.S_yα[i, J]' .- Z0[i, F]' * em.S_αα[F, J]) / em.S_αα[J, J]
        @test kf.Z[i, J] ≈ vec(expected) rtol = 1e-8
        @test kf.Z[i, F] == Z0[i, F]
    end

    # All free: the unrestricted solution S_yα / S_αα.
    fill!(em.Z_free, true)
    Siphon.update_Z!(kf, em, n)
    @test kf.Z ≈ em.S_yα / em.S_αα rtol = 1e-8
end
