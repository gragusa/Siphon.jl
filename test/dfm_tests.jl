"""
Tests for DynamicFactorModel: constructor validation, EM recovery of the
idiosyncratic variances, loadings and AR error coefficients, monotonicity of the
EM log-likelihood, and forecasting with AR errors.
"""

using Test
using Siphon
using LinearAlgebra
using Random

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
        Random.seed!(3)
        model = DynamicFactorModel(size(y, 1), 1, size(y, 2))
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
        Random.seed!(3)
        model = DynamicFactorModel(size(y, 1), 1, size(y, 2); factor_lags = 2)
        fit!(EM(), model, y; maxiter = 300, tol = 1e-12)
        @test _monotone(model.loglik_history)
        @test all(iszero, model.kf_ws.Z[:, 2])
        @test all(abs.(idiosyncratic_variances(model) .- DFM_σ²) .<= 0.15 .* DFM_σ² .+ 0.02)
        @test _abscor(vec(factors(model)), f) > 0.95
    end

    @testset "EM with AR(1) errors recovers loadings, δ and σ²" begin
        y, f = _simulate_dfm(; λ = DFM_λ, σ² = DFM_σ², δ = 0.6, n = 2000)
        Random.seed!(3)
        model = DynamicFactorModel(size(y, 1), 1, size(y, 2); error_lags = 1)
        fit!(EM(), model, y; maxiter = 1000)
        @test isconverged(model)
        @test _monotone(model.loglik_history)

        Λ = vec(loadings(model)[1])
        @test maximum(abs.(Λ ./ Λ[1] .- DFM_λ ./ DFM_λ[1])) < 0.1
        @test abs(ar_coefficients(model)[1] - 0.6) < 0.05
        @test all(abs.(idiosyncratic_variances(model) .- DFM_σ²) .<= 0.15 .* DFM_σ² .+ 0.03)
        @test _abscor(vec(factors(model)), f) > 0.95

        # The observation equation is quasi-differenced: Z = Λ₀ (1 - δL) on the
        # factor state [fₜ; fₜ₋₁], and H holds the innovation variances.
        δ = ar_coefficients(model)[1]
        @test model.kf_ws.Z ≈ [Λ -δ .* Λ]
        @test diag(model.kf_ws.H) ≈ idiosyncratic_variances(model)
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
        Random.seed!(3)
        model = DynamicFactorModel(N, 1, n; error_lags = 2, loading_lags = 1)
        fit!(EM(), model, y; maxiter = 2000)
        @test _monotone(model.loglik_history)
        @test ar_coefficients(model) ≈ [0.5, 0.2] atol = 0.05
        L = loadings(model)
        s = L[1][1]
        @test vec(L[1]) ./ s ≈ DFM_λ ./ DFM_λ[1] atol = 0.1
        @test vec(L[2]) ./ s ≈ λ1 ./ DFM_λ[1] atol = 0.1
    end

    @testset "AR errors with a missing period" begin
        y, _ = _simulate_dfm(; λ = DFM_λ, σ² = DFM_σ², δ = 0.6, n = 400)
        y[:, 100:102] .= NaN
        Random.seed!(3)
        model = DynamicFactorModel(size(y, 1), 1, size(y, 2); error_lags = 1)
        fit!(EM(), model, y; maxiter = 300)
        @test _monotone(model.loglik_history)
        @test isfinite(loglikelihood(model))
        @test abs(ar_coefficients(model)[1] - 0.6) < 0.1
    end

    @testset "forecast with AR errors" begin
        y, _ = _simulate_dfm(; λ = DFM_λ, σ² = DFM_σ², δ = 0.6, n = 300)
        Random.seed!(3)
        model = DynamicFactorModel(size(y, 1), 1, size(y, 2); error_lags = 1)
        fit!(EM(), model, y; maxiter = 300)
        kf, n = model.kf_ws, size(y, 2)
        δ = ar_coefficients(model)[1]
        Z, Tm = kf.Z, kf.Tmat
        fc = forecast(model, 2)

        # Xₙ₊₁ = δ Xₙ + Z αₙ₊₁ + vₙ₊₁ and Xₙ₊₂ = δ Xₙ₊₁ + Z αₙ₊₂ + vₙ₊₂.
        a1 = Tm * kf.att[:, n]
        P1 = Tm * kf.Ptt[:, :, n] * Tm' + kf.RQR
        @test fc isa ForecastResult
        @test forecast_periods(fc) == (n + 1):(n + 2)
        @test forecast_observations(fc)[:, 1] ≈ δ .* y[:, n] .+ Z * a1
        @test forecast_observations_cov(fc)[:, :, 1] ≈ Z * P1 * Z' + kf.H
        @test forecast_observations(fc)[:, 2] ≈
              δ .* forecast_observations(fc)[:, 1] .+ Z * Tm * a1
        @test forecast_states(fc)[:, 1] ≈ a1

        y[:, end] .= NaN
        Random.seed!(3)
        model = DynamicFactorModel(size(y, 1), 1, size(y, 2); error_lags = 1)
        fit!(EM(), model, y; maxiter = 5)
        @test_throws ArgumentError forecast(model, 1)
    end
end

@testset "update_Z! restricted M-step" begin
    # With some Z cells fixed, each row solves
    # Z[i, J] = (S_yα[i, J] - Z[i, F] S_αα[F, J]) / S_αα[J, J].
    rng = MersenneTwister(11)
    p, m, n = 3, 4, 50
    kf = KalmanWorkspace(randn(rng, p, m), Matrix(1.0I, p, p), Matrix(0.5I, m, m),
        Matrix(1.0I, m, m), Matrix(1.0I, m, m), zeros(m), Matrix(1.0I, m, m), n)
    em = EMWorkspace(kf)
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
