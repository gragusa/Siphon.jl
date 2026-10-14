# Disturbance smoother: functional, in-place and model APIs, checked against the
# conditional moments of the joint Gaussian distribution of (α₁, η, ε) given the
# observed values of y.

using Siphon
using Test
using LinearAlgebra
using Random
using ForwardDiff

# E[ε | y], E[η | y] and their per-period covariances by direct Gaussian conditioning
# on x = (α₁, η₁, …, ηₙ, ε₁, …, εₙ).
function disturbance_oracle(Z, H, T, R, Q, a1, P1, y)
    p, m = size(Z)
    r = size(R, 2)
    n = size(y, 2)
    iη(t) = m + (t - 1) * r .+ (1:r)
    iε(t) = m + r * n + (t - 1) * p .+ (1:p)
    nx = m + r * n + p * n
    μ = zeros(nx)
    μ[1:m] = a1
    Σ = zeros(nx, nx)
    Σ[1:m, 1:m] = P1
    for t in 1:n
        Σ[iη(t), iη(t)] = Q
        Σ[iε(t), iε(t)] = H
    end
    # αₜ as a linear map of x
    Aα = zeros(m, nx)
    Aα[:, 1:m] = I(m)
    rows = Vector{Float64}[]
    yo = Float64[]
    for t in 1:n
        t > 1 && (Aα = T * Aα; Aα[:, iη(t - 1)] .+= R)
        for i in 1:p
            isnan(y[i, t]) && continue
            a = vec(Z[i:i, :] * Aα)
            a[iε(t)[i]] += 1
            push!(rows, a)
            push!(yo, y[i, t])
        end
    end
    A = reduce(vcat, permutedims.(rows))
    G = Σ * A' / (A * Σ * A')
    mx = μ + G * (yo - A * μ)
    Vx = Σ - G * A * Σ
    εs = reduce(hcat, [mx[iε(t)] for t in 1:n])
    ηs = reduce(hcat, [mx[iη(t)] for t in 1:n])
    Vε = cat([Vx[iε(t), iε(t)] for t in 1:n]...; dims = 3)
    Vη = cat([Vx[iη(t), iη(t)] for t in 1:n]...; dims = 3)
    return εs, ηs, Vε, Vη
end

# Three series with a full H, two states driven by one shock, and a period with one
# value missing, one with all missing and one with two missing.
function disturbance_system()
    rng = Xoshiro(1)
    p, m, n = 3, 2, 12
    Z = randn(rng, p, m)
    B = randn(rng, p, p)
    H = B * B' / 3 + 0.2I
    T = [0.7 0.2; -0.1 0.5]
    R = reshape([1.0, 0.5], 2, 1)
    Q = fill(0.8, 1, 1)
    a1 = [0.3, -0.2]
    P1 = [1.0 0.2; 0.2 0.7]
    y = randn(rng, p, n)
    y[2, 3] = NaN
    y[:, 5] .= NaN
    y[[1, 3], 8] .= NaN
    return Z, H, T, R, Q, a1, P1, y
end

@testset "Disturbance smoother" begin
    Z, H, T, R, Q, a1, P1, y = disturbance_system()
    p, n = size(y)
    εo, ηo, Vεo, Vηo = disturbance_oracle(Z, H, T, R, Q, a1, P1, y)
    kp = KFParms(Z, H, T, R, Q)

    @testset "functional matches Gaussian conditioning" begin
        ds = disturbance_smoother(kp, y, a1, P1)
        @test ds isa DisturbanceSmootherResult{Float64}
        @test smoothed_obs_disturbances(ds) ≈ εo atol = 1e-12
        @test smoothed_state_disturbances(ds) ≈ ηo atol = 1e-12
        @test smoothed_obs_disturbances_cov(ds) ≈ Vεo atol = 1e-12
        @test smoothed_state_disturbances_cov(ds) ≈ Vηo atol = 1e-12
        @test disturbance_smoother(kalman_filter(kp, y, a1, P1)).εs == ds.εs

        # No information on εₜ in an unobserved period, nor on ηₙ.
        @test all(iszero, ds.εs[:, 5])
        @test ds.Vε[:, :, 5] ≈ H
        @test all(iszero, ds.ηs[:, n])
        @test ds.Vη[:, :, n] ≈ Q

        aux = auxiliary_residuals(ds)
        @test all(isnan, aux.obs[:, 5])
        @test all(isnan, aux.state[:, n])
        # Var[ε̂ₜ] = H − Var[εₜ | y]
        @test aux.obs[1, 1] ≈ ds.εs[1, 1] / sqrt(H[1, 1] - ds.Vε[1, 1, 1])
        @test aux.state[1, 1] ≈ ds.ηs[1, 1] / sqrt(Q[1, 1] - ds.Vη[1, 1, 1])
        @test sprint(show, ds) == "DisturbanceSmootherResult(p = 3, r = 1, n = 12)"
    end

    @testset "in-place matches functional" begin
        ds = disturbance_smoother(kp, y, a1, P1)
        ws = KalmanWorkspace(Z, H, T, R, Q, a1, P1, n)
        kalman_filter!(ws, y)
        dsw = disturbance_smoother(ws)
        for f in fieldnames(DisturbanceSmootherResult)
            @test isequal(isnan.(getfield(dsw, f)), isnan.(getfield(ds, f)))
            @test filter(!isnan, getfield(dsw, f)) ≈ filter(!isnan, getfield(ds, f)) atol = 1e-12
        end

        # Reuses `out` with an allocation count that does not grow with n.
        out = DisturbanceSmootherResult{Float64}(p, 1, n)
        @test disturbance_smoother!(out, ws) === out
        @test out.εs == dsw.εs
        a_short = @allocated disturbance_smoother!(out, ws)
        y_long = repeat(y, 1, 5)
        ws_long = KalmanWorkspace(Z, H, T, R, Q, a1, P1, 5n)
        kalman_filter!(ws_long, y_long)
        out_long = DisturbanceSmootherResult{Float64}(p, 1, 5n)
        disturbance_smoother!(out_long, ws_long)
        @test (@allocated disturbance_smoother!(out_long, ws_long)) == a_short

        # The smoother outputs of the workspace are untouched.
        kalman_smoother!(ws)
        αs = copy(smoothed_states(ws))
        disturbance_smoother!(out, ws)
        @test smoothed_states(ws) == αs
    end

    @testset "singular innovation covariance (rank-revealing)" begin
        # Rows 1 and 2 are the same exact measurement of the first state.
        Z2 = [1.0 0.0; 1.0 0.0; 0.0 1.0]
        H2 = Matrix(Diagonal([0.0, 0.0, 0.5]))
        y2 = randn(Xoshiro(2), 3, n)
        y2[2, :] = y2[1, :]
        ws = KalmanWorkspace(Z2, H2, T, R, Q, a1, P1, n)
        set_rank_revealing!(ws, true)
        kalman_filter!(ws, y2)
        ds = disturbance_smoother(ws)
        ε2, η2, _, Vη2 = disturbance_oracle(Z2[[1, 3], :], H2[[1, 3], [1, 3]],
            T, R, Q, a1, P1, y2[[1, 3], :])
        @test ds.ηs ≈ η2 atol = 1e-12
        @test ds.Vη ≈ Vη2 atol = 1e-12
        @test ds.εs[3, :] ≈ ε2[2, :] atol = 1e-12
        @test all(iszero, ds.εs[1:2, :])
    end

    @testset "scalar system and AD" begin
        yl = cumsum(randn(Xoshiro(3), 1, 30); dims = 2)
        yl[1, 7] = NaN
        ds_s = disturbance_smoother(KFParms(1.0, 2.0, 1.0, 1.0, 0.5), vec(yl), 0.0, 10.0)
        ds_m = disturbance_smoother(
            KFParms(fill(1.0, 1, 1), fill(2.0, 1, 1),
                fill(1.0, 1, 1), fill(1.0, 1, 1), fill(0.5, 1, 1)),
            yl,
            [0.0],
            fill(10.0, 1, 1))
        @test ds_s.εs ≈ ds_m.εs
        @test ds_s.Vη ≈ ds_m.Vη

        f(θ) = sum(disturbance_smoother(KFParms(Z, θ[1] * H, T, R, θ[2] * Q), y, a1, P1).εs)
        g = ForwardDiff.gradient(f, [1.0, 1.0])
        h = 1e-6
        @test g[1] ≈ (f([1 + h, 1.0]) - f([1 - h, 1.0])) / 2h rtol = 1e-6
        @test g[2] ≈ (f([1.0, 1 + h]) - f([1.0, 1 - h])) / 2h rtol = 1e-6
    end

    @testset "StateSpaceModel" begin
        yl = cumsum(randn(Xoshiro(4), 1, 40); dims = 2) .+ randn(Xoshiro(5), 1, 40)
        yl[1, 10] = NaN
        model = StateSpaceModel(local_level(), (var_obs = 1.0, var_level = 0.5), 40)
        @test_throws "kalman_filter!(model, y)" disturbance_smoother(model)
        kalman_filter!(model, yl)
        ds = disturbance_smoother(model)
        αs = smoothed_states(model)
        # ε̂ₜ = yₜ − α̂ₜ where observed, and η̂ₜ = α̂ₜ₊₁ − α̂ₜ for the local level.
        obs = .!isnan.(yl)
        @test ds.εs[obs] ≈ (yl - αs)[obs] rtol = 1e-8
        @test ds.ηs[:, 1:(end - 1)] ≈ diff(αs; dims = 2) rtol = 1e-8

        exact = StateSpaceModel(local_level(; diffuse = :exact),
            (var_obs = 1.0, var_level = 0.5), 40)
        kalman_filter!(exact, yl)
        @test_throws "diffuse = :exact" disturbance_smoother(exact)
    end

    @testset "errors" begin
        ws = KalmanWorkspace(Z, H, T, R, Q, a1, P1, n)
        @test_throws "kalman_filter!(ws, y)" disturbance_smoother(ws)
        kalman_filter!(ws, y)
        @test_throws DimensionMismatch disturbance_smoother!(
            DisturbanceSmootherResult{Float64}(2, 1, n), ws)
        diffuse = kalman_filter(kp, y, a1, zeros(2, 2), Matrix(1.0I, 2, 2))
        @test_throws "exact diffuse" disturbance_smoother(diffuse)
        dws = DiffuseKalmanWorkspace(Z, H, T, R, Q, a1, zeros(2, 2), Matrix(1.0I, 2, 2), n)
        @test_throws "exact diffuse" disturbance_smoother(dws)
    end
end
