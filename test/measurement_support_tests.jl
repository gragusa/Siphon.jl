using Test
using Siphon
using LinearAlgebra
using Random
using ForwardDiff

isdefined(@__MODULE__, :_ps_posterior) || include("period_scale_intercept_tests.jl")

function _ms_setup(::Type{T}) where {T}
    rng = Xoshiro(606)
    p, m, n = 3, 8, 8
    Tm = zeros(T, m, m)
    Tm[1:2, 1:4] .= [0.5 0.1 0.1 0.0; 0.0 0.4 0.05 0.1]
    Tm[3:m, 1:(m - 2)] .= Matrix{T}(I, m - 2, m - 2)
    Z = zeros(T, p, m)
    Z[1, 1] = one(T)
    Z[3, [2, 4]] .= T[0.75, 0.25]
    H = Matrix(Diagonal(T[0.3, 0.2, 0.1]))
    R = [Matrix{T}(I, 2, 2); zeros(T, m - 2, 2)]
    Q = T[1.0 0.2; 0.2 0.5]
    a1 = randn(rng, T, m)
    P1 = Matrix{T}(I, m, m)
    c = [T[0.1, -0.2]; zeros(T, m - 2)]
    s = T[1.0, 1.2, 0.8, 2.0, 1.5, 0.9, 1.0, 1.1]
    y = randn(rng, T, p, n)
    y[1, 3] = NaN
    y[2, 4] = NaN
    y[:, 5] .= NaN
    y[3, n] = NaN
    return (; Z, H, Tm, R, Q, a1, P1, c, s, y, n)
end

function _ms_workspace(d; Z = d.Z, H = d.H, covariances = false, block = 2)
    ws = KalmanWorkspace(Z, H, d.Tm, d.R, d.Q, d.a1, d.P1, d.n; covariances)
    set_companion_structure!(ws, block)
    Siphon.set_transition_intercept!(ws, d.c)
    Siphon.set_period_scales!(ws, d.s)
    return ws
end

@testset "measurement support — parameter updates match dense conditioning" begin
    for T in (Float32, Float64)
        d = _ms_setup(T)
        rng = Xoshiro(607)
        dense = randn(rng, T, size(d.Z)...)
        shifted = zeros(T, size(d.Z))
        shifted[1, 5] = T(1.2)
        shifted[2, [2, 6]] .= T[-0.5, 0.5]
        routes = ((:initial, d.Z), (:set_Z, dense), (:update, shifted),
            (:set_params, T(2) * shifted), (:direct, zeros(T, size(d.Z))),
            (:direct, d.Z), (:direct, dense), (:direct, shifted))
        tol = T === Float32 ? 1e-5 : 1e-10
        for covariances in (true, false), block in (0, 2)

            ws = _ms_workspace(d; covariances, block)
            for (route, Z) in routes
                if route == :set_Z
                    Siphon.set_Z!(ws, Z)
                elseif route == :update
                    update_params!(ws; Z)
                elseif route == :set_params
                    set_params!(ws, Z, d.H, d.Tm, d.R, d.Q)
                elseif route == :direct
                    ws.Z .= Z
                end
                ref = _ps_posterior(Z, d.H, d.Tm, d.R, d.Q, d.c, d.s, d.a1, d.P1, d.y)
                @test kalman_filter!(ws, view(d.y, :, :)) ≈ ref.loglik rtol = tol
                # Support storage is scratch; the smoother reconstructs it independently.
                fill!(ws.Z_cols, 0)
                fill!(ws.Z_nnz, 0)
                kalman_smoother!(ws)
                @test vec(smoothed_states(ws))≈ref.mean rtol=tol atol=tol
                if covariances
                    V = smoothed_states_cov(ws)
                    for t in axes(V, 3)
                        idx = _ps_block(t, size(Z, 2))
                        @test V[:, :, t]≈ref.cov[idx, idx] rtol=tol atol=tol
                    end
                end
                @test (@allocated kalman_filter!(ws, d.y)) == 0
                @test (@allocated kalman_smoother!(ws)) == 0
            end
        end
    end
end

@testset "measurement support — small nonzero loadings are retained" begin
    for T in (Float32, Float64)
        d = _ms_setup(T)
        Z = copy(d.Z)
        Z[3, 8] = T(1e-20)
        ws = _ms_workspace(d; Z)
        @test Siphon._refresh_measurement_support!(ws)
        P = Matrix(Diagonal([ones(T, 7); T(1e20)]))
        rows = [3, 2, 1]
        ZP = zeros(T, length(rows), size(Z, 2))
        Siphon._measurement_cov_mul!(ZP, ws, P, rows)
        @test ZP ≈ Z[rows, :] * P
        @test ZP[1, 8] ≈ one(T)
        x = zeros(T, size(Z, 2))
        u = T[1e20, 2, -1]
        Siphon._measurement_tmul_add!(x, ws, rows, u)
        @test x ≈ Z[rows, :]' * u
        @test x[8] ≈ one(T)
    end
end

@testset "measurement support — noiseless rank-deficient observations" begin
    d = _ms_setup(Float64)
    Z = copy(d.Z)
    Z[2, :] .= 2 .* Z[1, :]
    H = zeros(size(d.H))
    rng = Xoshiro(608)
    states = zeros(length(d.a1), d.n)
    states[:, 1] = d.a1 + cholesky(Symmetric(d.P1)).L * randn(rng, length(d.a1))
    LQ = cholesky(Symmetric(d.Q)).L
    for t in 2:d.n
        states[:, t] = d.c + d.Tm * states[:, t - 1] + d.s[t - 1] * d.R * LQ * randn(rng, 2)
    end
    y = Z * states
    y[:, 5] .= NaN
    y[3, end] = NaN
    keep = [1, 3]
    ref = _ps_posterior(Z[keep, :], H[keep, keep], d.Tm, d.R, d.Q, d.c, d.s,
        d.a1, d.P1, y[keep, :])
    ws = _ms_workspace(d; Z, H)
    set_rank_revealing!(ws, true)
    kalman_filter!(ws, y)
    kalman_smoother!(ws)
    @test maximum(innovation_ranks(ws)) == 2
    @test vec(smoothed_states(ws)) ≈ ref.mean atol = 1e-10
    draws = simulation_smoother(ws, y, 3; rng = Xoshiro(609))
    obs = .!isnan.(y)
    for draw in axes(draws, 3)
        @test (Z * draws[:, :, draw])[obs] ≈ y[obs] atol = 1e-10
    end
end

@testset "measurement support — derivatives at zero loadings" begin
    base = _ms_setup(Float64)
    d = merge(base, (c = zeros(8), s = ones(base.n)))
    function loglik(θ; functional = false)
        Z = d.Z .+ zero(θ)
        Z[2, 8] = θ
        if functional
            return kalman_loglik(KFParms(Z, d.H, d.Tm, d.R, d.Q), d.y, d.a1, d.P1)
        end
        return kalman_filter!(_ms_workspace(d; Z), d.y)
    end
    derivative = ForwardDiff.derivative(loglik, 0.0)
    reference = ForwardDiff.derivative(θ -> loglik(θ; functional = true), 0.0)
    @test abs(derivative) > 1
    @test derivative ≈ reference rtol = 1e-10
end
