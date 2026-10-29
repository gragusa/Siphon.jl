"""
Restricted M-steps for T and Z when the noise covariance of the equation is
singular.

  (1) DNS with a constant state (T = [diag(b) u; 0 1], R = [I; 0], full Q):
      R Q Rᵀ is singular and the rows of T have different free columns. The EM
      fixed point is the maximum likelihood estimate, with and without a
      measurement variance held at 0.

  (2) A Q diagonal frozen at 0: the T row driven only by that shock keeps its
      values; the other rows reach a zero score.

  (3) Tied Z cells with a zero measurement variance: a tie group that reaches
      the zero-variance row keeps its value; the other groups move.
"""

using Test
using Siphon
using LinearAlgebra
using Random
using CSV
using DataFrames
include("em_helpers.jl")

function constant_state_spec(C, b, u, Q, h, a1, P1)
    p = size(C, 1)
    Tm = vcat(
        hcat(
            Any[i == j ? FreeParam(Symbol(:b, i); init = b[i]) : 0.0
                for i in 1:3, j in 1:3],
            Any[FreeParam(Symbol(:u, i); init = u[i]) for i in 1:3]),
        Any[0 0 0 1])
    Qs = Matrix{Any}(undef, 3, 3)
    for j in 1:3, i in j:3

        Qs[i, j] = Qs[j, i] = FreeParam(Symbol(:q, i, j); init = Q[i, j],
            lower = i == j ? 0.0 : -Inf)
    end
    # A zero in `h` fixes that measurement variance at 0.
    Hs = Any[i == j ?
             (iszero(h[i]) ? 0.0 : FreeParam(Symbol(:h, i); init = h[i], lower = 0.0)) :
             0.0 for i in 1:p, j in 1:p]
    custom_ssm(Z = [C zeros(p)], T = Tm, R = [I(3); zeros(1, 3)], Q = Qs, H = Hs,
        a1 = a1, P1 = P1, name = :constant_state)
end

constant_state_free() = [i == j || (j == 4 && i <= 3) for i in 1:4, j in 1:4]

# Central-difference score of the log-likelihood in the cells `cells` of T.
function transition_score(r, y, a1, P1, cells; δ = 1e-6)
    ll(T) = kalman_loglik(KFParms(r.Z, r.H, T, r.R, r.Q), y, a1, P1)
    map(cells) do I
        Tp = copy(r.T)
        Tp[I] += δ
        Tm = copy(r.T)
        Tm[I] -= δ
        (ll(Tp) - ll(Tm)) / 2δ
    end
end

const B_U_CELLS = [CartesianIndex(1, 1), CartesianIndex(2, 2), CartesianIndex(3, 3),
    CartesianIndex(1, 4), CartesianIndex(2, 4), CartesianIndex(3, 4)]

@testset "EM with a constant state: T update on the support of the shocks" begin
    y = Matrix(Matrix(CSV.read(joinpath(@__DIR__, "marss_dns_data.csv"), DataFrame))')
    res = CSV.read(joinpath(@__DIR__, "marss_dns_diag_results.csv"), DataFrame)
    λ = only(res.value[res.parameter .== "lambda"])
    maturities = [3, 12, 24, 60, 120]
    p, n = size(y)
    C = [ones(p) Siphon.dns_loading1.(λ, maturities) Siphon.dns_loading2.(λ, maturities)]
    a1 = [0.0, 0.0, 0.0, 1.0]
    P1 = Matrix(Diagonal([100.0, 100.0, 100.0, 0.0]))
    T0 = [0.95 * I(3) zeros(3); 0 0 0 1.0]

    # allow_degen sets the 12-month measurement variance to 0 on this dataset.
    em = em_matrices([C zeros(p)], T0, [I(3); zeros(1, 3)], fill(0.01, p),
        0.1 * Matrix(1.0I, 3, 3), y, a1, P1;
        Z_free = falses(p, 4), T_free = constant_state_free(), H_free = trues(p),
        maxiter = 20000, tol = 1e-13, allow_degen = true, degen_lim = 1e-4,
        min_degen_iter = 5)
    @test em.converged
    @test em.H[3, 3] == 0
    @test all(diff(em.loglik_history) .>= -1e-8 * abs(em.loglik))
    @test maximum(abs, transition_score(em, y, a1, P1, B_U_CELLS)) < 1e-3

    mle = StateSpaceModel(
        constant_state_spec(C, diag(em.T), em.T[1:3, 4], em.Q, diag(em.H), a1, P1), n)
    fit!(MLE(), mle, y)
    sys = system_matrices(mle)
    @test isapprox(em.loglik, mle.loglik; rtol = 1e-9)
    @test isapprox(em.T, sys.T; atol = 1e-4)
    @test isapprox(em.Q, sys.Q; atol = 1e-4)
    @test isapprox(em.H, sys.H; atol = 1e-4)

    # No zero variance: data simulated from the model with H = 0.01 I.
    Random.seed!(20261029)
    B = Diagonal([0.97, 0.9, 0.8])
    u = [0.1, -0.05, 0.02]
    LQ = cholesky([0.09 0.03 0.0; 0.03 0.16 -0.04; 0.0 -0.04 0.25]).L
    nsim = 300
    β = zeros(3, nsim)
    β[:, 1] = (I - B) \ u
    for t in 2:nsim
        β[:, t] = u + B * β[:, t - 1] + LQ * randn(3)
    end
    ysim = C * β + 0.1 * randn(p, nsim)
    sim = em_matrices([C zeros(p)], T0, [I(3); zeros(1, 3)], fill(0.05, p),
        0.1 * Matrix(1.0I, 3, 3), ysim, a1, P1;
        Z_free = falses(p, 4), T_free = constant_state_free(), H_free = trues(p),
        maxiter = 20000, tol = 1e-13)
    @test sim.converged
    @test all(>(1e-4), diag(sim.H))
    @test all(diff(sim.loglik_history) .>= -1e-8 * abs(sim.loglik))
    @test maximum(abs, transition_score(sim, ysim, a1, P1, B_U_CELLS)) < 1e-3
    simle = StateSpaceModel(
        constant_state_spec(C, diag(sim.T), sim.T[1:3, 4], sim.Q, diag(sim.H), a1, P1),
        nsim)
    fit!(MLE(), simle, ysim)
    @test isapprox(sim.loglik, simle.loglik; rtol = 1e-9)
    @test isapprox(sim.T, system_matrices(simle).T; atol = 1e-4)

    # A shock frozen at 0: the factor it drives is deterministic, so its row of
    # T keeps its starting values while the other rows reach a zero score.
    Q0 = copy(em.Q)
    Q0[3, :] .= 0
    Q0[:, 3] .= 0
    kf_ws = KalmanWorkspace([C zeros(p)], Matrix(Diagonal(diag(em.H) .+ 1e-3)), em.T,
        [I(3); zeros(1, 3)], Q0, a1, P1, n)
    em_ws = Siphon.EMWorkspace(kf_ws)
    em_ws.Z_free .= false
    em_ws.T_free .= constant_state_free()
    em_ws.H_free .= Diagonal(trues(p))
    em_ws.H_diag_only = true
    em_ws.Q_zero[3] = true
    frozen = Siphon.em_estimate!(kf_ws, em_ws, y; maxiter = 20000, tol = 1e-13,
        warn = false, allow_degen = true, degen_lim = 1e-4, min_degen_iter = 5)
    @test frozen.converged
    @test frozen.Q[3, :] == zeros(3)
    @test frozen.T[3, :] == em.T[3, :]
    @test all(diff(frozen.loglik_history) .>= -1e-8 * abs(frozen.loglik))
    @test maximum(abs,
        transition_score(frozen, y, a1, P1, B_U_CELLS[[1, 2, 4, 5]])) < 1e-3
end

@testset "Tied Z cells with a zero measurement variance" begin
    Random.seed!(3)
    n = 300
    α = zeros(2, n)
    for t in 2:n
        α[:, t] = [0.8 0.0; 0.0 0.6] * α[:, t - 1] + randn(2)
    end
    Ztrue = [1.0 0.5; 0.3 1.0; 1.0 0.4]
    y = Ztrue * α + Diagonal([0.3, 0.3, 0.0]) * randn(3, n)

    Z0 = [0.8 0.4; 0.3 0.4; 0.8 0.4]
    zfree = [true true; false true; true false]
    ties = [[CartesianIndex(1, 1), CartesianIndex(3, 1)],
        [CartesianIndex(1, 2), CartesianIndex(2, 2)]]
    kf_ws = KalmanWorkspace(Z0, Matrix(Diagonal([0.1, 0.1, 0.0])),
        [0.8 0.0; 0.0 0.6], Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2),
        zeros(2), 10.0 * Matrix(1.0I, 2, 2), n)
    em_ws = Siphon.EMWorkspace(kf_ws)
    em_ws.Z_free .= zfree
    append!(em_ws.Z_ties, ties)
    em_ws.H_free .= Diagonal([true, true, false])
    em_ws.T_free .= false
    em_ws.Q_free .= false
    r = Siphon.em_estimate!(kf_ws, em_ws, y; maxiter = 200, tol = 1e-10, warn = false)

    # The first group reaches row 3, which is measured without error.
    @test r.Z[1, 1] == r.Z[3, 1] == 0.8
    @test r.Z[1, 2] == r.Z[2, 2]
    @test r.Z[1, 2] != 0.4
    @test r.H[3, 3] == 0
    @test all(diff(r.loglik_history) .>= -1e-8 * abs(r.loglik))
end
