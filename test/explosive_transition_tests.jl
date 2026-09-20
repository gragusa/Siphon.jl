# Filtering a transition whose spectral radius exceeds one. Such a transition is
# admissible — a Gibbs sampler for a VAR draws explosive coefficients from the
# posterior — and the filter has to carry it without losing positive
# definiteness of the innovation covariance.

using Test
using Siphon
using LinearAlgebra
using Random

"""
    _explosive_setup(; nser, nlag, n, ρ, seed)

A companion-form VAR scaled to spectral radius `ρ`, as a state-space model:
state `αₜ = (xₜ', xₜ₋₁', …)'` of dimension `nser·nlag`, `R = (I; 0)`, `Z = (I 0)`
observing the current period, `H = 0`, and an exact start `a1 = 0`, `P1 = 0`.

The panel is fully observed apart from the first period, which is all-`NaN`: the
state is pinned by the observations alone, so every filtered covariance is zero
in exact arithmetic and any growth in `Ptt` is accumulated rounding error.
"""
function _explosive_setup(; nser::Int = 5, nlag::Int = 3, n::Int = 150,
        ρ::Float64 = 1.3, seed::Int = 20260920)
    rng = Xoshiro(seed)
    m = nser * nlag

    A = [0.4^l .* randn(rng, nser, nser) ./ sqrt(nser) for l in 1:nlag]

    # Scaling lag block `Aₗ` by `cˡ` scales every companion eigenvalue by `c`,
    # so one pass over the blocks sets the spectral radius exactly.
    companion(blocks) = begin
        M = zeros(m, m)
        for l in 1:nlag
            M[1:nser, ((l - 1) * nser + 1):(l * nser)] = blocks[l]
        end
        if nlag > 1
            M[(nser + 1):end, 1:(nser * (nlag - 1))] = I(nser * (nlag - 1))
        end
        M
    end
    c = ρ / maximum(abs, eigvals(companion(A)))
    Tm = companion([c^l .* A[l] for l in 1:nlag])

    R = zeros(m, nser)
    R[1:nser, :] = I(nser)
    Xq = randn(rng, nser, nser)
    Q = Matrix(Symmetric(Xq * Xq' / nser + 0.5I))
    Z = zeros(nser, m)
    Z[:, 1:nser] = I(nser)
    H = zeros(nser, nser)
    a1 = zeros(m)
    P1 = zeros(m, m)

    Lq = cholesky(Q).L
    y = zeros(nser, n)
    a = zeros(m)
    for t in 1:n
        y[:, t] = Z * a
        a = Tm * a
        a[1:nser] += Lq * randn(rng, nser)
        # An explosive recursion overflows a long sample; renormalizing the
        # shock scale keeps the fixture's numbers in a readable range without
        # changing the transition.
        a ./= max(one(eltype(a)), maximum(abs, a) / 10)
    end
    y[:, 1] .= NaN

    return (; Z, H, Tm, R, Q, a1, P1, y, nser, nlag, m, n, ρ)
end

function _explosive_workspace(s; block::Int = 0, rank_revealing::Bool = false)
    ws = KalmanWorkspace(s.nser, s.m, s.nser, s.n)
    set_params!(ws, s.Z, s.H, s.Tm, s.R, s.Q)
    set_initial!(ws, s.a1, s.P1)
    block == 0 || set_companion_structure!(ws, block)
    rank_revealing && set_rank_revealing!(ws, true)
    return ws
end

@testset "explosive transition — the filtered covariance stays at zero" begin
    s = _explosive_setup()
    @test maximum(abs, eigvals(s.Tm)) ≈ s.ρ

    for block in (0, s.nser), rr in (false, true)

        ws = _explosive_workspace(s; block = block, rank_revealing = rr)
        @test filter_and_smooth!(ws, s.y) isa Real

        Ptt = variances_filtered_states(ws)
        @test maximum(abs, Ptt) < 1.0e-10

        for t in 1:(s.n)
            P = view(Ptt, :, :, t)
            @test P ≈ transpose(P) atol=1.0e-14
        end
    end
end

@testset "explosive transition — the simulation smoother runs" begin
    s = _explosive_setup()
    ws = _explosive_workspace(s; block = s.nser)
    sws = SimulationSmootherWorkspace(ws)
    refresh_factors!(sws, ws)

    draw = Matrix{Float64}(undef, s.m, s.n)
    simulation_smoother!(draw, sws, ws, s.y; rng = Xoshiro(11))
    @test all(isfinite, draw)
end
