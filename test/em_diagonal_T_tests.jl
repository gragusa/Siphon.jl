"""
Restricted transition matrices in `em_estimate!`, set through `T_free`:

  (1) Diagonal T. EM recovers the AR(1) coefficient of each factor and keeps
      the off-diagonal cells at zero.

  (2) Augmented-state intercept: T is (m+1) × (m+1), the last column holds a
      state intercept and the last row is [0 … 0 1]. The free cells are the
      factor diagonal and the intercept column of the factor rows.

  (3) A fully free T gives the unrestricted M-step.
"""

using Test
using LinearAlgebra
using Random
using Siphon
include("em_helpers.jl")

@testset "EM with a diagonal T" begin
    Random.seed!(20260424)

    # 3-factor DNS-like DGP with a DIAGONAL B. No observation loadings —
    # factors are observed 1-to-1 through an identity Z.
    m = 3
    n = 500
    B_true = diagm([0.95, 0.85, 0.70])
    Q_true = diagm([0.30, 0.20, 0.10])
    H_true = 1e-6 * I(m)
    α = zeros(m, n)
    α[:, 1] = zeros(m)
    for t in 2:n
        α[:, t] = B_true * α[:, t - 1] + sqrt.(diag(Q_true)) .* randn(m)
    end
    y = α + sqrt.(diag(H_true)) .* randn(m, n)

    Z = Matrix{Float64}(I, m, m)
    T_init = diagm([0.5, 0.5, 0.5])
    R = Matrix{Float64}(I, m, m)
    H_init = 0.01 .* Matrix{Float64}(I, m, m)
    Q_init = 0.1 .* Matrix{Float64}(I, m, m)
    a1 = zeros(m)
    P1 = 10.0 .* Matrix{Float64}(I, m, m)

    # T_free marks only the diagonal as free.
    T_free = Matrix(I(m)) .== 1

    em = em_matrices(
        Z, T_init, R, H_init, Q_init, y, a1, P1;
        Z_free = falses(m, m),
        T_free = T_free,
        H_free = Matrix(I(m)) .== 1,
        Q_free = Matrix(I(m)) .== 1,   # diagonal Q
        maxiter = 1000,
        tol = 1e-9
    )

    # Off-diagonals of T must stay exactly at 0.
    for i in 1:m, j in 1:m

        i == j && continue
        @test em.T[i, j] == 0.0
    end

    # Diagonal B recovers the DGP values within Monte Carlo tolerance.
    @test isapprox(em.T[1, 1], 0.95; atol = 0.03)
    @test isapprox(em.T[2, 2], 0.85; atol = 0.04)
    @test isapprox(em.T[3, 3], 0.70; atol = 0.05)
end

@testset "EM with a diagonal T and a state intercept" begin
    Random.seed!(20260425)

    # 2-factor DGP with diagonal B AND a state intercept u.
    # True factors drift around u ./ (1 − B) in the unconditional mean.
    m₀ = 2
    n = 800
    B_true = diagm([0.80, 0.60])
    u_true = [0.40, -0.25]
    Q_true = diagm([0.20, 0.15])
    H_true = 1e-6 * I(m₀)

    α₀ = zeros(m₀, n)
    α₀[:, 1] = u_true
    for t in 2:n
        α₀[:, t] = u_true + B_true * α₀[:, t - 1] + sqrt.(diag(Q_true)) .* randn(m₀)
    end
    y = α₀ + sqrt.(diag(H_true)) .* randn(m₀, n)

    # Augmented state: α̃ = (α, 1). Dimension m = m₀ + 1 = 3.
    m = m₀ + 1
    Z = [Matrix{Float64}(I, m₀, m₀) zeros(m₀, 1)]              # m₀ × m
    T_init = zeros(m, m)
    T_init[1:m₀, 1:m₀] = diagm(fill(0.5, m₀))
    T_init[1:m₀, m] .= 0.0            # start intercept at 0
    T_init[m, m] = 1.0            # constant row pinned
    R = [Matrix{Float64}(I, m₀, m₀); zeros(1, m₀)]            # m × m₀
    H_init = 0.01 .* Matrix{Float64}(I, m₀, m₀)
    Q_init = 0.1 .* Matrix{Float64}(I, m₀, m₀)
    ã1 = zeros(m)
    ã1[m] = 1.0
    P̃1 = diagm([10.0, 10.0, 0.0])

    # Free mask: diagonal of the leading block AND the intercept column (for
    # factor rows only). The constant row is fixed.
    T_free = falses(m, m)
    for i in 1:m₀
        T_free[i, i] = true
        T_free[i, m] = true
    end

    em = em_matrices(
        Z, T_init, R, H_init, Q_init, y, ã1, P̃1;
        Z_free = falses(m₀, m),
        T_free = T_free,
        H_free = Matrix(I(m₀)) .== 1,
        Q_free = Matrix(I(m₀)) .== 1,
        maxiter = 2000,
        tol = 1e-9
    )

    # Off-diagonal factor entries of the augmented T are zero.
    for i in 1:m₀, j in 1:m₀

        i == j && continue
        @test em.T[i, j] == 0.0
    end

    # Constant row is unchanged.
    @test em.T[m, 1:m₀] == zeros(m₀)
    @test em.T[m, m] == 1.0

    # Estimated B (diagonal) and u (intercept column) close to the DGP.
    B̂ = diag(em.T)[1:m₀]
    û = em.T[1:m₀, m]
    @test isapprox(B̂[1], 0.80; atol = 0.05)
    @test isapprox(B̂[2], 0.60; atol = 0.05)
    @test isapprox(û[1], 0.40; atol = 0.10)
    @test isapprox(û[2], -0.25; atol = 0.10)
end

@testset "EM with a fully free T" begin
    Random.seed!(20260426)

    # Same DGP as test 1 but now we allow a FULL T update (default path).
    m = 2
    n = 400
    B_true = [0.80 0.10; -0.05 0.70]
    Q_true = diagm([0.20, 0.15])

    α = zeros(m, n)
    for t in 2:n
        α[:, t] = B_true * α[:, t - 1] + sqrt.(diag(Q_true)) .* randn(m)
    end
    y = α + 1e-3 .* randn(m, n)

    Z = Matrix{Float64}(I, m, m)
    T_init = 0.5 .* Matrix{Float64}(I, m, m)
    R = Matrix{Float64}(I, m, m)
    H_init = 0.01 .* Matrix{Float64}(I, m, m)
    Q_init = 0.1 .* Matrix{Float64}(I, m, m)
    a1 = zeros(m)
    P1 = 10.0 .* Matrix{Float64}(I, m, m)

    em = em_matrices(
        Z, T_init, R, H_init, Q_init, y, a1, P1;
        Z_free = falses(m, m),
        T_free = trues(m, m),
        H_free = Matrix(I(m)) .== 1,
        Q_free = Matrix(I(m)) .== 1,
        maxiter = 1500,
        tol = 1e-9
    )

    # Full 2×2 B recovered within Monte Carlo tolerance. Off-diagonals are
    # genuinely nonzero (the default, unrestricted M-step finds them).
    @test isapprox(em.T[1, 1], 0.80; atol = 0.05)
    @test isapprox(em.T[1, 2], 0.10; atol = 0.10)
    @test isapprox(em.T[2, 1], -0.05; atol = 0.10)
    @test isapprox(em.T[2, 2], 0.70; atol = 0.05)
end
