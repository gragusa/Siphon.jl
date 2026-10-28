"""
EM started from a diagonal variance at its lower bound 0.

With `allow_degen = true` such a variance is held at 0, so warm-starting from a
fit in which `allow_degen` froze a variance continues that fit. Without
`allow_degen`, and for maximum likelihood, the start is rejected.
"""

using Test
using Siphon
using LinearAlgebra
using CSV
using DataFrames

# The MARSS DNS dataset: EM with `allow_degen` sets the third measurement
# variance to 0 (as MARSS does).
function zero_start_dns_spec(Z, B, H, Q, P1)
    p, m = size(Z)
    T = Any[i == j ? FreeParam(Symbol(:B, i); init = B[i]) : 0.0 for i in 1:m, j in 1:m]
    Hs = Any[i == j ? FreeParam(Symbol(:H, i); init = H[i], lower = 0.0) : 0.0
             for i in 1:p, j in 1:p]
    Qs = Any[i == j ? FreeParam(Symbol(:Q, i); init = Q[i], lower = 0.0) : 0.0
             for i in 1:m, j in 1:m]
    custom_ssm(Z = Z, H = Hs, T = T, R = Matrix(1.0I, m, m), Q = Qs,
        a1 = zeros(m), P1 = P1, name = :dns_zero_start)
end

@testset "EM warm start from a zero variance" begin
    y = Matrix(Matrix(CSV.read(joinpath(@__DIR__, "marss_dns_data.csv"), DataFrame))')
    res = CSV.read(joinpath(@__DIR__, "marss_dns_diag_results.csv"), DataFrame)
    λ = only(res.value[res.parameter .== "lambda"])
    maturities = [3, 12, 24, 60, 120]
    p, n, m = size(y, 1), size(y, 2), 3
    Z = [ones(p) Siphon.dns_loading1.(λ, maturities) Siphon.dns_loading2.(λ, maturities)]
    P1 = 100.0 * Matrix(1.0I, m, m)
    degen = (allow_degen = true, degen_lim = 1e-4, min_degen_iter = 5)

    cold = StateSpaceModel(
        zero_start_dns_spec(Z, fill(0.95, m), fill(0.01, p),
            fill(0.1, m), P1), n)
    fit!(EM(), cold, y; maxiter = 5000, tol = 1e-12, degen...)
    sys = system_matrices(cold)
    @test cold.converged
    @test sys.H[3, 3] == 0

    # The cold fit's estimates, H3 = 0 included, start the second fit.
    warm_spec = zero_start_dns_spec(Z, diag(sys.T), diag(sys.H), diag(sys.Q), P1)
    @test only(p.init for p in warm_spec.params if p.name == :H3) == 0

    warm = StateSpaceModel(warm_spec, n)
    fit!(EM(), warm, y; maxiter = 5000, tol = 1e-12, degen...)
    wsys = system_matrices(warm)
    @test warm.converged
    @test wsys.H[3, 3] == 0
    @test warm.loglik >= cold.loglik - 1e-8 * abs(cold.loglik)
    @test isapprox(warm.loglik, cold.loglik; rtol = 1e-10)
    @test isapprox(diag(wsys.T), diag(sys.T); rtol = 1e-5)
    @test isapprox(diag(wsys.H), diag(sys.H); rtol = 1e-4)

    # Held from the first iteration: two iterations stop before the degeneracy
    # test (min_degen_iter = 5) could set the cell to 0.
    short = StateSpaceModel(warm_spec, n)
    fit!(EM(), short, y; maxiter = 2, tol = 1e-12, warn = false, degen...)
    @test system_matrices(short).H[3, 3] == 0

    @test_throws "starts at its lower bound 0" fit!(EM(), StateSpaceModel(warm_spec, n), y)
    @test_throws "strictly inside the bounds" optimize_ssm(warm_spec, y)
    @test_throws "strictly inside the bounds" fit!(MLE(), StateSpaceModel(warm_spec, n), y)

    # A zero start on the diagonal of Q is held at 0 the same way.
    q_spec = zero_start_dns_spec(Z, diag(sys.T), diag(sys.H), [diag(sys.Q)[1:2]; 0.0], P1)
    q_model = StateSpaceModel(q_spec, n)
    fit!(EM(), q_model, y; maxiter = 200, tol = 1e-8, warn = false, degen...)
    @test system_matrices(q_model).Q[3, 3] == 0
    @test isfinite(q_model.loglik)
end
