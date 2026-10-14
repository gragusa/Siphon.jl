# Exact diffuse smoother: functional and in-place, checked against the ordinary
# smoother with initial covariance P1_star + κ P1_inf in high precision.

using Siphon
using Test
using LinearAlgebra
using Random
using ForwardDiff

# The ordinary smoother in BigFloat with κ = 1e40 agrees with the κ → ∞ limit far
# beyond Float64 precision.
function diffuse_smoother_oracle(p, y, a1, P1_star, P1_inf)
    setprecision(BigFloat, 512) do
        B(x) = big.(x)
        pb = KFParms(B(p.Z), B(p.H), B(p.T), B(p.R), B(p.Q))
        sm = kalman_smoother(kalman_filter(pb, B(y), B(a1),
            B(P1_star) + big(10)^40 * B(P1_inf)))
        (Float64.(smoothed_states(sm)), Float64.(smoothed_states_cov(sm)))
    end
end

# Two random walks and a stationary AR(1), observed through three rows.
function diffuse_smoother_system(H)
    Z = [1.0 0.0 1.0; 1.0 1.0 0.0; 0.5 0.5 1.0]
    Tm = [1.0 0 0; 0 1.0 0; 0 0 0.7]
    Q = Matrix(Diagonal([0.3, 0.2, 0.5]))
    P1_star = zeros(3, 3)
    P1_star[3, 3] = 0.5 / (1 - 0.49)
    P1_inf = Matrix(Diagonal([1.0, 1.0, 0.0]))
    return KFParms(Z, H, Tm, Matrix(1.0I, 3, 3), Q), zeros(3), P1_star, P1_inf
end

const DS_HDIAG = Matrix(Diagonal([1.0, 0.5, 0.8]))
const DS_HFULL = [1.0 0.3 0.1; 0.3 0.5 0.2; 0.1 0.2 0.8]

@testset "Exact diffuse smoother matches the large-κ limit" begin
    y = randn(MersenneTwister(2), 3, 25)
    y_partial = copy(y)
    y_partial[2:3, 1] .= NaN
    y_mixed = copy(y)
    y_mixed[:, 1] .= NaN
    y_mixed[1, 2] = NaN
    y_mixed[3, 3] = NaN

    @testset "$hname, $yname" for (hname, H) in (("diagonal H", DS_HDIAG),
            ("full H", DS_HFULL)),
        (yname, yy, flags) in (("complete", y, [2]), ("partial", y_partial, [1, 2]),
            ("missing", y_mixed, [-1, 2, 2]))

        p, a1, P1_star, P1_inf = diffuse_smoother_system(H)
        filt = kalman_filter(p, yy, a1, P1_star, P1_inf)
        @test diffuse_flags(filt) == flags
        sm = kalman_smoother(filt)
        α_ref, V_ref = diffuse_smoother_oracle(p, yy, a1, P1_star, P1_inf)
        @test smoothed_states(sm) ≈ α_ref atol = 1e-10
        @test smoothed_states_cov(sm) ≈ V_ref atol = 1e-10
    end

    @testset "Finf zero in the first period" begin
        p, a1, P1_star, P1_inf = diffuse_smoother_system(DS_HFULL)
        p = KFParms([0.0 0.0 1.0; 1.0 0.0 0.0; 0.0 1.0 1.0], p.H, p.T, p.R, p.Q)
        yy = copy(y)
        yy[2:3, 1] .= NaN
        filt = kalman_filter(p, yy, a1, P1_star, P1_inf)
        @test diffuse_flags(filt) == [0, 2]
        α_ref, V_ref = diffuse_smoother_oracle(p, yy, a1, P1_star, P1_inf)
        sm = kalman_smoother(filt)
        @test smoothed_states(sm) ≈ α_ref atol = 1e-10
        @test smoothed_states_cov(sm) ≈ V_ref atol = 1e-10
    end

    @testset "Local linear trend" begin
        p = KFParms([1.0 0.0], [2.0;;], [1.0 1.0; 0.0 1.0], Matrix(1.0I, 2, 2),
            Matrix(Diagonal([0.5, 0.1])))
        yy = cumsum(randn(MersenneTwister(1), 1, 30); dims = 2)
        yy[1, 2] = NaN
        filt = kalman_filter(p, yy, zeros(2), zeros(2, 2), Matrix(1.0I, 2, 2))
        @test diffuse_flags(filt) == [1, -1, 1]
        α_ref, V_ref = diffuse_smoother_oracle(p, yy, zeros(2), zeros(2, 2),
            Matrix(1.0I, 2, 2))
        sm = kalman_smoother(filt)
        @test smoothed_states(sm) ≈ α_ref atol = 1e-10
        @test smoothed_states_cov(sm) ≈ V_ref atol = 1e-10
    end
end

@testset "In-place exact diffuse smoother" begin
    y = randn(MersenneTwister(3), 3, 25)
    y[2:3, 1] .= NaN
    y[:, 4] .= NaN
    p, a1, P1_star, P1_inf = diffuse_smoother_system(DS_HFULL)
    sm = kalman_smoother(kalman_filter(p, y, a1, P1_star, P1_inf))

    ws = DiffuseKalmanWorkspace(p.Z, p.H, p.T, p.R, p.Q, a1, P1_star, P1_inf, size(y, 2))
    @test_throws "kalman_filter!(ws, y)" kalman_smoother!(ws)
    kalman_filter_and_smooth!(ws, y)
    @test smoothed_states(ws) ≈ smoothed_states(sm) atol = 1e-12
    @test smoothed_states_cov(ws) ≈ smoothed_states_cov(sm) atol = 1e-12

    res = SmootherResult(ws)
    @test smoothed_states(res) == smoothed_states(ws)
    @test smoothed_states_cov(res) == smoothed_states_cov(ws)

    kalman_smoother!(ws; covariances = false)
    @test smoothed_states(ws) ≈ smoothed_states(sm) atol = 1e-12
    @test_throws ArgumentError smoothed_states_cov(ws)

    @test_throws "cross-lag covariances are not available" kalman_smoother!(ws;
        crosscov = true)
    @test_throws "cross-lag covariances are not available" kalman_smoother(
        kalman_filter(p, y, a1, P1_star, P1_inf); crosscov = true)

    # Neither the filter nor the smoother allocates in the diffuse period.
    kalman_filter!(ws, y)
    @test (@allocated kalman_filter!(ws, y)) == 0
    kalman_smoother!(ws)
    @test (@allocated kalman_smoother!(ws)) == 0
end

@testset "DiffuseFilterResult(ws)" begin
    y = randn(MersenneTwister(4), 3, 20)
    y[1, 2] = NaN
    p, a1, P1_star, P1_inf = diffuse_smoother_system(DS_HDIAG)
    filt = kalman_filter(p, y, a1, P1_star, P1_inf)
    ws = DiffuseKalmanWorkspace(p.Z, p.H, p.T, p.R, p.Q, a1, P1_star, P1_inf, size(y, 2))
    @test_throws "kalman_filter!(ws, y)" DiffuseFilterResult(ws)
    kalman_filter!(ws, y)
    r = DiffuseFilterResult(ws)
    @test r isa DiffuseFilterResult
    @test loglikelihood(r) ≈ loglikelihood(filt)
    @test diffuse_period(r) == diffuse_period(filt)
    @test diffuse_flags(r) == diffuse_flags(filt)
    Pinf_r, Pstar_r = diffuse_covariances(r)
    Pinf_f, Pstar_f = diffuse_covariances(filt)
    @test Pinf_r ≈ Pinf_f
    @test Pstar_r ≈ Pstar_f
    @test smoothed_states(kalman_smoother(r)) ≈ smoothed_states(kalman_smoother(filt))
    kalman_filter!(ws, 2 .* y)
    @test loglikelihood(r) ≈ loglikelihood(filt)
end

@testset "Diffuse period that does not end" begin
    # Only the sum of the two random walks is observed.
    p, a1, P1_star, P1_inf = diffuse_smoother_system(DS_HDIAG)
    p = KFParms([1.0 1.0 0.0], [1.0;;], p.T, p.R, p.Q)
    y = randn(MersenneTwister(5), 1, 10)
    filt = kalman_filter(p, y, a1, P1_star, P1_inf)
    @test_throws "does not end within the sample" kalman_smoother(filt)
    @test_throws "does not end within the sample" forecast(filt, 2)
    ws = DiffuseKalmanWorkspace(p.Z, p.H, p.T, p.R, p.Q, a1, P1_star, P1_inf, 10)
    kalman_filter!(ws, y)
    @test_throws "does not end within the sample" kalman_smoother!(ws)

    # A diffuse period that ends in the last period is not an error.
    pl = KFParms([1.0;;], [1.0;;], [1.0;;], [1.0;;], [0.5;;])
    r = kalman_filter(pl, [1.5;;], [0.0], [0.0;;], [1.0;;])
    @test diffuse_period(r) == 1
    @test smoothed_states(kalman_smoother(r)) ≈ [1.5;;]
    @test forecast(r, 1) isa ForecastResult
end

@testset "StateSpaceModel smoothing with diffuse = :exact" begin
    y = randn(MersenneTwister(6), 1, 40)
    y[1, 3] = NaN
    spec = local_linear_trend(; diffuse = :exact)
    θ = (var_obs = 1.0, var_level = 0.5, var_slope = 0.1)
    model = StateSpaceModel(spec, θ, size(y, 2))
    kalman_filter!(model, y)
    ss = Siphon.build_linear_state_space(spec, θ, y)
    sm = kalman_smoother(kalman_filter(ss.p, y, ss.a1, ss.P1, ss.P1_inf))
    @test smoothed_states(model) ≈ smoothed_states(sm)
    @test smoothed_states_cov(model) ≈ smoothed_states_cov(sm)
end

@testset "Exact diffuse smoother is AD-compatible" begin
    y = randn(MersenneTwister(7), 3, 15)
    function f(θ)
        p, a1, P1_star, P1_inf = diffuse_smoother_system(θ[1] .* DS_HFULL)
        sm = kalman_smoother(kalman_filter(p, y, a1, P1_star, P1_inf))
        return sum(smoothed_states(sm)) + sum(smoothed_states_cov(sm))
    end
    g = ForwardDiff.gradient(f, [1.0])
    h = 1e-6
    @test g[1] ≈ (f([1.0 + h]) - f([1.0 - h])) / (2h) rtol = 1e-5
end
