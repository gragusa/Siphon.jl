# EKF extension tests.
#
# Coverage (matches Plan §15):
#   - Linear-equivalence: under a LinearMeasurement, kalman_loglik / kalman_filter /
#     EKFWorkspace + kalman_filter! must match kalman_loglik / kalman_filter exactly.
#   - Missing-data handling.
#   - AD vs analytic Jacobian agreement on a smooth nonlinear measurement.
#   - Workspace allocation smoke-test (zero in-loop allocations after warm-up).
#   - Parameter-AD: ForwardDiff through kalman_loglik wrt system parameters.

using Siphon
using Test
using LinearAlgebra
using Random
using ForwardDiff

# ---------------------------------------------------------------------------
# Helpers — measurement structs used across testsets
# ---------------------------------------------------------------------------

# Defines only the functional measurement, not the in-place one EKFWorkspace needs
struct EKFTestIncompleteMeasurement <: Siphon.AbstractEKFMeasurement end
Siphon.measurement(::EKFTestIncompleteMeasurement, a, t) = a[1:1]

# Stable softplus / logistic without LogExpFunctions (test-local)
@inline _ekf_softplus(x::Real) = x >= 0 ? x + log1p(exp(-x)) : log1p(exp(x))
@inline _ekf_logistic(x::Real) = x >= 0 ? inv(one(x) + exp(-x)) : exp(x) / (one(x) + exp(x))

struct EKFTestSoftplusMeasurement{LT <: AbstractMatrix, T <: Real} <:
       Siphon.AbstractEKFMeasurement
    Λ::LT
    r_LB::T
end

function Siphon.measurement(m::EKFTestSoftplusMeasurement, β, t)
    sh = m.Λ * β
    return m.r_LB .+ _ekf_softplus.(sh .- m.r_LB)
end

function Siphon.measurement!(out, m::EKFTestSoftplusMeasurement, β, t)
    mul!(out, m.Λ, β)
    @inbounds for i in eachindex(out)
        out[i] = m.r_LB + _ekf_softplus(out[i] - m.r_LB)
    end
    return out
end

# Allocation-free analytic Jacobian: hand-rolled inner product avoids m.Λ * β.
function Siphon.measurement_jacobian!(Z, m::EKFTestSoftplusMeasurement, β, t)
    @inbounds for i in axes(m.Λ, 1)
        sh_i = zero(eltype(β))
        for k in axes(m.Λ, 2)
            sh_i += m.Λ[i, k] * β[k]
        end
        Si = _ekf_logistic(sh_i - m.r_LB)
        for j in axes(m.Λ, 2)
            Z[i, j] = Si * m.Λ[i, j]
        end
    end
    return Z
end

function Siphon.measurement_jacobian(m::EKFTestSoftplusMeasurement, β, t)
    Z = Matrix{eltype(β)}(undef, size(m.Λ, 1), size(m.Λ, 2))
    Siphon.measurement_jacobian!(Z, m, β, t)
    return Z
end

# Common small linear test setup
function _ekf_test_linear_setup(; m = 3, p = 2, n = 50, seed = 42)
    Random.seed!(seed)
    Z = randn(p, m)
    Tmat = 0.7 .* Matrix(I, m, m) .+ 0.05 .* randn(m, m)
    H = 0.1 .* Matrix(I, p, p)
    R = Matrix(I, m, m)
    Q = 0.05 .* Matrix(I, m, m)
    a1 = zeros(m)
    P1 = 1.0 .* Matrix(I, m, m)
    α = zeros(m, n)
    y = zeros(p, n)
    α[:, 1] = a1 .+ randn(m)
    for t in 1:n
        if t > 1
            α[:, t] = Tmat * α[:, t - 1] + 0.05 .* randn(m)
        end
        y[:, t] = Z * α[:, t] + 0.1 .* randn(p)
    end
    return (; m, p, n, Z, Tmat, H, R, Q, a1, P1, y)
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "EKF — EKFWorkspace checks the measurement protocol" begin
    M = [1.0;;]
    @test_throws "requires `measurement!(out, model, a, t)`" EKFWorkspace(
        EKFTestIncompleteMeasurement(), ADJacobian(), M, M, M, M, [0.0], [1.0;;], 10)
    @test_throws "requires `measurement!(out, model, a, t)`" EKFWorkspace(
        EKFTestIncompleteMeasurement(), AnalyticJacobian(), M, M, M, M, [0.0], [1.0;;], 10)
    @test_throws "requires `measurement!(out, model, a, t)`" EKFWorkspace(
        EKFTestIncompleteMeasurement(), FiniteDiffJacobian(), M, M, M, M, [0.0], [1.0;;], 10)
    @test_throws MethodError Siphon.measurement!(zeros(1), EKFTestIncompleteMeasurement(), [0.0], 1)
end

@testset "EKF — linear-measurement equivalence (functional)" begin
    s = _ekf_test_linear_setup()
    kfp = KFParms(s.Z, s.H, s.Tmat, s.R, s.Q)
    ll_kf = kalman_loglik(kfp, s.y, s.a1, s.P1)

    for mode in (AnalyticJacobian(), ADJacobian())
        ekf_p = EKFParms(LinearMeasurement(s.Z), mode, s.H, s.Tmat, s.R, s.Q)
        @test kalman_loglik(ekf_p, s.y, s.a1, s.P1) ≈ ll_kf rtol = 1e-12

        res_kf = kalman_filter(kfp, s.y, s.a1, s.P1)
        res_ekf = kalman_filter(ekf_p, s.y, s.a1, s.P1)
        @test res_ekf.loglik ≈ res_kf.loglik rtol = 1e-12
        @test maximum(abs, res_ekf.at .- res_kf.at) < 1e-12
        @test maximum(abs, res_ekf.att .- res_kf.att) < 1e-12
        @test maximum(abs, res_ekf.Pt .- res_kf.Pt) < 1e-12
        @test maximum(abs, res_ekf.Ptt .- res_kf.Ptt) < 1e-12
        @test maximum(abs, res_ekf.vt .- res_kf.vt) < 1e-12
        @test maximum(abs, res_ekf.Ft .- res_kf.Ft) < 1e-12
        @test maximum(abs, res_ekf.Kt .- res_kf.Kt) < 1e-12
        # EKF-specific fields populated correctly
        @test res_ekf.Zt[:, :, 1] ≈ s.Z
        @test res_ekf.yt[:, 1] ≈ s.Z * s.a1
    end
end

@testset "EKF — forecast" begin
    s = _ekf_test_linear_setup()
    kfp = KFParms(s.Z, s.H, s.Tmat, s.R, s.Q)
    fc_kf = forecast(kalman_filter(kfp, s.y, s.a1, s.P1), 3)
    for mode in (AnalyticJacobian(), ADJacobian())
        ekf_p = EKFParms(LinearMeasurement(s.Z), mode, s.H, s.Tmat, s.R, s.Q)
        fc = forecast(kalman_filter(ekf_p, s.y, s.a1, s.P1), 3)
        @test fc.yhat ≈ fc_kf.yhat && fc.F ≈ fc_kf.F && fc.a ≈ fc_kf.a && fc.P ≈ fc_kf.P
        @test forecast_periods(fc) == forecast_periods(fc_kf)
    end
end

@testset "EKF — linear-measurement equivalence (in-place)" begin
    s = _ekf_test_linear_setup()
    kfp = KFParms(s.Z, s.H, s.Tmat, s.R, s.Q)
    ll_kf = kalman_loglik(kfp, s.y, s.a1, s.P1)
    res_kf = kalman_filter(kfp, s.y, s.a1, s.P1)

    for mode in (AnalyticJacobian(), ADJacobian())
        ws = EKFWorkspace(LinearMeasurement(s.Z), mode, s.H, s.Tmat,
            s.R, s.Q, s.a1, s.P1, s.n)
        ll_ws = kalman_filter!(ws, s.y)
        @test ll_ws ≈ ll_kf rtol = 1e-10
        @test maximum(abs, ws.at .- res_kf.at) < 1e-10
        @test maximum(abs, ws.att .- res_kf.att) < 1e-10
        @test maximum(abs, ws.Ptt .- res_kf.Ptt) < 1e-10
        @test ws.Zt[:, :, 1] ≈ s.Z
    end
end

@testset "EKF — missing observations" begin
    s = _ekf_test_linear_setup()
    y_miss = copy(s.y)
    y_miss[:, 5] .= NaN
    y_miss[:, 17] .= NaN
    y_miss[:, end] .= NaN

    kfp = KFParms(s.Z, s.H, s.Tmat, s.R, s.Q)
    ll_kf = kalman_loglik(kfp, y_miss, s.a1, s.P1)

    ekf_p_an = EKFParms(
        LinearMeasurement(s.Z), AnalyticJacobian(), s.H, s.Tmat, s.R, s.Q)
    ekf_p_ad = EKFParms(LinearMeasurement(s.Z), ADJacobian(), s.H, s.Tmat, s.R, s.Q)
    @test kalman_loglik(ekf_p_an, y_miss, s.a1, s.P1) ≈ ll_kf rtol = 1e-12
    @test kalman_loglik(ekf_p_ad, y_miss, s.a1, s.P1) ≈ ll_kf rtol = 1e-12

    ws = EKFWorkspace(LinearMeasurement(s.Z), AnalyticJacobian(),
        s.H, s.Tmat, s.R, s.Q, s.a1, s.P1, s.n)
    ll_ws = kalman_filter!(ws, y_miss)
    @test ll_ws ≈ ll_kf rtol = 1e-10
    @test ws.missing_mask[5]
    @test ws.missing_mask[17]
    @test ws.missing_mask[end]
    # Filtered = predicted at missing time points
    @test ws.att[:, 5] ≈ ws.at[:, 5]
    @test ws.att[:, 17] ≈ ws.at[:, 17]
end

@testset "EKF — a row missing in every period is the model without it" begin
    # Under a nonlinear measurement, a row that is never observed must leave the
    # same filter as the measurement restricted to the other rows. Scattered NaNs
    # in the remaining rows make some periods partial in both models.
    Random.seed!(19)
    λ = 0.069
    maturities = [3.0, 12.0, 36.0, 60.0, 120.0]
    Λ = hcat(ones(length(maturities)),
        (1 .- exp.(-λ .* maturities)) ./ (λ .* maturities),
        (1 .- exp.(-λ .* maturities)) ./ (λ .* maturities) .- exp.(-λ .* maturities))
    m, p, n = 3, length(maturities), 60
    Tmat = Matrix(Diagonal([0.97, 0.95, 0.90]))
    H = (0.05^2) .* Matrix(I, p, p)
    H[3, 4] = H[4, 3] = 0.0005
    R = Matrix(I, m, m)
    Q = (0.10^2) .* Matrix(I, m, m)
    a1 = zeros(m)
    P1 = 0.5 .* Matrix(I, m, m)
    y = 0.5 .+ 0.3 .* randn(p, n)
    y[2, :] .= NaN
    y[4, 10:14] .= NaN
    y[[1, 3], 30] .= NaN
    y[:, 40] .= NaN

    keep = [1, 3, 4, 5]
    yr = y[keep, :]
    full = EKFTestSoftplusMeasurement(Λ, 0.0)
    reduced = EKFTestSoftplusMeasurement(Λ[keep, :], 0.0)

    for mode in (AnalyticJacobian(), ADJacobian())
        pf = EKFParms(full, mode, H, Tmat, R, Q)
        pr = EKFParms(reduced, mode, H[keep, keep], Tmat, R, Q)

        ll = kalman_loglik(pr, yr, a1, P1)
        @test kalman_loglik(pf, y, a1, P1) ≈ ll rtol = 1e-12

        ff, fr = kalman_filter(pf, y, a1, P1), kalman_filter(pr, yr, a1, P1)
        @test loglikelihood(ff) ≈ ll rtol = 1e-12
        @test filtered_states(ff) ≈ filtered_states(fr) rtol = 1e-10
        sf, sr = kalman_smoother(ff; crosscov = true), kalman_smoother(fr; crosscov = true)
        @test smoothed_states(sf) ≈ smoothed_states(sr) rtol = 1e-10
        @test smoothed_states_cov(sf) ≈ smoothed_states_cov(sr) rtol = 1e-10
        @test sf.Pcross ≈ sr.Pcross rtol = 1e-10

        wf = EKFWorkspace(pf, a1, P1, n)
        wr = EKFWorkspace(pr, a1, P1, n)
        @test kalman_filter!(wf, y) ≈ ll rtol = 1e-12
        kalman_filter!(wr, yr)
        kalman_smoother!(wf; crosscov = true)
        kalman_smoother!(wr; crosscov = true)
        @test wf.αs ≈ wr.αs rtol = 1e-10
        @test wf.Vs ≈ wr.Vs rtol = 1e-10
        @test wf.Pcross ≈ wr.Pcross rtol = 1e-10
        @test wf.αs ≈ smoothed_states(sf) rtol = 1e-10
    end
end

@testset "EKF — AD vs analytic Jacobian on smooth nonlinear measurement" begin
    Random.seed!(7)
    λ = 0.069
    maturities = [3.0, 6.0, 12.0, 24.0, 36.0, 60.0, 84.0, 120.0]
    Λ = hcat(ones(length(maturities)),
        (1 .- exp.(-λ .* maturities)) ./ (λ .* maturities),
        (1 .- exp.(-λ .* maturities)) ./ (λ .* maturities) .- exp.(-λ .* maturities))
    soft = EKFTestSoftplusMeasurement(Λ, 0.0)

    m = 3
    p = size(Λ, 1)
    n = 80
    Tmat = Matrix(Diagonal([0.97, 0.95, 0.90]))
    H = (0.05^2) .* Matrix(I, p, p)
    R = Matrix(I, m, m)
    Q = (0.10^2) .* Matrix(I, m, m)
    β0 = [3.0, -1.0, 0.5]
    P0 = 0.5 .* Matrix(I, m, m)

    β_true = zeros(m, n)
    y = zeros(p, n)
    β_true[:, 1] = β0 .+ randn(m)
    for t in 1:n
        if t > 1
            β_true[:, t] = Tmat * β_true[:, t - 1] + 0.05 .* randn(m)
        end
        y[:, t] = Siphon.measurement(soft, β_true[:, t], t) + 0.05 .* randn(p)
    end

    ekf_an = EKFParms(soft, AnalyticJacobian(), H, Tmat, R, Q)
    ekf_ad = EKFParms(soft, ADJacobian(), H, Tmat, R, Q)

    ll_an = kalman_loglik(ekf_an, y, β0, P0)
    ll_ad = kalman_loglik(ekf_ad, y, β0, P0)
    @test ll_an ≈ ll_ad rtol = 1e-12

    res_an = kalman_filter(ekf_an, y, β0, P0)
    res_ad = kalman_filter(ekf_ad, y, β0, P0)
    @test maximum(abs, res_an.Zt .- res_ad.Zt) < 1e-12
    @test maximum(abs, res_an.att .- res_ad.att) < 1e-10

    # In-place agrees with functional, both modes
    ws_an = EKFWorkspace(soft, AnalyticJacobian(), H, Tmat, R, Q, β0, P0, n)
    ws_ad = EKFWorkspace(soft, ADJacobian(), H, Tmat, R, Q, β0, P0, n)
    @test kalman_filter!(ws_an, y) ≈ ll_an rtol = 1e-10
    @test kalman_filter!(ws_ad, y) ≈ ll_ad rtol = 1e-10
    @test maximum(abs, ws_an.att .- res_an.att) < 1e-10
    @test maximum(abs, ws_ad.att .- res_ad.att) < 1e-10
end

@testset "EKF — workspace allocation budget" begin
    # In-place kalman_filter! should allocate effectively zero memory inside the
    # time loop after a warm-up call, in both AnalyticJacobian and ADJacobian
    # modes. We use a generously chunky panel and require a small fixed budget.
    Random.seed!(7)
    λ = 0.069
    maturities = [3.0, 6.0, 12.0, 24.0, 36.0, 60.0, 84.0, 120.0]
    Λ = hcat(ones(length(maturities)),
        (1 .- exp.(-λ .* maturities)) ./ (λ .* maturities),
        (1 .- exp.(-λ .* maturities)) ./ (λ .* maturities) .- exp.(-λ .* maturities))
    soft = EKFTestSoftplusMeasurement(Λ, 0.0)

    m = 3
    p = size(Λ, 1)
    n = 200
    Tmat = Matrix(Diagonal([0.97, 0.95, 0.90]))
    H = (0.05^2) .* Matrix(I, p, p)
    R = Matrix(I, m, m)
    Q = (0.10^2) .* Matrix(I, m, m)
    β0 = [3.0, -1.0, 0.5]
    P0 = 0.5 .* Matrix(I, m, m)
    y = randn(p, n)

    # Bound interpretation: the analytic path provides allocation-free user
    # methods, so we expect literally O(1) bytes total. The AD path goes through
    # ForwardDiff.jacobian!; in the standalone test process this is also ~16 B,
    # but inside `Pkg.test()` other testsets can invalidate ForwardDiff
    # specialisations and reintroduce ~48 B/step. We bound to a value well
    # below the n × p × m × sizeof(Float64) "fully allocating" baseline so any
    # genuine in-loop allocation regression is still caught.
    ws_an = EKFWorkspace(soft, AnalyticJacobian(), H, Tmat, R, Q, β0, P0, n)
    kalman_filter!(ws_an, y)  # warm-up
    allocs_an = @allocated kalman_filter!(ws_an, y)
    @test allocs_an < 256

    # Partially observed periods run in the leading blocks of the same buffers.
    y_partial = copy(y)
    y_partial[2, 1:2:n] .= NaN
    y_partial[[1, 5], 10:20] .= NaN
    kalman_filter!(ws_an, y_partial)  # warm-up
    @test @allocated(kalman_filter!(ws_an, y_partial)) < 256

    ws_ad = EKFWorkspace(soft, ADJacobian(), H, Tmat, R, Q, β0, P0, n)
    kalman_filter!(ws_ad, y)  # warm-up
    allocs_ad = @allocated kalman_filter!(ws_ad, y)
    # 48 bytes/step would be ~9.6 KB at n=200. A "naive" all-allocating impl
    # would allocate at least p*sizeof(Vector{Dual{Tag,Float64,3}}) per step
    # ≈ 200 × (16 + 8 × (1 + 3) × 8) = ~57 KB or more. Bound at 16 KB.
    @test allocs_ad < 16_000
end

@testset "EKF — smoother equivalence vs linear smoother" begin
    s = _ekf_test_linear_setup()
    # Filter + smoother on the linear baseline
    ws_kf = Siphon.KalmanWorkspace(s.Z, s.H, s.Tmat, s.R, s.Q, s.a1, s.P1, s.n)
    Siphon.kalman_filter!(ws_kf, s.y)
    Siphon.kalman_smoother!(ws_kf; crosscov = true)

    for mode in (AnalyticJacobian(), ADJacobian())
        ws = EKFWorkspace(LinearMeasurement(s.Z), mode, s.H, s.Tmat,
            s.R, s.Q, s.a1, s.P1, s.n)
        kalman_filter_and_smooth!(ws, s.y; crosscov = true)
        @test maximum(abs, ws.αs .- ws_kf.αs) < 1e-10
        @test maximum(abs, ws.Vs .- ws_kf.Vs) < 1e-10
        @test maximum(abs, ws.Pcross .- ws_kf.Pcross) < 1e-10
    end
end

@testset "EKF — smoother with missing observations" begin
    s = _ekf_test_linear_setup()
    y_miss = copy(s.y)
    y_miss[:, 7] .= NaN
    y_miss[:, 23] .= NaN

    ws_kf = Siphon.KalmanWorkspace(s.Z, s.H, s.Tmat, s.R, s.Q, s.a1, s.P1, s.n)
    Siphon.kalman_filter!(ws_kf, y_miss)
    Siphon.kalman_smoother!(ws_kf; crosscov = true)

    ws = EKFWorkspace(LinearMeasurement(s.Z), AnalyticJacobian(),
        s.H, s.Tmat, s.R, s.Q, s.a1, s.P1, s.n)
    kalman_filter_and_smooth!(ws, y_miss; crosscov = true)
    @test maximum(abs, ws.αs .- ws_kf.αs) < 1e-10
    @test maximum(abs, ws.Vs .- ws_kf.Vs) < 1e-10
    @test maximum(abs, ws.Pcross .- ws_kf.Pcross) < 1e-10
end

@testset "EKF — smoother on nonlinear measurement: PSD + tighter than filter" begin
    Random.seed!(11)
    λ = 0.069
    maturities = [3.0, 6.0, 12.0, 24.0, 36.0, 60.0, 84.0, 120.0]
    Λ = hcat(ones(length(maturities)),
        (1 .- exp.(-λ .* maturities)) ./ (λ .* maturities),
        (1 .- exp.(-λ .* maturities)) ./ (λ .* maturities) .- exp.(-λ .* maturities))
    soft = EKFTestSoftplusMeasurement(Λ, 0.0)
    m = 3
    p = size(Λ, 1)
    n = 120
    Tmat = Matrix(Diagonal([0.97, 0.95, 0.90]))
    H = (0.05^2) .* Matrix(I, p, p)
    R = Matrix(I, m, m)
    Q = (0.10^2) .* Matrix(I, m, m)
    β0 = [3.0, -1.0, 0.5]
    P0 = 0.5 .* Matrix(I, m, m)
    β_true = zeros(m, n)
    y = zeros(p, n)
    β_true[:, 1] = β0 .+ randn(m)
    for t in 1:n
        if t > 1
            β_true[:, t] = Tmat * β_true[:, t - 1] + 0.05 .* randn(m)
        end
        y[:, t] = Siphon.measurement(soft, β_true[:, t], t) + 0.05 .* randn(p)
    end

    ws = EKFWorkspace(soft, AnalyticJacobian(), H, Tmat, R, Q, β0, P0, n)
    kalman_filter_and_smooth!(ws, y; crosscov = true)

    # PSD smoothed covariances
    @test all(eigmin(Symmetric(ws.Vs[:, :, t])) > -1e-10 for t in 1:n)

    # In the interior, smoothed RMSE ≤ filtered RMSE
    filt_err = ws.att .- β_true
    smoo_err = ws.αs .- β_true
    for i in 1:m
        rmse_filt = sqrt(sum(filt_err[i, :] .^ 2) / n)
        rmse_smoo = sqrt(sum(smoo_err[i, :] .^ 2) / n)
        @test rmse_smoo <= rmse_filt + 1e-6
    end

    # Smoothed diag ≤ filtered diag in the interior
    for i in 1:m
        @test ws.Vs[i, i, 50] <= ws.Ptt[i, i, 50] + 1e-10
    end
end

# A three-factor yield curve observed through a softplus floor, simulated.
function _ekf_test_softplus_setup(; n = 80, seed = 11)
    rng = Random.Xoshiro(seed)
    λ = 0.069
    mats = [3.0, 12.0, 36.0, 60.0, 120.0]
    Λ = hcat(ones(length(mats)), (1 .- exp.(-λ .* mats)) ./ (λ .* mats),
        (1 .- exp.(-λ .* mats)) ./ (λ .* mats) .- exp.(-λ .* mats))
    soft = EKFTestSoftplusMeasurement(Λ, 0.0)
    m, p = 3, length(mats)
    Tmat = Matrix(Diagonal([0.97, 0.95, 0.90]))
    H = (0.05^2) .* Matrix(I, p, p)
    R = Matrix(1.0I, m, m)
    Q = (0.10^2) .* Matrix(I, m, m)
    a1 = [1.0, -1.0, 0.5]
    P1 = 0.5 .* Matrix(I, m, m)
    β = a1 .+ randn(rng, m)
    y = zeros(p, n)
    for t in 1:n
        t > 1 && (β = Tmat * β + 0.1 .* randn(rng, m))
        y[:, t] = Siphon.measurement(soft, β, t) + 0.05 .* randn(rng, p)
    end
    y[:, 9] .= NaN
    return (; soft, H, Tmat, R, Q, a1, P1, y, n)
end

@testset "EKF — functional smoother" begin
    s = _ekf_test_linear_setup()
    y = copy(s.y)
    y[:, 12] .= NaN
    lin = kalman_smoother(KFParms(s.Z, s.H, s.Tmat, s.R, s.Q), y, s.a1, s.P1;
        crosscov = true)
    for mode in (ADJacobian(), AnalyticJacobian(), FiniteDiffJacobian())
        p = EKFParms(LinearMeasurement(s.Z), mode, s.H, s.Tmat, s.R, s.Q)
        sm = kalman_smoother(p, y, s.a1, s.P1; crosscov = true)
        @test sm isa SmootherResult
        tol = mode isa FiniteDiffJacobian ? 1e-7 : 1e-12
        @test smoothed_states(sm) ≈ smoothed_states(lin) atol = tol
        @test smoothed_states_cov(sm) ≈ smoothed_states_cov(lin) atol = tol
        @test crosslag_covs(sm) ≈ crosslag_covs(lin) atol = tol
    end

    # On a nonlinear measurement, the functional and in-place smoothers agree.
    q = _ekf_test_softplus_setup()
    p = EKFParms(q.soft, AnalyticJacobian(), q.H, q.Tmat, q.R, q.Q)
    filt, sm = kalman_filter_and_smooth(p, q.y, q.a1, q.P1; crosscov = true)
    @test filt isa EKFFilterResult
    @test kalman_smoother(filt; crosscov = true).αs == sm.αs
    ws = EKFWorkspace(p, q.a1, q.P1, q.n)
    kalman_filter_and_smooth!(ws, q.y; crosscov = true)
    @test smoothed_states(sm) ≈ smoothed_states(ws) atol = 1e-10
    @test smoothed_states_cov(sm) ≈ smoothed_states_cov(ws) atol = 1e-10
    @test crosslag_covs(sm) ≈ crosslag_covs(ws) atol = 1e-10

    # AD through the smoother matches the linear smoother.
    f_ekf(θ) = sum(smoothed_states(kalman_smoother(
        EKFParms(LinearMeasurement(s.Z), s.H, θ[1] .* s.Tmat, s.R, s.Q), y, s.a1, s.P1)))
    f_lin(θ) = sum(smoothed_states(kalman_smoother(
        KFParms(s.Z, s.H, θ[1] .* s.Tmat, s.R, s.Q), y, s.a1, s.P1)))
    @test ForwardDiff.gradient(f_ekf, [1.0]) ≈ ForwardDiff.gradient(f_lin, [1.0])

    # A filter that stopped on a non-positive-definite F cannot be smoothed.
    bad = EKFParms(LinearMeasurement(s.Z), -s.H, s.Tmat, s.R, s.Q)
    @test_throws "not positive definite" kalman_smoother(bad, y, s.a1, zeros(3, 3))
end

@testset "EKF — FiniteDiffJacobian in EKFWorkspace" begin
    q = _ekf_test_softplus_setup()
    ws_ad = EKFWorkspace(q.soft, ADJacobian(), q.H, q.Tmat, q.R, q.Q, q.a1, q.P1, q.n)
    ws_fd = EKFWorkspace(q.soft, FiniteDiffJacobian(), q.H, q.Tmat, q.R, q.Q,
        q.a1, q.P1, q.n)
    ll_ad = kalman_filter_and_smooth!(ws_ad, q.y)
    ll_fd = kalman_filter_and_smooth!(ws_fd, q.y)
    @test ll_fd ≈ ll_ad rtol = 1e-8
    # Central differences with a √eps step carry rounding error near 1e-8.
    @test measurement_jacobians(ws_fd) ≈ measurement_jacobians(ws_ad) atol = 1e-6
    @test smoothed_states(ws_fd) ≈ smoothed_states(ws_ad) atol = 1e-6

    # The functional finite-difference path uses the same step: identical in the first
    # period, where both evaluate at a1.
    p_fd = EKFParms(q.soft, FiniteDiffJacobian(), q.H, q.Tmat, q.R, q.Q)
    Zt_func = measurement_jacobians(kalman_filter(p_fd, q.y, q.a1, q.P1))
    @test Zt_func[:, :, 1] == measurement_jacobians(ws_fd)[:, :, 1]
    @test Zt_func ≈ measurement_jacobians(ws_fd) atol = 1e-6

    kalman_filter!(ws_fd, q.y)
    @test (@allocated kalman_filter!(ws_fd, q.y)) == 0
end

@testset "EKF — filter+smoother allocation budget" begin
    Random.seed!(7)
    λ = 0.069
    maturities = [3.0, 6.0, 12.0, 24.0, 36.0, 60.0, 84.0, 120.0]
    Λ = hcat(ones(length(maturities)),
        (1 .- exp.(-λ .* maturities)) ./ (λ .* maturities),
        (1 .- exp.(-λ .* maturities)) ./ (λ .* maturities) .- exp.(-λ .* maturities))
    soft = EKFTestSoftplusMeasurement(Λ, 0.0)
    m = 3
    p = size(Λ, 1)
    n = 200
    Tmat = Matrix(Diagonal([0.97, 0.95, 0.90]))
    H = (0.05^2) .* Matrix(I, p, p)
    R = Matrix(I, m, m)
    Q = (0.10^2) .* Matrix(I, m, m)
    β0 = [3.0, -1.0, 0.5]
    P0 = 0.5 .* Matrix(I, m, m)
    y = randn(p, n)

    ws = EKFWorkspace(soft, AnalyticJacobian(), H, Tmat, R, Q, β0, P0, n)
    kalman_filter_and_smooth!(ws, y)
    allocs = @allocated kalman_filter_and_smooth!(ws, y)
    # filter+smoother adds the smoother's lag-1 cross-cov path which uses
    # Cholesky on Pt_t — at the moment we accept a small fixed budget; the
    # bulk of the allocation budget is the same as `kalman_filter!`.
    @test allocs < 256
end

@testset "custom_ekf checks the measurement length against H" begin
    s = _ekf_test_linear_setup()
    kw = (measurement = LinearMeasurement(s.Z), T = I, R = I,
        Q = diag_free(s.m, :var_q), a1 = zeros(s.m), P1 = 10.0I)
    spec = custom_ekf(; kw..., H = diag_free(s.p, :var_h))
    @test spec isa AbstractSSMSpec
    @test spec.n_obs == s.p
    @test_throws "the measurement returns $(s.p) values but H is" custom_ekf(; kw...,
        H = diag_free(s.p + 1, :var_h))
    @test_throws "cannot size `I`" custom_ekf(; kw..., H = I)
end

@testset "EKF DSL — EKFSpec introspection and codegen" begin
    s = _ekf_test_linear_setup()

    # Build an EKFSpec using diag_free (variance parameterisation).
    spec = custom_ekf(
        measurement = LinearMeasurement(s.Z),
        jacobian_mode = AnalyticJacobian(),
        H = diag_free(s.p, :σh; init = 0.04, lower = 1e-8),
        T = diag_free(s.m, :ψ; init = 0.5, lower = -0.99, upper = 0.99),
        R = Matrix(1.0I, s.m, s.m),
        Q = diag_free(s.m, :σq; init = 0.01, lower = 1e-8),
        a1 = zeros(s.m),
        P1 = 10.0 .* Matrix(I, s.m, s.m)
    )
    @test n_params(spec) == s.p + s.m + s.m
    @test param_names(spec) ==
          [Symbol("σh_$i") for i in 1:s.p] ∪
          [Symbol("ψ_$i") for i in 1:s.m] ∪
          [Symbol("σq_$i") for i in 1:s.m]
    @test length(initial_values(spec)) == n_params(spec)

    # build_ekfparms returns an EKFParms whose system matrices match diag_free.
    θ_nt = (; (p.name => p.init for p in spec.params)...)
    p_built = build_ekfparms(spec, θ_nt)
    @test p_built isa EKFParms
    @test diag(p_built.H) ≈ fill(0.04, s.p)
    @test diag(p_built.T) ≈ fill(0.5, s.m)
    @test diag(p_built.Q) ≈ fill(0.01, s.m)

    # ssm_loglik(spec, θ, y) matches manual EKFParms construction
    a1, P1 = Siphon.build_initial_state(spec, θ_nt)
    ll_spec = ssm_loglik(spec, θ_nt, s.y)
    ll_manual = Siphon.kalman_loglik(p_built, s.y, a1, P1)
    @test ll_spec ≈ ll_manual rtol = 1e-12
end

@testset "EKF DSL — optimize_ssm on EKFSpec vs SSMSpec for a linear model" begin
    s = _ekf_test_linear_setup()

    spec_ssm = custom_ssm(
        Z = s.Z,
        H = diag_free(s.p, :σh; init = 0.04, lower = 1e-8),
        T = diag_free(s.m, :ψ; init = 0.5, lower = -0.99, upper = 0.99),
        R = Matrix(1.0I, s.m, s.m),
        Q = diag_free(s.m, :σq; init = 0.01, lower = 1e-8),
        a1 = zeros(s.m),
        P1 = 10.0 .* Matrix(I, s.m, s.m)
    )
    res_ssm = optimize_ssm(spec_ssm, s.y; show_trace = false)

    spec_ekf = custom_ekf(
        measurement = LinearMeasurement(s.Z),
        jacobian_mode = AnalyticJacobian(),
        H = diag_free(s.p, :σh; init = 0.04, lower = 1e-8),
        T = diag_free(s.m, :ψ; init = 0.5, lower = -0.99, upper = 0.99),
        R = Matrix(1.0I, s.m, s.m),
        Q = diag_free(s.m, :σq; init = 0.01, lower = 1e-8),
        a1 = zeros(s.m),
        P1 = 10.0 .* Matrix(I, s.m, s.m)
    )
    res_ekf = optimize_ssm(spec_ekf, s.y; show_trace = false)

    @test res_ssm.converged
    @test res_ekf.converged
    @test res_ssm.loglik ≈ res_ekf.loglik atol = 1e-6
    for k in keys(res_ssm.θ)
        @test getproperty(res_ssm.θ, k) ≈ getproperty(res_ekf.θ, k) rtol = 1e-4
    end
end

@testset "EKF DSL — SSMLogDensity with a prior matches the linear spec" begin
    s = _ekf_test_linear_setup()
    kw = (H = diag_free(s.p, :σh; init = 0.04, lower = 1e-8),
        T = diag_free(s.m, :ψ; init = 0.5, lower = -0.99, upper = 0.99),
        R = Matrix(1.0I, s.m, s.m),
        Q = diag_free(s.m, :σq; init = 0.01, lower = 1e-8),
        a1 = zeros(s.m),
        P1 = 10.0 .* Matrix(I, s.m, s.m))
    spec_ssm = custom_ssm(; Z = s.Z, kw...)
    spec_ekf = custom_ekf(; measurement = LinearMeasurement(s.Z),
        jacobian_mode = AnalyticJacobian(), kw...)
    prior = ParameterPriors(σh_1 = InverseGammaPrior(2.0, 0.1), ψ_1 = NormalPrior(0.5, 0.2))

    ld_ssm = SSMLogDensity(spec_ssm, s.y; prior)
    ld_ekf = SSMLogDensity(spec_ekf, s.y; prior)
    @test ld_ekf isa SSMLogDensity
    θ_u = transform_to_unconstrained(spec_ekf, initial_values(spec_ekf)) .+ 0.1
    @test logdensity(ld_ekf, θ_u) ≈ logdensity(ld_ssm, θ_u) rtol = 1e-10
    @test logdensity(ld_ekf, θ_u) - logdensity(SSMLogDensity(spec_ekf, s.y), θ_u) ≈
          prior(transform_to_constrained(spec_ekf, θ_u)[1])
    @test_throws "does not estimate" SSMLogDensity(spec_ekf, s.y;
        prior = ParameterPriors(var_obs = NormalPrior(0.0, 1.0)))
end

@testset "EKF DSL — StateSpaceModel matches the linear spec" begin
    s = _ekf_test_linear_setup()
    kw = (H = diag_free(s.p, :σh; init = 0.04, lower = 1e-8),
        T = diag_free(s.m, :ψ; init = 0.5, lower = -0.99, upper = 0.99),
        R = Matrix(1.0I, s.m, s.m),
        Q = diag_free(s.m, :σq; init = 0.01, lower = 1e-8),
        a1 = zeros(s.m),
        P1 = 10.0 .* Matrix(I, s.m, s.m))
    spec_ssm = custom_ssm(; Z = s.Z, kw...)
    spec_ekf = custom_ekf(; measurement = LinearMeasurement(s.Z), kw...)
    n = size(s.y, 2)
    y = copy(s.y)
    y[:, 10:12] .= NaN
    y[1, 20:25] .= NaN

    θ = (σh_1 = 0.1, σh_2 = 0.12, ψ_1 = 0.7, ψ_2 = 0.65, ψ_3 = 0.72,
        σq_1 = 0.05, σq_2 = 0.04, σq_3 = 0.06)
    ml = StateSpaceModel(spec_ssm, θ, n)
    me = StateSpaceModel(spec_ekf, θ, n)
    @test me isa StateSpaceModel{Float64, typeof(spec_ekf)}
    @test kalman_filter!(me, y) ≈ kalman_filter!(ml, y) rtol = 1e-12
    @test kalman_loglik(me, y) ≈ kalman_loglik(ml, y) rtol = 1e-12
    @test filtered_states(me) ≈ filtered_states(ml) rtol = 1e-10
    pe, pl = prediction_errors(me), prediction_errors(ml)
    @test isnan.(pe) == isnan.(pl)
    @test filter(!isnan, pe) ≈ filter(!isnan, pl) rtol = 1e-10
    @test Siphon.observed_mask(me) == Siphon.observed_mask(ml)
    kalman_smoother!(me)
    @test smoothed_states(me) ≈ smoothed_states(ml) rtol = 1e-10
    @test smoothed_states_cov(me) ≈ smoothed_states_cov(ml) rtol = 1e-10
    fe, fl = forecast(me, 4), forecast(ml, 4)
    @test forecast_observations(fe) ≈ forecast_observations(fl) rtol = 1e-10
    @test forecast_observations_cov(fe) ≈ forecast_observations_cov(fl) rtol = 1e-10
    @test system_matrices(me) isa EKFParms
    @test obs_cov(me) ≈ obs_cov(ml)
    @test transition_matrix(me) ≈ transition_matrix(ml)

    fit_l = fit!(MLE(), StateSpaceModel(spec_ssm, n), s.y; static = :off)
    fit_e = fit!(MLE(), StateSpaceModel(spec_ekf, n), s.y)
    @test isconverged(fit_e)
    @test loglikelihood(fit_e) ≈ loglikelihood(fit_l) rtol = 1e-6
    @test coef(fit_e) ≈ coef(fit_l) rtol = 1e-3
    @test coefnames(fit_e) == coefnames(fit_l)
    @test dof(fit_e) == dof(fit_l)
    @test vcov(fit_e) ≈ vcov(fit_l; static = :off) rtol = 1e-2
    @test aic(fit_e) ≈ aic(fit_l) rtol = 1e-6
end

@testset "EKF DSL — StateSpaceModel on a nonlinear measurement" begin
    Random.seed!(7)
    m, p, n = 2, 3, 60
    Λ = [1.0 0.0; 1.0 0.5; 1.0 1.0]
    meas = EKFTestSoftplusMeasurement(Λ, 0.0)
    spec = custom_ekf(; measurement = meas,
        H = diag_free(p, :var_obs; init = 0.01, lower = 0.0),
        T = diag_free(m, :ψ; init = 0.8, lower = -0.99, upper = 0.99),
        R = Matrix(1.0I, m, m),
        Q = diag_free(m, :var_q; init = 0.05, lower = 0.0),
        a1 = zeros(m),
        P1 = Matrix(1.0I, m, m))
    α = zeros(m, n)
    y = zeros(p, n)
    for t in 1:n
        α[:, t] = t == 1 ? 0.3 .* randn(m) : 0.9 .* α[:, t - 1] .+ 0.2 .* randn(m)
        y[:, t] = Siphon.measurement(meas, α[:, t], t) .+ 0.1 .* randn(p)
    end
    θ = (var_obs_1 = 0.01, var_obs_2 = 0.01, var_obs_3 = 0.01,
        ψ_1 = 0.9, ψ_2 = 0.9, var_q_1 = 0.04, var_q_2 = 0.04)
    model = StateSpaceModel(spec, θ, n)
    ll = kalman_filter!(model, y)

    ss = Siphon.build_nonlinear_state_space(spec, θ, y)
    filt, sm = kalman_filter_and_smooth(ss.p, y, ss.a1, ss.P1)
    @test ll ≈ loglikelihood(filt)
    @test filtered_states(model) ≈ filtered_states(filt)
    @test smoothed_states(model) ≈ smoothed_states(sm)
    @test smoothed_states_cov(model) ≈ smoothed_states_cov(sm)
    fc = forecast(model, 3)
    @test forecast_observations(fc) ≈ forecast_observations(forecast(filt, 3))
    @test SmootherResult(model) isa SmootherResult

    fitted = fit!(MLE(), StateSpaceModel(spec, n), y)
    @test isfinite(loglikelihood(fitted))
    @test loglikelihood(fitted) >= ll - 1e-8
    @test all(isfinite, stderror(fitted))
end

@testset "EKF DSL — StateSpaceModel rejects linear-only operations" begin
    s = _ekf_test_linear_setup()
    spec = custom_ekf(; measurement = LinearMeasurement(s.Z),
        H = diag_free(s.p, :σh; init = 0.04, lower = 1e-8),
        T = 0.7 .* Matrix(1.0I, s.m, s.m), R = Matrix(1.0I, s.m, s.m),
        Q = 0.05 .* Matrix(1.0I, s.m, s.m), a1 = zeros(s.m),
        P1 = Matrix(1.0I, s.m, s.m))
    n = size(s.y, 2)
    @test_throws "does not support EKF models" fit!(EM(), StateSpaceModel(spec, n), s.y)
    @test_throws "static = :on is not available for an EKF model" fit!(
        MLE(), StateSpaceModel(spec, n), s.y; static = :on)
    model = StateSpaceModel(spec, (σh_1 = 0.1, σh_2 = 0.1), n)
    @test_throws "Filter not run. Call fit!(MLE(), model, y), or kalman_filter!" smoothed_states(model)
    @test_throws "Model not fitted. Call fit!(MLE(), model, y) first." coef(
        StateSpaceModel(spec, n))
    kalman_filter!(model, s.y)
    @test_throws "an EKF model has no observation matrix" obs_matrix(model)
    @test_throws "EKF models are not supported" disturbance_smoother(model)
    @test_throws "static must be :auto, :on or :off" kalman_loglik(model, s.y; static = :x)
end

@testset "EKF DSL — MeasurementExpr nonlinear MLE recovers parameters" begin
    Random.seed!(99)
    λ = 0.069
    maturities = [3.0, 6.0, 12.0, 24.0, 36.0, 60.0, 84.0, 120.0]
    Λ = hcat(ones(length(maturities)),
        (1 .- exp.(-λ .* maturities)) ./ (λ .* maturities),
        (1 .- exp.(-λ .* maturities)) ./ (λ .* maturities) .- exp.(-λ .* maturities))

    # Centered softplus measurement, two type params so AD can promote μ.
    @eval struct EKFTestCenteredSP{LT <: AbstractMatrix, T1 <: Real, T2 <: Real} <:
                 Siphon.AbstractEKFMeasurement
        Λ::LT
        r_LB::T1
        μ::Vector{T2}
    end
    @eval function Siphon.measurement(m::EKFTestCenteredSP, ξ, t)
        sh = m.Λ * (ξ .+ m.μ)
        return m.r_LB .+ _ekf_softplus.(sh .- m.r_LB)
    end
    @eval function Siphon.measurement!(out, m::EKFTestCenteredSP, ξ, t)
        @inbounds for i in axes(m.Λ, 1)
            s = zero(eltype(ξ))
            for k in axes(m.Λ, 2)
                s += m.Λ[i, k] * (ξ[k] + m.μ[k])
            end
            out[i] = m.r_LB + _ekf_softplus(s - m.r_LB)
        end
        return out
    end
    @eval function Siphon.measurement_jacobian!(Z, m::EKFTestCenteredSP, ξ, t)
        @inbounds for i in axes(m.Λ, 1)
            s = zero(eltype(ξ))
            for k in axes(m.Λ, 2)
                s += m.Λ[i, k] * (ξ[k] + m.μ[k])
            end
            Si = _ekf_logistic(s - m.r_LB)
            for j in axes(m.Λ, 2)
                Z[i, j] = Si * m.Λ[i, j]
            end
        end
        return Z
    end
    @eval function Siphon.measurement_jacobian(m::EKFTestCenteredSP, ξ, t)
        Z = Matrix{eltype(ξ)}(undef, size(m.Λ, 1), size(m.Λ, 2))
        Siphon.measurement_jacobian!(Z, m, ξ, t)
        return Z
    end

    m_state = 3
    p_obs = size(Λ, 1)
    n = 250
    μ_true = [3.5, -1.0, 0.5]
    Ψ_true = Matrix(Diagonal([0.97, 0.95, 0.90]))
    Q_true = Matrix(Diagonal([0.10, 0.15, 0.20] .^ 2))
    σh_true = 0.05

    α_true = zeros(m_state, n)
    y = zeros(p_obs, n)
    α_true[:, 1] = μ_true .+ randn(m_state)
    for t in 1:n
        if t > 1
            α_true[:, t] = μ_true + Ψ_true * (α_true[:, t - 1] - μ_true) +
                           cholesky(Q_true).L * randn(m_state)
        end
        ξ = α_true[:, t] .- μ_true
        yhat = Siphon.measurement(EKFTestCenteredSP(Λ, 0.0, μ_true), ξ, t)
        y[:, t] = yhat .+ σh_true .* randn(p_obs)
    end

    mexpr = MeasurementExpr(
        params = [Siphon.SSMParameter(:μ_L; init = 3.5),
            Siphon.SSMParameter(:μ_S; init = -1.0),
            Siphon.SSMParameter(:μ_C; init = 0.5)],
        builder = (
            θ, data) -> EKFTestCenteredSP(data.Λ, data.r_LB,
            [θ[:μ_L], θ[:μ_S], θ[:μ_C]]),
        data = (Λ = Λ, r_LB = 0.0)
    )

    spec = custom_ekf(
        measurement = mexpr,
        jacobian_mode = AnalyticJacobian(),
        H = diag_free(p_obs, :σh2; init = σh_true^2, lower = 1e-10),
        T = diag_free(m_state, :ψ; init = 0.9, lower = -0.999, upper = 0.999),
        R = Matrix(1.0I, m_state, m_state),
        Q = diag_free(m_state, :σq2; init = 0.04, lower = 1e-10),
        a1 = zeros(m_state),
        P1 = 1.0 .* Matrix(I, m_state, m_state)
    )

    res = optimize_ssm(spec, y; maxiters = 400)
    @test res.converged

    # Loose finite-sample tolerances — this is a smoke test that the optimizer
    # converges to a sensible neighbourhood of truth, not an identification test.
    @test res.θ.μ_L ≈ μ_true[1] atol = 0.5
    @test res.θ.μ_S ≈ μ_true[2] atol = 0.5
    @test res.θ.μ_C ≈ μ_true[3] atol = 0.5
    @test res.θ.ψ_1 ≈ Ψ_true[1, 1] atol = 0.15
    @test res.θ.ψ_2 ≈ Ψ_true[2, 2] atol = 0.15
    @test res.θ.ψ_3 ≈ Ψ_true[3, 3] atol = 0.15
end

@testset "EKF — parameter AD through kalman_loglik" begin
    s = _ekf_test_linear_setup()

    function neg_ll(σ)
        H_local = σ^2 .* Matrix(I, s.p, s.p)
        ekf_p = EKFParms(LinearMeasurement(s.Z), ADJacobian(),
            H_local, s.Tmat, s.R, s.Q)
        return -kalman_loglik(ekf_p, s.y, s.a1, s.P1)
    end

    σ0 = 0.5
    g_ad = ForwardDiff.derivative(neg_ll, σ0)
    δ = 1e-5
    g_fd = (neg_ll(σ0 + δ) - neg_ll(σ0 - δ)) / (2δ)

    @test isfinite(g_ad)
    @test g_ad ≈ g_fd rtol = 1e-6

    # With partially observed periods the derivative matches the linear filter's.
    y = copy(s.y)
    y[1, 5:8] .= NaN
    y[2, 20] = NaN
    function neg_ll_partial(σ, linear)
        H_local = σ^2 .* Matrix(I, s.p, s.p)
        parms = linear ? KFParms(s.Z, H_local, s.Tmat, s.R, s.Q) :
                EKFParms(LinearMeasurement(s.Z), ADJacobian(), H_local, s.Tmat, s.R, s.Q)
        return -kalman_loglik(parms, y, s.a1, s.P1)
    end
    @test ForwardDiff.derivative(σ -> neg_ll_partial(σ, false), σ0) ≈
          ForwardDiff.derivative(σ -> neg_ll_partial(σ, true), σ0) rtol = 1e-10
end
