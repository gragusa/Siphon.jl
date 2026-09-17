# EKF extension tests.
#
# Coverage (matches Plan §15):
#   - Linear-equivalence: under a LinearMeasurement, ekf_loglik / ekf_filter /
#     EKFWorkspace + ekf_filter! must match kalman_loglik / kalman_filter exactly.
#   - Missing-data handling.
#   - AD vs analytic Jacobian agreement on a smooth nonlinear measurement.
#   - Workspace allocation smoke-test (zero in-loop allocations after warm-up).
#   - Parameter-AD: ForwardDiff through ekf_loglik wrt system parameters.

using Siphon
using Test
using LinearAlgebra
using Random
using ForwardDiff

# ---------------------------------------------------------------------------
# Helpers — measurement structs used across testsets
# ---------------------------------------------------------------------------

struct EKFTestLinearMeasurement{ZT <: AbstractMatrix} <: Siphon.AbstractEKFMeasurement
    Z::ZT
end

Siphon.measurement(m::EKFTestLinearMeasurement, a, t) = m.Z * a
Siphon.measurement!(out, m::EKFTestLinearMeasurement, a, t) = (mul!(out, m.Z, a); out)
Siphon.measurement_jacobian(m::EKFTestLinearMeasurement, a, t) = m.Z
Siphon.measurement_jacobian!(Z, m::EKFTestLinearMeasurement, a, t) = (copyto!(Z, m.Z); Z)

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

@testset "EKF — linear-measurement equivalence (functional)" begin
    s = _ekf_test_linear_setup()
    kfp = KFParms(s.Z, s.H, s.Tmat, s.R, s.Q)
    ll_kf = kalman_loglik(kfp, s.y, s.a1, s.P1)

    for mode in (AnalyticJacobian(), ADJacobian())
        ekf_p = EKFParms(EKFTestLinearMeasurement(s.Z), mode, s.H, s.Tmat, s.R, s.Q)
        @test ekf_loglik(ekf_p, s.y, s.a1, s.P1) ≈ ll_kf rtol = 1e-12

        res_kf = kalman_filter(kfp, s.y, s.a1, s.P1)
        res_ekf = ekf_filter(ekf_p, s.y, s.a1, s.P1)
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

@testset "EKF — linear-measurement equivalence (in-place)" begin
    s = _ekf_test_linear_setup()
    kfp = KFParms(s.Z, s.H, s.Tmat, s.R, s.Q)
    ll_kf = kalman_loglik(kfp, s.y, s.a1, s.P1)
    res_kf = kalman_filter(kfp, s.y, s.a1, s.P1)

    for mode in (AnalyticJacobian(), ADJacobian())
        ws = EKFWorkspace(EKFTestLinearMeasurement(s.Z), mode, s.H, s.Tmat,
            s.R, s.Q, s.a1, s.P1, s.n)
        ll_ws = ekf_filter!(ws, s.y)
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
        EKFTestLinearMeasurement(s.Z), AnalyticJacobian(), s.H, s.Tmat, s.R, s.Q)
    ekf_p_ad = EKFParms(EKFTestLinearMeasurement(s.Z), ADJacobian(), s.H, s.Tmat, s.R, s.Q)
    @test ekf_loglik(ekf_p_an, y_miss, s.a1, s.P1) ≈ ll_kf rtol = 1e-12
    @test ekf_loglik(ekf_p_ad, y_miss, s.a1, s.P1) ≈ ll_kf rtol = 1e-12

    ws = EKFWorkspace(EKFTestLinearMeasurement(s.Z), AnalyticJacobian(),
        s.H, s.Tmat, s.R, s.Q, s.a1, s.P1, s.n)
    ll_ws = ekf_filter!(ws, y_miss)
    @test ll_ws ≈ ll_kf rtol = 1e-10
    @test ws.missing_mask[5]
    @test ws.missing_mask[17]
    @test ws.missing_mask[end]
    # Filtered = predicted at missing time points
    @test ws.att[:, 5] ≈ ws.at[:, 5]
    @test ws.att[:, 17] ≈ ws.at[:, 17]
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

    ll_an = ekf_loglik(ekf_an, y, β0, P0)
    ll_ad = ekf_loglik(ekf_ad, y, β0, P0)
    @test ll_an ≈ ll_ad rtol = 1e-12

    res_an = ekf_filter(ekf_an, y, β0, P0)
    res_ad = ekf_filter(ekf_ad, y, β0, P0)
    @test maximum(abs, res_an.Zt .- res_ad.Zt) < 1e-12
    @test maximum(abs, res_an.att .- res_ad.att) < 1e-10

    # In-place agrees with functional, both modes
    ws_an = EKFWorkspace(soft, AnalyticJacobian(), H, Tmat, R, Q, β0, P0, n)
    ws_ad = EKFWorkspace(soft, ADJacobian(), H, Tmat, R, Q, β0, P0, n)
    @test ekf_filter!(ws_an, y) ≈ ll_an rtol = 1e-10
    @test ekf_filter!(ws_ad, y) ≈ ll_ad rtol = 1e-10
    @test maximum(abs, ws_an.att .- res_an.att) < 1e-10
    @test maximum(abs, ws_ad.att .- res_ad.att) < 1e-10
end

@testset "EKF — workspace allocation budget" begin
    # In-place ekf_filter! should allocate effectively zero memory inside the
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
    ekf_filter!(ws_an, y)  # warm-up
    allocs_an = @allocated ekf_filter!(ws_an, y)
    @test allocs_an < 256

    ws_ad = EKFWorkspace(soft, ADJacobian(), H, Tmat, R, Q, β0, P0, n)
    ekf_filter!(ws_ad, y)  # warm-up
    allocs_ad = @allocated ekf_filter!(ws_ad, y)
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
        ws = EKFWorkspace(EKFTestLinearMeasurement(s.Z), mode, s.H, s.Tmat,
            s.R, s.Q, s.a1, s.P1, s.n)
        ekf_filter_and_smooth!(ws, s.y; crosscov = true)
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

    ws = EKFWorkspace(EKFTestLinearMeasurement(s.Z), AnalyticJacobian(),
        s.H, s.Tmat, s.R, s.Q, s.a1, s.P1, s.n)
    ekf_filter_and_smooth!(ws, y_miss; crosscov = true)
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
    ekf_filter_and_smooth!(ws, y; crosscov = true)

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
    ekf_filter_and_smooth!(ws, y)
    allocs = @allocated ekf_filter_and_smooth!(ws, y)
    # filter+smoother adds the smoother's lag-1 cross-cov path which uses
    # Cholesky on Pt_t — at the moment we accept a small fixed budget; the
    # bulk of the allocation budget is the same as `ekf_filter!`.
    @test allocs < 256
end

@testset "EKF DSL — EKFSpec introspection and codegen" begin
    s = _ekf_test_linear_setup()

    # Build an EKFSpec using diag_free (variance parameterisation).
    spec = custom_ekf(
        measurement = EKFTestLinearMeasurement(s.Z),
        jacobian_mode = AnalyticJacobian(),
        H = diag_free(s.p, :σh; init = 0.04, lower = 1e-8),
        T = diag_free(s.m, :ψ; init = 0.5, lower = -0.99, upper = 0.99),
        R = identity_mat(s.m),
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

    # ekf_loglik(spec, θ, y) matches manual EKFParms construction
    a1, P1 = Siphon.build_initial_state(spec, θ_nt)
    ll_spec = Siphon.ekf_loglik(spec, θ_nt, s.y)
    ll_manual = Siphon.ekf_loglik(p_built, s.y, a1, P1)
    @test ll_spec ≈ ll_manual rtol = 1e-12
end

@testset "EKF DSL — optimize_ekf vs optimize_ssm on a linear model" begin
    s = _ekf_test_linear_setup()

    spec_ssm = custom_ssm(
        Z = s.Z,
        H = diag_free(s.p, :σh; init = 0.04, lower = 1e-8),
        T = diag_free(s.m, :ψ; init = 0.5, lower = -0.99, upper = 0.99),
        R = identity_mat(s.m),
        Q = diag_free(s.m, :σq; init = 0.01, lower = 1e-8),
        a1 = zeros(s.m),
        P1 = 10.0 .* Matrix(I, s.m, s.m)
    )
    res_ssm = optimize_ssm(spec_ssm, s.y; show_trace = false)

    spec_ekf = custom_ekf(
        measurement = EKFTestLinearMeasurement(s.Z),
        jacobian_mode = AnalyticJacobian(),
        H = diag_free(s.p, :σh; init = 0.04, lower = 1e-8),
        T = diag_free(s.m, :ψ; init = 0.5, lower = -0.99, upper = 0.99),
        R = identity_mat(s.m),
        Q = diag_free(s.m, :σq; init = 0.01, lower = 1e-8),
        a1 = zeros(s.m),
        P1 = 10.0 .* Matrix(I, s.m, s.m)
    )
    res_ekf = optimize_ekf(spec_ekf, s.y; show_trace = false)

    @test res_ssm.converged
    @test res_ekf.converged
    @test res_ssm.loglik ≈ res_ekf.loglik atol = 1e-6
    for k in keys(res_ssm.θ)
        @test getproperty(res_ssm.θ, k) ≈ getproperty(res_ekf.θ, k) rtol = 1e-4
    end
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
        R = identity_mat(m_state),
        Q = diag_free(m_state, :σq2; init = 0.04, lower = 1e-10),
        a1 = zeros(m_state),
        P1 = 1.0 .* Matrix(I, m_state, m_state)
    )

    res = optimize_ekf(spec, y; maxiters = 400)
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

@testset "EKF — parameter AD through ekf_loglik" begin
    s = _ekf_test_linear_setup()

    function neg_ll(σ)
        H_local = σ^2 .* Matrix(I, s.p, s.p)
        ekf_p = EKFParms(EKFTestLinearMeasurement(s.Z), ADJacobian(),
            H_local, s.Tmat, s.R, s.Q)
        return -ekf_loglik(ekf_p, s.y, s.a1, s.P1)
    end

    σ0 = 0.5
    g_ad = ForwardDiff.derivative(neg_ll, σ0)
    δ = 1e-5
    g_fd = (neg_ll(σ0 + δ) - neg_ll(σ0 - δ)) / (2δ)

    @test isfinite(g_ad)
    @test g_ad ≈ g_fd rtol = 1e-6
end
