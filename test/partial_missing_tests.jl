# Per-element missing observations: a period in which only some rows are present
# must use those rows rather than discarding the whole observation vector.

using Test
using Siphon
using LinearAlgebra
using Random

"""
    _reference_filter_smoother(Z, H, T, R, Q, a1, P1, y)

A deliberately plain Kalman filter and RTS smoother that forms the reduced system
`(Z[rows,:], H[rows,rows], y[rows,t])` for each period's observed rows. It shares
no code with the implementation under test, so agreement is evidence rather than
tautology.
"""
function _reference_filter_smoother(Z, H, T, R, Q, a1, P1, y)
    p, n = size(y)
    m = length(a1)
    a_pred = zeros(m, n)
    P_pred = zeros(m, m, n)
    a_filt = zeros(m, n)
    P_filt = zeros(m, m, n)
    a = collect(float.(a1))
    P = collect(float.(P1))
    loglik = 0.0

    for t in 1:n
        a_pred[:, t] = a
        P_pred[:, :, t] = P
        y_t = view(y, :, t)
        rows = [i for i in 1:p if !isnan(y_t[i])]
        if isempty(rows)
            a_filt[:, t] = a
            P_filt[:, :, t] = P
        else
            Zr = Z[rows, :]
            Hr = H[rows, rows]
            v = y_t[rows] - Zr * a
            F = Symmetric(Zr * P * Zr' + Hr)
            loglik += -0.5 * (logdet(F) + dot(v, F \ v) + length(rows) * log(2π))
            K = P * Zr' / F
            a_filt[:, t] = a + K * v
            P_filt[:, :, t] = P - K * Zr * P
        end
        a = T * a_filt[:, t]
        P = T * P_filt[:, :, t] * T' + R * Q * R'
    end

    α = zeros(m, n)
    V = zeros(m, m, n)
    α[:, n] = a_filt[:, n]
    V[:, :, n] = P_filt[:, :, n]
    for t in (n - 1):-1:1
        J = P_filt[:, :, t] * T' / P_pred[:, :, t + 1]
        α[:, t] = a_filt[:, t] + J * (α[:, t + 1] - a_pred[:, t + 1])
        V[:, :, t] = P_filt[:, :, t] + J * (V[:, :, t + 1] - P_pred[:, :, t + 1]) * J'
    end

    return (loglik = loglik, a_filt = a_filt, α = α, V = V)
end

# A model with m > p, the shape in which a p-sized scratch buffer used for an
# m-sized quantity would overflow.
function _partial_missing_setup(; p = 3, m = 4, n = 25, seed = 20260917)
    rng = Xoshiro(seed)
    Z = randn(rng, p, m)
    H = let A = randn(rng, p, p)
        A * A' + p * I
    end
    T = randn(rng, m, m) ./ 4
    R = Matrix{Float64}(I, m, m)
    Q = let A = randn(rng, m, m)
        A * A' + m * I
    end
    a1 = zeros(m)
    P1 = Matrix{Float64}(I, m, m) * 3
    y = randn(rng, p, n)
    return (; Z, H, T, R, Q, a1, P1, y, p, m, n)
end

function _workspace(s)
    ws = KalmanWorkspace{Float64}(s.p, s.m, size(s.Q, 1), s.n)
    set_params!(ws, s.Z, s.H, s.T, s.R, s.Q)
    set_initial!(ws, s.a1, s.P1)
    return ws
end

@testset "partial missing — an observed row survives a missing sibling" begin
    # Two directly observed states with tiny noise. Series 1 carries a sharp
    # reading at t=3; blanking series 2 in the same period must not discard it.
    Z = Matrix{Float64}(I, 2, 2)
    H = Matrix{Float64}(I, 2, 2) * 1e-6
    T = Matrix{Float64}(I, 2, 2)
    R = Matrix{Float64}(I, 2, 2)
    Q = Matrix{Float64}(I, 2, 2) * 1e-6
    a1 = zeros(2)
    P1 = Matrix{Float64}(I, 2, 2)

    y = ones(2, 5)
    y[1, 3] = 3.0

    ws_both = KalmanWorkspace{Float64}(2, 2, 2, 5)
    set_params!(ws_both, Z, H, T, R, Q)
    set_initial!(ws_both, a1, P1)
    kalman_filter!(ws_both, y)
    state_both = filtered_states(ws_both)[1, 3]

    y_partial = copy(y)
    y_partial[2, 3] = NaN
    ws_partial = KalmanWorkspace{Float64}(2, 2, 2, 5)
    set_params!(ws_partial, Z, H, T, R, Q)
    set_initial!(ws_partial, a1, P1)
    kalman_filter!(ws_partial, y_partial)
    state_partial = filtered_states(ws_partial)[1, 3]

    # Series 1's own update is unaffected by series 2 going missing.
    @test state_partial ≈ state_both atol = 1e-9

    # The value is genuinely used: it pulls the state well away from the prior.
    @test state_partial > 2.0

    # The period is observed, not missing.
    @test !Siphon.missing_mask(ws_partial)[3]
    @test Siphon.n_observed(ws_partial, 3) == 1
    @test collect(Siphon.observed_rows(ws_partial, 3)) == [1]
end

@testset "partial missing — deleting a row matches the reduced model" begin
    s = _partial_missing_setup()

    # Row 2 absent in every period must equal a genuine 2-row model on rows 1,3.
    y_gone = copy(s.y)
    y_gone[2, :] .= NaN

    ws_full = _workspace(s)
    ll_full = kalman_filter!(ws_full, y_gone)
    kalman_smoother!(ws_full)

    keep = [1, 3]
    ws_reduced = KalmanWorkspace{Float64}(length(keep), s.m, size(s.Q, 1), s.n)
    set_params!(ws_reduced, s.Z[keep, :], s.H[keep, keep], s.T, s.R, s.Q)
    set_initial!(ws_reduced, s.a1, s.P1)
    ll_reduced = kalman_filter!(ws_reduced, s.y[keep, :])
    kalman_smoother!(ws_reduced)

    @test ll_full ≈ ll_reduced
    @test filtered_states(ws_full) ≈ filtered_states(ws_reduced)
    @test smoothed_states(ws_full) ≈ smoothed_states(ws_reduced)
    @test variances_smoothed_states(ws_full) ≈ variances_smoothed_states(ws_reduced)
end

@testset "partial missing — in-place filter matches a reduced-system reference" begin
    s = _partial_missing_setup()
    y = copy(s.y)
    y[1, 4] = NaN                      # one row gone
    y[2, 9] = NaN
    y[[1, 3], 12] .= NaN               # two rows gone
    y[:, 17] .= NaN                    # whole period gone

    ws = _workspace(s)
    loglik = kalman_filter!(ws, y)
    kalman_smoother!(ws)
    ref = _reference_filter_smoother(s.Z, s.H, s.T, s.R, s.Q, s.a1, s.P1, y)

    @test loglik ≈ ref.loglik
    @test filtered_states(ws) ≈ ref.a_filt
    @test smoothed_states(ws) ≈ ref.α
    @test variances_smoothed_states(ws) ≈ ref.V

    @test Siphon.n_observed(ws, 4) == 2
    @test Siphon.n_observed(ws, 12) == 1
    @test Siphon.n_observed(ws, 17) == 0
    @test Siphon.n_observed(ws, 1) == s.p
    @test collect(Siphon.observed_rows(ws, 12)) == [2]
    @test Siphon.missing_mask(ws)[17]
    @test !Siphon.missing_mask(ws)[12]
end

@testset "partial missing — functional filter matches the in-place filter" begin
    s = _partial_missing_setup()
    y = copy(s.y)
    y[1, 4] = NaN
    y[[2, 3], 8] .= NaN
    y[:, 20] .= NaN

    parms = KFParms(s.Z, s.H, s.T, s.R, s.Q)
    ref = _reference_filter_smoother(s.Z, s.H, s.T, s.R, s.Q, s.a1, s.P1, y)

    @test kalman_loglik(parms, y, s.a1, s.P1) ≈ ref.loglik

    result = kalman_filter(parms, y, s.a1, s.P1)
    @test loglikelihood(result) ≈ ref.loglik
    @test filtered_states(result) ≈ ref.a_filt

    smoothed = kalman_smoother(result, s.Z, s.T)
    @test smoothed.alpha ≈ ref.α
    @test smoothed.V ≈ ref.V

    ws = _workspace(s)
    @test kalman_filter!(ws, y) ≈ loglikelihood(result)
    kalman_smoother!(ws)
    @test smoothed_states(ws) ≈ smoothed.alpha
    @test variances_smoothed_states(ws) ≈ smoothed.V
end

@testset "partial missing — mixed-frequency censoring" begin
    # Rows 1 and 2 observed only every third period, as a quarterly series
    # embedded in a monthly panel.
    s = _partial_missing_setup(; p = 4, m = 4, n = 36)
    y = copy(s.y)
    for t in 1:s.n, i in 1:2

        t % 3 == 0 || (y[i, t] = NaN)
    end

    ws = _workspace(s)
    loglik = kalman_filter!(ws, y)
    kalman_smoother!(ws)
    ref = _reference_filter_smoother(s.Z, s.H, s.T, s.R, s.Q, s.a1, s.P1, y)

    @test loglik ≈ ref.loglik
    @test smoothed_states(ws) ≈ ref.α
    @test variances_smoothed_states(ws) ≈ ref.V

    @test [Siphon.n_observed(ws, t) for t in 1:6] == [2, 2, 4, 2, 2, 4]
    @test count(Siphon.observed_mask(ws)) == 2 * s.n + 2 * div(s.n, 3)
end

@testset "partial missing — the likelihood constant counts observed scalars" begin
    s = _partial_missing_setup(; p = 3, m = 2, n = 10)
    parms = KFParms(s.Z, s.H, s.T, s.R, s.Q)

    # Blanking one scalar changes the constant term by exactly log(2π)/2 relative
    # to the same model refitted with that row genuinely absent, which is what
    # distinguishes a per-element count from a per-period one.
    y = copy(s.y)
    y[3, 5] = NaN

    ll = kalman_loglik(parms, y, s.a1, s.P1)
    ref = _reference_filter_smoother(s.Z, s.H, s.T, s.R, s.Q, s.a1, s.P1, y)
    @test ll ≈ ref.loglik

    # Every row present in every period: the count is p*n.
    ll_full = kalman_loglik(parms, s.y, s.a1, s.P1)
    ref_full = _reference_filter_smoother(s.Z, s.H, s.T, s.R, s.Q, s.a1, s.P1, s.y)
    @test ll_full ≈ ref_full.loglik
end

@testset "partial missing — fully observed results are unchanged" begin
    s = _partial_missing_setup()
    ws = _workspace(s)
    loglik = kalman_filter!(ws, s.y)
    kalman_smoother!(ws)
    ref = _reference_filter_smoother(s.Z, s.H, s.T, s.R, s.Q, s.a1, s.P1, s.y)

    @test loglik ≈ ref.loglik
    @test filtered_states(ws) ≈ ref.a_filt
    @test smoothed_states(ws) ≈ ref.α

    @test all(Siphon.observed_mask(ws))
    @test !any(Siphon.missing_mask(ws))
    @test all(t -> Siphon.n_observed(ws, t) == s.p, 1:s.n)
    @test all(t -> collect(Siphon.observed_rows(ws, t)) == 1:s.p, 1:s.n)
end

@testset "partial missing — per-period paths reject a partial period" begin
    s = _partial_missing_setup(; p = 2, m = 2, n = 12)
    y = copy(s.y)
    y[1, 5] = NaN

    ws = DiffuseKalmanWorkspace{Float64}(s.p, s.m, size(s.Q, 1), s.n)
    set_params!(ws.base, s.Z, s.H, s.T, s.R, s.Q)
    set_initial_diffuse!(ws, s.a1, zeros(s.m, s.m), Matrix{Float64}(I, s.m, s.m))

    @test_throws "in period 5 only rows [2] of 2 are observed" Siphon.kalman_filter_diffuse!(ws, y)
    @test_throws "approximate diffuse initialisation" Siphon.kalman_filter_diffuse!(ws, y)
    P1_inf = Matrix{Float64}(I, s.m, s.m)
    parms = KFParms(s.Z, s.H, s.T, s.R, s.Q)
    @test_throws "kalman_loglik_diffuse does not support partially observed periods" kalman_loglik(
        parms, y, s.a1, zeros(s.m, s.m), P1_inf)
    @test_throws "kalman_filter_diffuse does not support partially observed periods" kalman_filter(
        parms, y, s.a1, zeros(s.m, s.m), P1_inf)

    # A period that is entirely missing is still accepted.
    y_whole = copy(s.y)
    y_whole[:, 5] .= NaN
    @test Siphon.kalman_filter_diffuse!(ws, y_whole) isa Real
end

@testset "partial missing — the static path falls back to the per-element filter" begin
    using StaticArrays

    s = _partial_missing_setup(; p = 2, m = 2, n = 10)
    parms_static = KFParms(
        SMatrix{2, 2}(s.Z), SMatrix{2, 2}(s.H), SMatrix{2, 2}(s.T),
        SMatrix{2, 2}(s.R), SMatrix{2, 2}(s.Q))
    a1s = SVector{2}(s.a1)
    P1s = SMatrix{2, 2}(s.P1)

    y = copy(s.y)
    y[1, 4] = NaN

    # The StaticArrays specialization fixes the observation dimension in its type
    # parameters, so a partially observed panel goes to the dense per-element filter.
    parms = KFParms(s.Z, s.H, s.T, s.R, s.Q)
    @test kalman_loglik(parms_static, y, a1s, P1s) ≈ kalman_loglik(parms, y, s.a1, s.P1)
    filt_static = kalman_filter(parms_static, y, a1s, P1s)
    filt = kalman_filter(parms, y, s.a1, s.P1)
    @test filt_static.att ≈ filt.att
    @test filt_static.observed_mask == filt.observed_mask

    # Fully observed and fully missing periods use the static kernel and agree
    # with the dense path.
    y_whole = copy(s.y)
    y_whole[:, 4] .= NaN
    @test kalman_loglik(parms_static, y_whole, a1s, P1s) ≈
          kalman_loglik(parms, y_whole, s.a1, s.P1)
    @test kalman_loglik(parms_static, s.y, a1s, P1s) ≈
          kalman_loglik(parms, s.y, s.a1, s.P1)
end

@testset "partial missing — filter result carries the per-element mask" begin
    s = _partial_missing_setup(; p = 3, m = 2, n = 8)
    y = copy(s.y)
    y[2, 3] = NaN
    y[:, 6] .= NaN

    result = kalman_filter(KFParms(s.Z, s.H, s.T, s.R, s.Q), y, s.a1, s.P1)

    @test size(Siphon.observed_mask(result)) == (s.p, s.n)
    @test Siphon.observed_rows(result, 3) == [1, 3]
    @test Siphon.observed_rows(result, 1) == [1, 2, 3]
    @test isempty(Siphon.observed_rows(result, 6))
    @test Siphon.missing_mask(result)[6]
    @test !Siphon.missing_mask(result)[3]

    # The period mask is exactly the all-false columns of the element mask.
    for t in 1:s.n
        @test Siphon.missing_mask(result)[t] ==
              !any(view(Siphon.observed_mask(result), :, t))
    end
end

# A linear measurement makes the EKF reduce to the Kalman filter, so these tests
# exercise the guard rather than any nonlinear behavior.
struct PartialMissingEKFMeasurement{ZT <: AbstractMatrix} <: Siphon.AbstractEKFMeasurement
    Z::ZT
end

Siphon.measurement(m::PartialMissingEKFMeasurement, a, t) = m.Z * a
Siphon.measurement!(out, m::PartialMissingEKFMeasurement, a, t) = (mul!(out, m.Z, a); out)
Siphon.measurement_jacobian(m::PartialMissingEKFMeasurement, a, t) = m.Z
function Siphon.measurement_jacobian!(Z, m::PartialMissingEKFMeasurement, a, t)
    copyto!(Z, m.Z)
    return Z
end

@testset "partial missing — the EKF paths reject a partial period" begin
    s = _partial_missing_setup(; p = 3, m = 2, n = 10)
    meas = PartialMissingEKFMeasurement(s.Z)
    parms = EKFParms(meas, AnalyticJacobian(), s.H, s.T, s.R, s.Q)

    y = copy(s.y)
    y[2, 4] = NaN

    # The EKF updates one period at a time, so a partially observed period would
    # otherwise discard the rows that are present.
    @test_throws "in period 4 only rows [1, 3] of 3 are observed" ekf_loglik(parms, y, s.a1, s.P1)
    @test_throws "ekf_filter does not support partially observed periods" ekf_filter(
        parms, y, s.a1, s.P1)

    ws = EKFWorkspace(meas, AnalyticJacobian(), s.H, s.T, s.R, s.Q, s.a1, s.P1, s.n)
    @test_throws "in period 4 only rows [1, 3] of 3 are observed" ekf_filter!(ws, y)
    @test_throws "ekf_filter!" ekf_filter_and_smooth!(ws, y)

    # Fully observed and entirely missing periods remain valid on these paths.
    y_whole = copy(s.y)
    y_whole[:, 4] .= NaN
    @test ekf_loglik(parms, y_whole, s.a1, s.P1) isa Real
    @test ekf_filter!(ws, y_whole) isa Real
    @test ekf_loglik(parms, s.y, s.a1, s.P1) isa Real
end

@testset "partial missing — the static path returns on a singular innovation covariance" begin
    using StaticArrays

    # A negative eigenvalue in H makes F = Z P Z' + H indefinite, so its Cholesky
    # reports failure and the filter returns -Inf instead of factoring.
    Z = SMatrix{2, 2}(Matrix{Float64}(I, 2, 2))
    H = SMatrix{2, 2}([1.0 0.0; 0.0 -5.0])
    T = SMatrix{2, 2}([0.5 0.0; 0.0 0.5])
    R = SMatrix{2, 2}(Matrix{Float64}(I, 2, 2))
    Q = SMatrix{2, 2}(Matrix{Float64}(I, 2, 2))
    a1 = SVector{2}(zeros(2))
    P1 = SMatrix{2, 2}(Matrix{Float64}(I, 2, 2))
    parms = KFParms(Z, H, T, R, Q)
    y = [1.0 1.0; 1.0 1.0]

    result = kalman_filter(parms, y, a1, P1)
    @test loglikelihood(result) == -Inf
    @test Siphon.missing_mask(result) isa BitVector
    @test size(Siphon.observed_mask(result)) == (2, 2)
end

@testset "partial missing — StateSpaceModel and spec paths use the observed rows" begin
    spec = custom_ssm(
        Z = reshape([1.0, 1.0], 2, 1),
        H = diag_free([:h1, :h2]; init = 1.0),
        T = [1.0],
        R = [1.0],
        Q = [FreeParam(:q, init = 1.0, lower = 0.0)],
        a1 = [0.0],
        P1 = [10.0]
    )
    rng = Random.Xoshiro(2)
    n = 60
    level = cumsum(randn(rng, n))
    y = vcat(level', level') .+ randn(rng, 2, n)
    y[1, 10:15] .= NaN   # row 1 missing, row 2 observed
    y[:, 30] .= NaN      # fully missing period

    θ = (h1 = 1.0, h2 = 1.0, q = 1.0)
    ss = build_linear_state_space(spec, θ, y; use_static = false)
    filt = kalman_filter(ss.p, y, ss.a1, ss.P1)
    ref = kalman_smoother(ss.p, y, ss.a1, ss.P1)

    # The default use_static = true no longer rejects the partial periods
    @test predict(spec, θ, y).loglik ≈ filt.loglik
    @test forecast(spec, θ, y, 2).yhat ≈ forecast(ss.p, y, ss.a1, ss.P1, 2).yhat
    @test Siphon.DSL._spec_loglik(spec, θ, y, true) ≈ filt.loglik

    # The model smoother uses the observed-row mask
    model = StateSpaceModel(spec, θ, n)
    @test kalman_filter!(model, y) ≈ filt.loglik
    @test smoothed_states(model) ≈ ref.α
    @test smoothed_states_cov(model) ≈ ref.V

    model = StateSpaceModel(spec, n)
    fit!(MLE(), model, y)
    ss_fit = build_linear_state_space(spec, parameters(model), y; use_static = false)
    @test smoothed_states(model) ≈ kalman_smoother(ss_fit.p, y, ss_fit.a1, ss_fit.P1).α

    # SmootherResult's convenience constructor uses the mask too
    @test SmootherResult(ss.p, y, ss.a1, ss.P1).alpha ≈ ref.α

    @test_throws "Use fit!(MLE(), ...)" fit!(EM(), StateSpaceModel(spec, n), y)
end

@testset "kalman_loglik and kalman_filter agree for a nearly asymmetric F" begin
    # Z P Z' + H can be slightly asymmetric in floating point; both paths
    # symmetrise it the same way.
    rng = Random.Xoshiro(7)
    p, m, n = 3, 3, 40
    A = randn(rng, m, m)
    P1 = A * A' + I
    Z = randn(rng, p, m)
    H = [1.0 0.3 0.0; 0.3000001 1.0 0.2; 0.0 0.2 1.0]   # asymmetric by 1e-7
    parms = KFParms(Z, H, 0.9 * Matrix(1.0I, m, m), Matrix(1.0I, m, m), Matrix(0.5I, m, m))
    y = randn(rng, p, n)
    @test kalman_loglik(parms, y, zeros(m), P1) ≈
          kalman_filter(parms, y, zeros(m), P1).loglik rtol = 1e-12
end
