# The accessor vocabulary shared by filter results, workspaces and models.

using Siphon
using Test
using LinearAlgebra
using Random

@testset "Accessors: results and workspaces agree" begin
    rng = MersenneTwister(11)
    p_obs, m, n = 2, 2, 40
    Z = [1.0 0.0; 0.5 1.0]
    H = Diagonal([0.4, 0.7]) |> Matrix
    Tm = [0.8 0.1; 0.0 0.6]
    R = Matrix(1.0I, m, m)
    Q = [0.3 0.05; 0.05 0.2]
    a1 = [0.1, -0.2]
    P1 = [2.0 0.3; 0.3 1.5]
    y = randn(rng, p_obs, n)
    y[1, 7] = NaN
    y[:, 12] .= NaN

    kf = KFParms(Z, H, Tm, R, Q)
    res = kalman_filter(kf, y, a1, P1)
    ws = KalmanWorkspace(Z, H, Tm, R, Q, a1, P1, n)
    kalman_filter!(ws, y)
    kalman_smoother!(ws; crosscov = true)

    @test res isa AbstractFilterResult
    @test ws isa AbstractFilterWorkspace

    for acc in (predicted_states, predicted_states_cov, filtered_states,
        filtered_states_cov, kalman_gains)
        @test acc(res) ≈ acc(ws)
    end
    @test prediction_errors(res)[:, [1:6; 13:n]] ≈ prediction_errors(ws)[:, [1:6; 13:n]]
    @test prediction_errors_cov(res)[:, :, 20] ≈ prediction_errors_cov(ws)[:, :, 20]
    @test loglikelihood(res) ≈ loglikelihood(ws)
    @test Siphon.missing_mask(res) == Siphon.missing_mask(ws)

    @test system_matrices(res) === kf
    sm = system_matrices(ws)
    @test sm isa KFParms
    for (acc, M) in ((obs_matrix, Z), (obs_cov, H), (transition_matrix, Tm),
        (selection_matrix, R), (state_cov, Q))
        @test acc(res) == M
        @test acc(ws) == M
    end

    @test initial_state(res) == a1
    @test initial_state_cov(res) == P1
    @test initial_state(ws) == a1
    @test initial_state_cov(ws) == P1

    # The system matrices of a result feed back into the functional filter
    @test kalman_loglik(system_matrices(res), y, initial_state(res),
        initial_state_cov(res)) ≈ loglikelihood(res)

    @test size(smoothed_states(ws)) == (m, n)
    @test size(smoothed_states_cov(ws)) == (m, m, n)
    @test size(crosslag_covs(ws)) == (m, m, n - 1)

    # Single-matrix setters
    Siphon.set_Z!(ws, 2 .* Z)
    @test obs_matrix(ws) == 2 .* Z
    Siphon.set_Q!(ws, 2 .* Q)
    @test state_cov(ws) == 2 .* Q
    @test ws.RQR ≈ R * (2 .* Q) * R'
end

@testset "Accessors: diffuse result and workspace" begin
    y = reshape(collect(range(1.0, 5.0; length = 30)) .+ sin.(1:30), 1, :)
    Z, H, Tm, R, Q = [1.0;;], [1.0;;], [1.0;;], [1.0;;], [0.5;;]
    a1, P1_star, P1_inf = [0.0], [0.0;;], [1.0;;]
    res = kalman_filter(KFParms(Z, H, Tm, R, Q), y, a1, P1_star, P1_inf)
    ws = DiffuseKalmanWorkspace(Z, H, Tm, R, Q, a1, P1_star, P1_inf, size(y, 2))
    kalman_filter!(ws, y)

    @test res isa AbstractFilterResult
    @test ws isa AbstractFilterWorkspace
    @test loglikelihood(res) ≈ loglikelihood(ws)
    @test filtered_states(res) ≈ filtered_states(ws)
    @test predicted_states_cov(res) ≈ predicted_states_cov(ws)
    @test system_matrices(ws) isa KFParms
    @test obs_matrix(ws) == Z
    @test initial_state_cov(res) == P1_star
    @test initial_state_cov(ws) == P1_star
end

@testset "Accessors: extended Kalman filter" begin
    rng = MersenneTwister(5)
    Z = [1.0 0.0; 0.3 1.0]
    H, Tm, R, Q = Matrix(0.5I, 2, 2), [0.9 0.0; 0.1 0.7], Matrix(1.0I, 2, 2),
    Matrix(0.2I, 2, 2)
    a1, P1 = zeros(2), Matrix(1.0I, 2, 2)
    y = randn(rng, 2, 25)
    meas = EKFTestLinearMeasurement(Z)

    res = kalman_filter(EKFParms(meas, H, Tm, R, Q), y, a1, P1)
    ws = EKFWorkspace(meas, ADJacobian(), H, Tm, R, Q, a1, P1, size(y, 2))
    kalman_filter!(ws, y)
    lin = kalman_filter(KFParms(Z, H, Tm, R, Q), y, a1, P1)

    @test res isa AbstractFilterResult
    @test ws isa AbstractFilterWorkspace
    @test filtered_states(res) ≈ filtered_states(lin)
    @test filtered_states_cov(ws) ≈ filtered_states_cov(lin)
    @test loglikelihood(ws) ≈ loglikelihood(lin)
    @test system_matrices(res) isa EKFParms
    @test system_matrices(ws) isa EKFParms
    @test transition_matrix(ws) == Tm
    @test state_cov(res) == Q
    @test initial_state(ws) == a1
    @test_throws ArgumentError obs_matrix(res)
    @test_throws ArgumentError obs_matrix(ws)
end

@testset "Accessors: StateSpaceModel" begin
    nile_y = y  # Nile data from runtests.jl
    spec = local_level()
    model = StateSpaceModel(spec, size(nile_y, 2))

    @test model isa AbstractStateSpaceModel
    @test !isfitted(model)
    @test_throws "fit!(MLE(), model, y)" parameters(model)
    @test_throws "fit!(MLE(), model, y)" system_matrices(model)
    @test_throws "fit!(MLE(), model, y)" loglikelihood(model)
    @test_throws ArgumentError isconverged(model)
    @test_throws ArgumentError niterations(model)
    @test_throws "kalman_filter!(model, y)" filtered_states(model)
    @test_throws ArgumentError kalman_gains(model)

    fit!(MLE(), model, nile_y)
    p = system_matrices(model)
    @test p isa KFParms
    @test obs_cov(model) == p.H
    @test state_cov(model) == p.Q
    @test initial_state(model) == [0.0]
    @test size(initial_state_cov(model)) == (1, 1)

    # A fitted model's matrices and initial state reproduce its log-likelihood
    filt = kalman_filter(p, nile_y, initial_state(model), initial_state_cov(model))
    @test loglikelihood(filt) ≈ loglikelihood(model)
    @test filtered_states(filt) ≈ filtered_states(model)
    @test kalman_gains(filt) ≈ kalman_gains(model)
    @test Siphon.missing_mask(model) == Siphon.missing_mask(filt)
    @test size(smoothed_states_cov(model)) == (1, 1, size(nile_y, 2))
end

@testset "Accessors: DynamicFactorModel status" begin
    model = DynamicFactorModel(4, 1, 30)
    @test model isa AbstractStateSpaceModel
    @test !isfitted(model)
    @test_throws "fit!(EM(), model, y)" isconverged(model)
    @test_throws "fit!(EM(), model, y)" loglikelihood(model)
end
