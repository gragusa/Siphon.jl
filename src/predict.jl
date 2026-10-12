"""
    predict.jl

Prediction and forecasting functions for state-space models.

Provides:
- `predict`: In-sample fitted values (yₜ|ₜ₋₁ = Z * aₜ)
- `forecast`: Out-of-sample h-step ahead forecasts, as a `ForecastResult`
- `forecast_paths`: Simulated forecast trajectories
"""

using LinearAlgebra
using .DSL: SSMSpec, build_linear_state_space, _ss_filter

# ============================================
# Forecast result
# ============================================

"""
    ForecastResult{T<:Real}

Forecasts of a state-space model for the `h` periods after the end of a sample of
`origin` periods. Every `forecast` method returns one.

Column `j` holds the forecast of period `origin + j` given the observations up to
`origin`:
- `yhat::Matrix{T}`: observation forecasts E[yₙ₊ⱼ | y₁:ₙ] (p × h)
- `F::Array{T,3}`: their covariances (p × p × h)
- `a::Matrix{T}`: state forecasts E[αₙ₊ⱼ | y₁:ₙ] (m × h)
- `P::Array{T,3}`: their covariances (m × m × h)
- `origin::Int`: the last sample period `n`

# Accessors
- `forecast_observations(fc)`, `forecast_observations_cov(fc)` → `yhat`, `F`
- `forecast_states(fc)`, `forecast_states_cov(fc)` → `a`, `P`
- `forecast_periods(fc)` → `origin+1:origin+h`
- `forecast_interval(fc, level)` → pointwise normal intervals

`plot(fc)` draws the forecasts against `forecast_periods(fc)`; pass `time` to use another
axis.

# Example
```julia
result = kalman_filter(p, y, a1, P1)
fc = forecast(result, 12)
forecast_observations(fc)            # p × 12
lo, hi = forecast_interval(fc, 0.9)
```
"""
struct ForecastResult{T <: Real}
    yhat::Matrix{T}
    F::Array{T, 3}
    a::Matrix{T}
    P::Array{T, 3}
    origin::Int
end

"""
    forecast_observations(fc::ForecastResult) -> Matrix

Return the observation forecasts (p × h); column `j` is period `origin + j`.
"""
forecast_observations(fc::ForecastResult) = fc.yhat

"""
    forecast_observations_cov(fc::ForecastResult) -> Array{T,3}

Return the covariances of the observation forecasts (p × p × h).
"""
forecast_observations_cov(fc::ForecastResult) = fc.F

"""
    forecast_states(fc::ForecastResult) -> Matrix

Return the state forecasts (m × h); column `j` is period `origin + j`.
"""
forecast_states(fc::ForecastResult) = fc.a

"""
    forecast_states_cov(fc::ForecastResult) -> Array{T,3}

Return the covariances of the state forecasts (m × m × h).
"""
forecast_states_cov(fc::ForecastResult) = fc.P

"""
    forecast_periods(fc::ForecastResult) -> UnitRange{Int}

Return the periods the forecasts are for, `origin+1:origin+h`.
"""
forecast_periods(fc::ForecastResult) = fc.origin .+ (1:size(fc.yhat, 2))

"""
    forecast_interval(fc::ForecastResult, level=0.95; what=:observations) -> (lower, upper)

Return pointwise normal intervals with coverage `level` for the observation forecasts,
or for the state forecasts when `what = :states`. `lower` and `upper` have the shape of
the forecasts.
"""
function forecast_interval(fc::ForecastResult, level::Real = 0.95;
        what::Symbol = :observations)
    if what === :observations
        lower, upper = confidence_bands(fc.yhat, fc.F, level)
    elseif what === :states
        lower, upper = confidence_bands(fc.a, fc.P, level)
    else
        throw(ArgumentError("what must be :observations or :states, got :$what"))
    end
    return (lower = lower, upper = upper)
end

# ============================================
# Prediction (In-Sample Fitted Values)
# ============================================

"""
    predicted_observations(r::KalmanFilterResult) -> Matrix
    predicted_observations(r::DiffuseFilterResult) -> Matrix

Return the one-step-ahead predictions ŷₜ|ₜ₋₁ = Z aₜ for t = 1:n (p × n), including
the periods where `y` was missing.
"""
function predicted_observations(r::Union{KalmanFilterResult, DiffuseFilterResult})
    _require_output(r, :predicted_observations)
    return _as_matrix(r.p.Z) * r.at
end

"""
    predict(spec::SSMSpec, θ::NamedTuple, y::AbstractMatrix; use_static=true) -> Matrix
    predict(p::KFParms, y, a1, P1) -> Matrix

Return the in-sample one-step-ahead predictions ŷₜ|ₜ₋₁ = Z aₜ (p × n) of a model at
parameters `θ`, or of the system `p` with initial state `a1`, `P1`.

This is `predicted_observations(kalman_filter(…))`. Call the filter instead when you
need the innovations, the states or the log-likelihood as well.

# Arguments
- `use_static::Bool=true`: Use StaticArrays for small matrices (dimensions ≤ 13)

# Example
```julia
spec = local_level()
θ = (var_obs = 1.0, var_level = 0.5)
y = randn(1, 100)

yhat = predict(spec, θ, y)
plot(vec(y), label="observed")
plot!(vec(yhat), label="predicted")
```
"""
function predict(spec::SSMSpec, θ::NamedTuple, y::AbstractMatrix; use_static::Bool = true)
    y = _nan_obs(y)
    ss = build_linear_state_space(spec, θ, y; use_static = use_static)
    return predicted_observations(_ss_filter(ss, y))
end

function predict(p::KFParms, y::AbstractVecOrMat, a1, P1)
    return predicted_observations(kalman_filter(p, _nan_obs(y), a1, P1))
end

# ============================================
# Forecasting (Out-of-Sample)
# ============================================

"""
    forecast(r::AbstractFilterResult, h::Integer) -> ForecastResult

Forecast the `h` periods after the sample of the filter result `r`, starting from the
filtered state of the last period. The filter is not run again.

For an exact diffuse filter the diffuse period must end within the sample. For an
extended Kalman filter the observation forecast is `h(aₙ₊ⱼ, n + j)` and its covariance
uses the measurement Jacobian at that state.
"""
function forecast(r::AbstractFilterResult, h::Integer)
    _require_output(r, :forecast)
    n = size(r.att, 2)
    _check_diffuse_ended(r, n)
    return _forecast_from_filtered(r.p, r.att[:, n], r.Ptt[:, :, n], n, h)
end

"""
    forecast(spec::SSMSpec, θ::NamedTuple, y::AbstractMatrix, h::Integer; use_static=true) -> ForecastResult
    forecast(p::KFParms, y, a1, P1, h::Integer) -> ForecastResult

Filter `y` and forecast the `h` periods after it. See
[`forecast(::AbstractFilterResult, ::Integer)`](@ref).

# Arguments
- `use_static::Bool=true`: Use StaticArrays for small matrices (dimensions ≤ 13)

# Example
```julia
spec = local_level()
θ = (var_obs = 1.0, var_level = 0.5)
y = randn(1, 100)

fc = forecast(spec, θ, y, 12)        # periods 101:112
forecast_observations(fc)
sqrt.(forecast_observations_cov(fc)[1, 1, :])
```
"""
function forecast(
        spec::SSMSpec,
        θ::NamedTuple,
        y::AbstractMatrix,
        h::Integer;
        use_static::Bool = true
)
    y = _nan_obs(y)
    ss = build_linear_state_space(spec, θ, y; use_static = use_static)
    return forecast(_ss_filter(ss, y), h)
end

function forecast(p::KFParms, y::AbstractVecOrMat, a1, P1, h::Integer)
    return forecast(kalman_filter(p, _nan_obs(y), a1, P1), h)
end

_check_diffuse_ended(::AbstractFilterResult, n) = nothing
function _check_diffuse_ended(r::DiffuseFilterResult, n)
    r.d < n && return nothing
    throw(ArgumentError(
        "forecast: the diffuse period lasts the whole sample, so the forecast " *
        "variances are infinite"))
end

# The measurement prediction at state `a` in period `t`, and its Jacobian.
_measurement_mean(p::KFParms, a, t) = _as_matrix(p.Z) * a
_measurement_mean(p::EKFParms, a, t) = measurement(p.measurement, a, t)

_measurement_jacobian(p::KFParms, a, t) = _as_matrix(p.Z)
function _measurement_jacobian(p::EKFParms, a, t)
    return _ekf_jacobian(p.jacobian_mode, p.measurement, a, t)
end

# Forecasts for periods n+1, …, n+h from the filtered state of period n.
function _forecast_from_filtered(p, att::AbstractVector, Ptt::AbstractMatrix,
        n::Integer, h::Integer)
    h >= 1 || throw(ArgumentError("forecast horizon must be at least 1, got $h"))
    Tm, R, Q, H = _as_matrix(p.T), _as_matrix(p.R), _as_matrix(p.Q), _as_matrix(p.H)
    ET = float(promote_type(eltype(att), eltype(Ptt), eltype(Tm), eltype(R),
        eltype(Q), eltype(H)))
    m = length(att)
    obs_dim = size(H, 1)
    RQR = R * Q * R'

    yhat = Matrix{ET}(undef, obs_dim, h)
    F_fc = Array{ET}(undef, obs_dim, obs_dim, h)
    a_fc = Matrix{ET}(undef, m, h)
    P_fc = Array{ET}(undef, m, m, h)

    a = Tm * att
    P = Tm * Ptt * Tm' + RQR
    for j in 1:h
        a_fc[:, j] = a
        P_fc[:, :, j] = P
        Z = _measurement_jacobian(p, a, n + j)
        yhat[:, j] = _measurement_mean(p, a, n + j)
        F_fc[:, :, j] = Z * P * Z' + H
        if j < h
            a = Tm * a
            P = Tm * P * Tm' + RQR
        end
    end

    return ForecastResult(yhat, F_fc, a_fc, P_fc, Int(n))
end

# ============================================
# Simulated forecast paths
# ============================================

"""
    forecast_paths(r::AbstractFilterResult, h::Integer, n_paths::Integer; rng=Random.default_rng()) -> Array
    forecast_paths(spec::SSMSpec, θ, y, h, n_paths; use_static=true, rng=Random.default_rng()) -> Array
    forecast_paths(p::KFParms, y, a1, P1, h, n_paths; rng=Random.default_rng()) -> Array

Simulate `n_paths` trajectories of the observations over the `h` periods after the
sample, drawn from their joint predictive distribution given the data.

Returns an array of size `p × h × n_paths`. Singular covariances (a state shock of
reduced rank, or `H = 0`) are allowed.

# Example
```julia
using Random, Statistics
paths = forecast_paths(spec, θ, y, 12, 1000; rng = Xoshiro(1))
quantiles = [quantile(paths[1, j, :], [0.1, 0.5, 0.9]) for j in 1:12]
```
"""
function forecast_paths(r::AbstractFilterResult, h::Integer, n_paths::Integer;
        rng::AbstractRNG = Random.default_rng())
    _require_output(r, :forecast_paths)
    n = size(r.att, 2)
    _check_diffuse_ended(r, n)
    return _forecast_paths(r.p, r.att[:, n], r.Ptt[:, :, n], n, h, n_paths, rng)
end

function forecast_paths(
        spec::SSMSpec,
        θ::NamedTuple,
        y::AbstractMatrix,
        h::Integer,
        n_paths::Integer;
        use_static::Bool = true,
        rng::AbstractRNG = Random.default_rng()
)
    y = _nan_obs(y)
    ss = build_linear_state_space(spec, θ, y; use_static = use_static)
    return forecast_paths(_ss_filter(ss, y), h, n_paths; rng = rng)
end

function forecast_paths(p::KFParms, y::AbstractVecOrMat, a1, P1, h::Integer,
        n_paths::Integer; rng::AbstractRNG = Random.default_rng())
    return forecast_paths(kalman_filter(p, _nan_obs(y), a1, P1), h, n_paths; rng = rng)
end

function _forecast_paths(p, att::AbstractVector, Ptt::AbstractMatrix, n::Integer,
        h::Integer, n_paths::Integer, rng::AbstractRNG)
    h >= 1 || throw(ArgumentError("forecast horizon must be at least 1, got $h"))
    n_paths >= 0 || throw(ArgumentError("n_paths must be nonnegative, got $n_paths"))
    Tm, R, Q, H = _as_matrix(p.T), _as_matrix(p.R), _as_matrix(p.Q), _as_matrix(p.H)
    ET = float(promote_type(eltype(att), eltype(Ptt), eltype(Tm), eltype(R),
        eltype(Q), eltype(H)))
    m = length(att)
    obs_dim = size(H, 1)

    # The state of the first forecast period, a_{n+1} ~ N(T a_{n|n}, T P_{n|n} T' + RQR').
    a_start = Tm * att
    P_start = Tm * Ptt * Tm' + R * Q * R'
    L_P = psd_factor(P_start).L
    L_RQ = R * psd_factor(Q).L
    L_H = psd_factor(H).L

    paths = Array{ET}(undef, obs_dim, h, n_paths)
    for s in 1:n_paths
        a = a_start + L_P * randn(rng, ET, m)
        for j in 1:h
            paths[:, j, s] = _measurement_mean(p, a, n + j) + L_H * randn(rng, ET, obs_dim)
            a = Tm * a + L_RQ * randn(rng, ET, size(L_RQ, 2))
        end
    end

    return paths
end
