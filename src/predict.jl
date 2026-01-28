"""
    predict.jl

Prediction and forecasting functions for state-space models.

Provides:
- `predict`: In-sample fitted values (yₜ|ₜ₋₁ = Z * aₜ)
- `forecast`: Out-of-sample h-step ahead forecasts
- Missing data utilities (NaN-based)
"""

using LinearAlgebra
using .DSL: SSMSpec, build_linear_state_space

# ============================================
# Missing Data Utilities
# ============================================

"""
    ismissing_obs(y_t::AbstractVector) -> Bool

Check if observation vector contains any missing values (NaN).
"""
ismissing_obs(y_t::AbstractVector) = any(isnan, y_t)

"""
    ismissing_obs(y_t::Real) -> Bool

Check if scalar observation is missing (NaN).
"""
ismissing_obs(y_t::Real) = isnan(y_t)

"""
    missing_to_nan(y::AbstractArray)

Convert `missing` values to `NaN`. Returns a Float64 array.

Use this if your data contains `missing` values:
```julia
y_clean = missing_to_nan(y_with_missing)
```
"""
function missing_to_nan(y::AbstractArray{<:Union{Missing, Real}})
    result = Array{Float64}(undef, size(y))
    for i in eachindex(y)
        val = y[i]
        result[i] = ismissing(val) ? NaN : Float64(val)
    end
    result
end

missing_to_nan(y::AbstractArray{<:Real}) = convert(Array{Float64}, y)

# Edge case: array of only Missing values
missing_to_nan(y::AbstractArray{Missing}) = fill(NaN, size(y))

"""
    nan_to_missing(y::AbstractArray)

Convert `NaN` values to `missing`. Returns a Union{Float64, Missing} array.

Use this if you prefer working with `missing`:
```julia
y_missing = nan_to_missing(y_with_nan)
```
"""
function nan_to_missing(y::AbstractArray{<:Real})
    result = Array{Union{Float64, Missing}}(undef, size(y))
    for i in eachindex(y)
        result[i] = isnan(y[i]) ? missing : y[i]
    end
    result
end

"""
    count_missing(y::AbstractMatrix) -> Int

Count number of time periods with at least one missing observation.
"""
function count_missing(y::AbstractMatrix)
    n = size(y, 2)
    count = 0
    for t in 1:n
        if ismissing_obs(view(y, :, t))
            count += 1
        end
    end
    count
end

# ============================================
# Prediction (In-Sample Fitted Values)
# ============================================

"""
    predict(spec::SSMSpec, θ::NamedTuple, y::AbstractMatrix; use_static=true) -> NamedTuple

Compute in-sample one-step-ahead predictions for a state-space model.

For `n` observations, returns:
- `yhat`: One-step-ahead predictions ŷₜ|ₜ₋₁ = Z * aₜ (obs_dim × n)
- `a`: States (state_dim × (n+1)), where a[:, t] = E[αₜ | y₁:ₜ₋₁]
- `P`: State covariances (state_dim × state_dim × (n+1))
- `v`: Innovations vₜ = yₜ - ŷₜ|ₜ₋₁ (obs_dim × n)
- `F`: Innovation covariances (obs_dim × obs_dim × n)
- `loglik`: Log-likelihood

Note: `a[:, 1]` is the initial state, `a[:, n+1]` is the forecast state.

# Arguments
- `use_static::Bool=true`: Use StaticArrays for small matrices (dimensions ≤ 13)

# Example
```julia
spec = local_level()
θ = (σ_obs = 1.0, σ_level = 0.5)
y = randn(1, 100)

pred = predict(spec, θ, y)
plot(vec(y), label="observed")
plot!(vec(pred.yhat), label="predicted")
```
"""
function predict(spec::SSMSpec, θ::NamedTuple, y::AbstractMatrix; use_static::Bool = true)
    ss = build_linear_state_space(spec, θ, y; use_static = use_static)
    predict(ss.p, y, ss.a1, ss.P1)
end

"""
    predict(p::KFParms, y, a1, P1) -> NamedTuple

Low-level prediction using KFParms directly.

Returns predicted states (at = a_{t|t-1}) and related quantities.
"""
function predict(p::KFParms, y::AbstractMatrix, a1::AbstractVector, P1::AbstractMatrix)
    filt = kalman_filter(p, y, a1, P1)

    n = size(y, 2)
    obs_dim = size(y, 1)
    ET = eltype(filt.at)

    # Compute one-step-ahead predictions: ŷₜ|ₜ₋₁ = Z * at[:, t]
    yhat = Matrix{ET}(undef, obs_dim, n)
    for t in 1:n
        yhat[:, t] = p.Z * filt.at[:, t]
    end

    return (
        yhat = yhat,
        at = filt.at,
        Pt = filt.Pt,
        att = filt.att,
        Ptt = filt.Ptt,
        vt = filt.vt,
        Ft = filt.Ft,
        loglik = filt.loglik,
        missing_mask = filt.missing_mask
    )
end

# ============================================
# Forecasting (Out-of-Sample)
# ============================================

"""
    forecast(spec::SSMSpec, θ::NamedTuple, y::AbstractMatrix, h::Int; use_static=true) -> NamedTuple

Forecast h steps ahead beyond the observed data.

Returns a NamedTuple with:
- `yhat`: Forecasted observations (obs_dim × h)
- `a`: Forecasted states (state_dim × h)
- `P`: Forecasted state covariances (state_dim × state_dim × h)
- `F`: Forecasted observation covariances (obs_dim × obs_dim × h)

The forecast starts from `a[:, n+1]` = E[αₙ₊₁ | y₁:ₙ] from the filter.

# Arguments
- `use_static::Bool=true`: Use StaticArrays for small matrices (dimensions ≤ 13)

# Example
```julia
spec = local_level()
θ = (σ_obs = 1.0, σ_level = 0.5)
y = randn(1, 100)

fc = forecast(spec, θ, y, 12)  # 12-step ahead forecast
println("Forecast: ", fc.yhat)
println("Forecast std: ", sqrt.(fc.F[1,1,:]))
```
"""
function forecast(
        spec::SSMSpec,
        θ::NamedTuple,
        y::AbstractMatrix,
        h::Int;
        use_static::Bool = true
)
    ss = build_linear_state_space(spec, θ, y; use_static = use_static)
    forecast(ss.p, y, ss.a1, ss.P1, h)
end

"""
    forecast(p::KFParms, y, a1, P1, h) -> NamedTuple

Low-level forecasting using KFParms directly.

Starts from the last filtered state and propagates forward h steps.
"""
function forecast(
        p::KFParms,
        y::AbstractMatrix,
        a1::AbstractVector,
        P1::AbstractMatrix,
        h::Int
)
    filt = kalman_filter(p, y, a1, P1)

    n = size(y, 2)
    obs_dim = size(y, 1)
    state_dim = length(a1)
    ET = eltype(filt.at)

    # Compute one-step-ahead from last filtered state: a_{n+1|n} = T * a_{n|n}
    a = p.T * filt.att[:, n]
    P = p.T * filt.Ptt[:, :, n] * p.T' + p.R * p.Q * p.R'

    # Allocate forecast storage
    yhat = Matrix{ET}(undef, obs_dim, h)
    a_fc = Matrix{ET}(undef, state_dim, h)
    P_fc = Array{ET}(undef, state_dim, state_dim, h)
    F_fc = Array{ET}(undef, obs_dim, obs_dim, h)

    for j in 1:h
        # Store forecast state
        a_fc[:, j] = a
        P_fc[:, :, j] = P

        # Forecasted observation
        yhat[:, j] = p.Z * a
        F_fc[:, :, j] = p.Z * P * p.Z' + p.H

        # Predict next state (no update since no observation)
        a = p.T * a
        P = p.T * P * p.T' + p.R * p.Q * p.R'
    end

    return (yhat = yhat, a = a_fc, P = P_fc, F = F_fc)
end

"""
    forecast_paths(spec::SSMSpec, θ::NamedTuple, y::AbstractMatrix, h::Int, n_paths::Int; use_static=true) -> Array

Simulate n_paths forecast trajectories of length h.

Returns an array of size (obs_dim × h × n_paths).

Useful for fan charts and prediction intervals.

# Arguments
- `use_static::Bool=true`: Use StaticArrays for small matrices (dimensions ≤ 13)

# Example
```julia
paths = forecast_paths(spec, θ, y, 12, 1000)
quantiles = [quantile(paths[1, j, :], [0.1, 0.5, 0.9]) for j in 1:12]
```
"""
function forecast_paths(
        spec::SSMSpec,
        θ::NamedTuple,
        y::AbstractMatrix,
        h::Int,
        n_paths::Int;
        use_static::Bool = true
)
    ss = build_linear_state_space(spec, θ, y; use_static = use_static)
    forecast_paths(ss.p, y, ss.a1, ss.P1, h, n_paths)
end

function forecast_paths(
        p::KFParms,
        y::AbstractMatrix,
        a1::AbstractVector,
        P1::AbstractMatrix,
        h::Int,
        n_paths::Int
)
    filt = kalman_filter(p, y, a1, P1)

    n = size(y, 2)
    obs_dim = size(y, 1)
    state_dim = length(a1)

    # Compute one-step-ahead from last filtered state: a_{n+1|n} = T * a_{n|n}
    a_start = p.T * filt.att[:, n]
    P_start = p.T * filt.Ptt[:, :, n] * p.T' + p.R * p.Q * p.R'

    # Cholesky factors
    chol_P = cholesky(Symmetric(P_start)).L
    chol_Q = cholesky(Symmetric(p.Q)).L
    chol_H = cholesky(Symmetric(p.H)).L

    paths = Array{Float64}(undef, obs_dim, h, n_paths)

    for s in 1:n_paths
        # Sample initial state from N(a_start, P_start)
        a = a_start + chol_P * randn(state_dim)

        for j in 1:h
            # Observation with noise
            paths[:, j, s] = p.Z * a + chol_H * randn(obs_dim)

            # State transition with noise
            a = p.T * a + p.R * chol_Q * randn(size(p.Q, 1))
        end
    end

    paths
end
