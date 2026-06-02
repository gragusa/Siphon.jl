"""
    recipes.jl

RecipesBase plotting recipes for Kalman filter/smoother visualization.
Provides recipes for plotting filtered states, smoothed states, and forecasts
with confidence bands.
"""

using RecipesBase

# ============================================================================
# Helper Functions
# ============================================================================

"""
    quantile_normal(p::Real)

Compute the quantile (inverse CDF) of the standard normal distribution.

Uses a rational approximation accurate to about 4.5e-4 for p in (0.5, 1).

# Arguments
- `p`: Probability value in (0.5, 1)

# Returns
- Quantile value z such that P(Z ≤ z) = p for Z ~ N(0,1)
"""
function quantile_normal(p::Real)
    # Abramowitz & Stegun approximation 26.2.23
    t = sqrt(-2 * log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1*t + c2*t^2) / (1 + d1*t + d2*t^2 + d3*t^3)
end

"""
    confidence_bands(mean::AbstractVector, var::AbstractVector, level::Real=0.95)

Compute confidence band bounds for a vector of means and variances.

# Arguments
- `mean`: Vector of point estimates
- `var`: Vector of variances (must be same length as mean)
- `level`: Confidence level in (0, 1), default 0.95

# Returns
- `(lower, upper)`: Tuple of vectors with lower and upper bounds
"""
function confidence_bands(mean::AbstractVector, var::AbstractVector, level::Real = 0.95)
    z = quantile_normal((1 + level) / 2)
    std = sqrt.(var)
    lower = mean .- z .* std
    upper = mean .+ z .* std
    return lower, upper
end

"""
    confidence_bands(mean::Matrix, cov::Array{T,3}, level::Real=0.95) where T

Compute confidence bands for matrix of means and 3D array of covariances.

Extracts diagonal variances from covariance matrices.

# Arguments
- `mean`: Matrix of point estimates (m × n)
- `cov`: 3D array of covariances (m × m × n)
- `level`: Confidence level in (0, 1), default 0.95

# Returns
- `(lower, upper)`: Tuple of matrices (m × n) with lower and upper bounds
"""
function confidence_bands(mean::Matrix{T}, cov::Array{T, 3}, level::Real = 0.95) where {T}
    m, n = size(mean)
    z = quantile_normal((1 + level) / 2)
    lower = similar(mean)
    upper = similar(mean)

    @inbounds for t in 1:n
        for i in 1:m
            std_i = sqrt(cov[i, i, t])
            lower[i, t] = mean[i, t] - z * std_i
            upper[i, t] = mean[i, t] + z * std_i
        end
    end

    return lower, upper
end

"""
    select_vars(vars, n_vars::Int) -> AbstractVector{Int}

Parse variable selection keyword into vector of indices.

# Arguments
- `vars`: Variable selection - can be:
  - `:all` or `nothing`: Select all variables (1:n_vars)
  - `Int`: Single variable index
  - `Vector{Int}` or `AbstractRange`: Multiple variable indices
- `n_vars`: Total number of variables available

# Returns
- Vector or range of selected variable indices

# Examples
```julia
select_vars(:all, 5)     # 1:5
select_vars(3, 5)        # [3]
select_vars([1,3], 5)    # [1, 3]
select_vars(2:4, 5)      # 2:4
```
"""
function select_vars(vars, n_vars::Int)
    if vars === :all || vars === nothing
        return 1:n_vars
    elseif vars isa Integer
        1 ≤ vars ≤ n_vars ||
            throw(ArgumentError("Variable index $vars out of range [1, $n_vars]"))
        return [vars]
    elseif vars isa AbstractVector{<:Integer}
        all(1 .≤ vars .≤ n_vars) ||
            throw(ArgumentError("Variable indices out of range [1, $n_vars]"))
        return vars
    elseif vars isa AbstractRange
        all(1 .≤ vars .≤ n_vars) ||
            throw(ArgumentError("Variable indices out of range [1, $n_vars]"))
        return vars
    else
        throw(
            ArgumentError(
            "vars must be :all, an integer, vector of integers, or range. Got: $(typeof(vars))",
        ),
        )
    end
end

# ============================================================================
# Wrapper Types for Plotting
# ============================================================================

"""
    SmootherResult{T<:Real}

Wrapper for Kalman smoother output to enable plot recipes.

This is a convenience wrapper around the NamedTuple returned by `kalman_smoother()`.
The underlying `kalman_smoother()` function is unchanged and continues to return
its original NamedTuple format.

# Fields
- `alpha::Matrix{T}`: Smoothed states E[αₜ | y₁:ₙ] (m × n)
- `V::Array{T,3}`: Smoothed covariances Var[αₜ | y₁:ₙ] (m × m × n)
- `p::Union{KFParms, Nothing}`: Optional model parameters for observable predictions
- `time::Union{AbstractVector, Nothing}`: Optional time axis for plotting

# Constructors
```julia
# Wrap existing smoother output
nt = kalman_smoother(Z, T, at, Pt, vt, Ft)
sr = SmootherResult(nt)

# Convenience: run filter+smoother and wrap
sr = SmootherResult(p, y, a1, P1)
```
"""
struct SmootherResult{T <: Real}
    alpha::Matrix{T}
    V::Array{T, 3}
    p::Union{KFParms, Nothing}
    time::Union{AbstractVector, Nothing}
end

# Constructor from NamedTuple (output of kalman_smoother)
function SmootherResult(
        nt::NamedTuple;
        p::Union{KFParms, Nothing} = nothing,
        time::Union{AbstractVector, Nothing} = nothing
)
    SmootherResult(nt.alpha, nt.V, p, time)
end

# Convenience constructor: run filter+smoother (functional API)
function SmootherResult(
        p::KFParms,
        y::AbstractMatrix,
        a1,
        P1;
        time::Union{AbstractVector, Nothing} = nothing
)
    filt = kalman_filter(p, y, a1, P1)
    nt = kalman_smoother(
        p.Z,
        p.T,
        filt.at,
        filt.Pt,
        filt.vt,
        filt.Ft;
        missing_mask = filt.missing_mask,
        Ptt = filt.Ptt
    )
    SmootherResult(nt; p = p, time = time)
end

# Constructor from KalmanWorkspace (in-place API - RECOMMENDED)
"""
    SmootherResult(ws::KalmanWorkspace; time=nothing)

Create SmootherResult from a KalmanWorkspace after running `kalman_smoother!(ws)`.

This is the recommended approach for large models as it uses the
zero-allocation in-place filter and smoother.

# Example
```julia
ws = KalmanWorkspace(Z, H, T, R, Q, a1, P1, n)
kalman_filter!(ws, y)
kalman_smoother!(ws)
smooth = SmootherResult(ws)
plot(smooth)
```
"""
function SmootherResult(
        ws::KalmanWorkspace{T};
        time::Union{AbstractVector, Nothing} = nothing
) where {T}
    SmootherResult{T}(ws.αs, ws.Vs, nothing, time)
end

# Accessor methods
smoothed_states(r::SmootherResult) = r.alpha
variances_smoothed_states(r::SmootherResult) = r.V

"""
    ForecastResult{T<:Real}

Wrapper for forecast output to enable plot recipes.

This is a convenience wrapper around the NamedTuple returned by `forecast()`.
The underlying `forecast()` function is unchanged.

# Fields
- `yhat::Matrix{T}`: Forecasted observations (p × h)
- `a::Matrix{T}`: Forecasted states (m × h)
- `P::Array{T,3}`: State forecast covariances (m × m × h)
- `F::Array{T,3}`: Observation forecast covariances (p × p × h)
- `time::Union{AbstractVector, Nothing}`: Optional time axis for forecast horizon

# Constructors
```julia
# Wrap existing forecast output
fc_nt = forecast(spec, θ, y, h)
fc = ForecastResult(fc_nt)

# With custom time axis
fc = ForecastResult(fc_nt; time=101:112)
```
"""
struct ForecastResult{T <: Real}
    yhat::Matrix{T}
    a::Matrix{T}
    P::Array{T, 3}
    F::Array{T, 3}
    time::Union{AbstractVector, Nothing}
end

# Constructor from NamedTuple (output of forecast)
function ForecastResult(nt::NamedTuple; time::Union{AbstractVector, Nothing} = nothing)
    ForecastResult(nt.yhat, nt.a, nt.P, nt.F, time)
end

# Accessor methods
forecast_observations(r::ForecastResult) = r.yhat
forecast_states(r::ForecastResult) = r.a
variances_forecast_states(r::ForecastResult) = r.P
variances_forecast_observations(r::ForecastResult) = r.F

# ============================================================================
# Color Scheme
# ============================================================================

const FILTER_COLOR = :steelblue
const FILTER_FILL = :lightblue
const SMOOTHER_COLOR = :darkgreen
const SMOOTHER_FILL = :lightgreen
const FORECAST_COLOR = :darkorange
const FORECAST_FILL = :moccasin  # Light orange
const PREDICTED_COLOR = :purple
const PREDICTED_FILL = :lavender  # Light purple

# ============================================================================
# KalmanFilterResult Recipe (States)
# ============================================================================

"""
    @recipe for KalmanFilterResult

Plot filtered or predicted states with confidence bands.

# Keywords
- `vars=:all`: Which state variables to plot (Int, Vector, Range, :all)
- `level=0.95`: Confidence level for bands
- `filtered=true`: Plot filtered states (att); if false, plot predicted (at)
- `band=true`: Show confidence bands
- `time=nothing`: Custom time axis (default: 1:n)

# Examples
```julia
result = kalman_filter(p, y, a1, P1)
plot(result)                    # All filtered states
plot(result, vars=1)            # State 1 only
plot(result, vars=[1,3])        # States 1 and 3
plot(result, level=0.90)        # 90% CI
plot(result, band=false)        # No confidence bands
plot(result, filtered=false)    # Predicted instead of filtered
```
"""
@recipe function f(result::KalmanFilterResult)
    # Extract options
    vars = get(plotattributes, :vars, :all)
    level = get(plotattributes, :level, 0.95)
    filtered_opt = get(plotattributes, :filtered, true)
    band = get(plotattributes, :band, true)
    time_axis = get(plotattributes, :time, nothing)

    # Select data based on filtered vs predicted
    if filtered_opt
        states = result.att
        covs = result.Ptt
        plot_title = "Filtered States"
        line_color = FILTER_COLOR
        fill_color = FILTER_FILL
    else
        states = result.at
        covs = result.Pt
        plot_title = "Predicted States"
        line_color = PREDICTED_COLOR
        fill_color = PREDICTED_FILL
    end

    m, n = size(states)
    var_indices = select_vars(vars, m)
    n_vars = length(var_indices)

    # Time axis
    t_axis = time_axis !== nothing ? time_axis : 1:n

    # Layout for multiple variables
    if n_vars > 1
        layout --> (n_vars, 1)
    end

    # Global defaults
    legend --> :topright
    xlabel --> "Time"

    for (subplot_idx, var_idx) in enumerate(var_indices)
        state_mean = states[var_idx, :]
        state_var = [covs[var_idx, var_idx, t] for t in 1:n]

        # Confidence bands
        if band
            lower, upper = confidence_bands(state_mean, state_var, level)

            @series begin
                subplot := subplot_idx
                seriestype := :path
                primary := false
                linecolor := nothing
                fillcolor --> fill_color
                fillalpha --> 0.4
                fillrange := lower
                label := ""
                t_axis, upper
            end
        end

        # Mean series
        @series begin
            subplot := subplot_idx
            seriestype := :path
            linewidth --> 2
            linecolor --> line_color
            label --> (filtered_opt ? "Filtered" : "Predicted")
            ylabel --> "State $var_idx"
            title --> (subplot_idx == 1 ? plot_title : "")
            t_axis, state_mean
        end
    end
end

# ============================================================================
# KalmanWorkspace Recipe (In-place API - RECOMMENDED)
# ============================================================================

"""
    @recipe for KalmanWorkspace

Plot filtered, predicted, or smoothed states from in-place workspace.

This is the recommended plotting approach for large models as it uses
the zero-allocation in-place filter and smoother.

# Keywords
- `vars=:all`: Which state variables to plot (Int, Vector, Range, :all)
- `level=0.95`: Confidence level for bands
- `what=:smoothed`: What to plot - `:filtered`, `:predicted`, or `:smoothed`
- `band=true`: Show confidence bands
- `time=nothing`: Custom time axis (default: 1:n)

# Examples
```julia
ws = KalmanWorkspace(Z, H, T, R, Q, a1, P1, n)
kalman_filter!(ws, y)
kalman_smoother!(ws)

plot(ws)                        # Smoothed states (default)
plot(ws, what=:filtered)        # Filtered states
plot(ws, what=:predicted)       # Predicted states
plot(ws, vars=1, level=0.90)    # State 1 with 90% CI
```
"""
@recipe function f(ws::KalmanWorkspace)
    vars = get(plotattributes, :vars, :all)
    level = get(plotattributes, :level, 0.95)
    what = get(plotattributes, :what, :smoothed)
    band = get(plotattributes, :band, true)
    time_axis = get(plotattributes, :time, nothing)

    # Select data based on what to plot
    if what === :smoothed
        states = ws.αs
        covs = ws.Vs
        plot_title = "Smoothed States"
        line_color = SMOOTHER_COLOR
        fill_color = SMOOTHER_FILL
    elseif what === :filtered
        states = ws.att
        covs = ws.Ptt
        plot_title = "Filtered States"
        line_color = FILTER_COLOR
        fill_color = FILTER_FILL
    else  # :predicted
        states = ws.at
        covs = ws.Pt
        plot_title = "Predicted States"
        line_color = PREDICTED_COLOR
        fill_color = PREDICTED_FILL
    end

    m, n = size(states)
    var_indices = select_vars(vars, m)
    n_vars = length(var_indices)

    # Time axis
    t_axis = time_axis !== nothing ? time_axis : 1:n

    if n_vars > 1
        layout --> (n_vars, 1)
    end

    legend --> :topright
    xlabel --> "Time"

    for (subplot_idx, var_idx) in enumerate(var_indices)
        state_mean = states[var_idx, :]
        state_var = [covs[var_idx, var_idx, t] for t in 1:n]

        if band
            lower, upper = confidence_bands(state_mean, state_var, level)

            @series begin
                subplot := subplot_idx
                seriestype := :path
                primary := false
                linecolor := nothing
                fillcolor --> fill_color
                fillalpha --> 0.4
                fillrange := lower
                label := ""
                t_axis, upper
            end
        end

        @series begin
            subplot := subplot_idx
            seriestype := :path
            linewidth --> 2
            linecolor --> line_color
            label --> string(titlecase(string(what)))
            ylabel --> "State $var_idx"
            title --> (subplot_idx == 1 ? plot_title : "")
            t_axis, state_mean
        end
    end
end

# ============================================================================
# SmootherResult Recipe
# ============================================================================

"""
    @recipe for SmootherResult

Plot smoothed states with confidence bands.

# Keywords
- `vars=:all`: Which state variables to plot
- `level=0.95`: Confidence level for bands
- `band=true`: Show confidence bands

# Examples
```julia
smooth = SmootherResult(p, y, a1, P1)
plot(smooth)                       # All smoothed states
plot(smooth, vars=1, level=0.90)   # State 1 with 90% CI
```
"""
@recipe function f(result::SmootherResult)
    vars = get(plotattributes, :vars, :all)
    level = get(plotattributes, :level, 0.95)
    band = get(plotattributes, :band, true)

    alpha = result.alpha
    V = result.V

    m, n = size(alpha)
    var_indices = select_vars(vars, m)
    n_vars = length(var_indices)

    # Time axis
    t_axis = result.time !== nothing ? result.time : 1:n

    if n_vars > 1
        layout --> (n_vars, 1)
    end

    legend --> :topright
    xlabel --> "Time"

    for (subplot_idx, var_idx) in enumerate(var_indices)
        state_mean = alpha[var_idx, :]
        state_var = [V[var_idx, var_idx, t] for t in 1:n]

        if band
            lower, upper = confidence_bands(state_mean, state_var, level)

            @series begin
                subplot := subplot_idx
                seriestype := :path
                primary := false
                linecolor := nothing
                fillcolor --> SMOOTHER_FILL
                fillalpha --> 0.4
                fillrange := lower
                label := ""
                t_axis, upper
            end
        end

        @series begin
            subplot := subplot_idx
            seriestype := :path
            linewidth --> 2
            linecolor --> SMOOTHER_COLOR
            label --> "Smoothed"
            ylabel --> "State $var_idx"
            title --> (subplot_idx == 1 ? "Smoothed States" : "")
            t_axis, state_mean
        end
    end
end

# ============================================================================
# ForecastResult Recipe
# ============================================================================

"""
    @recipe for ForecastResult

Plot forecasted states or observations with confidence bands.

# Keywords
- `vars=:all`: Which variables to plot
- `level=0.95`: Confidence level for bands
- `what=:observations`: Plot :observations (yhat) or :states (a)
- `band=true`: Show confidence bands

# Examples
```julia
fc = ForecastResult(forecast_output)
plot(fc)                           # Forecasted observations (default)
plot(fc, what=:states, vars=1)     # Forecasted state 1
plot(fc, level=0.90)               # 90% CI
```
"""
@recipe function f(result::ForecastResult)
    vars = get(plotattributes, :vars, :all)
    level = get(plotattributes, :level, 0.95)
    what = get(plotattributes, :what, :observations)
    band = get(plotattributes, :band, true)

    if what === :observations
        means = result.yhat
        covs = result.F
        ylabel_prefix = "Obs"
        plot_title = "Observation Forecasts"
    else  # :states
        means = result.a
        covs = result.P
        ylabel_prefix = "State"
        plot_title = "State Forecasts"
    end

    dim, h = size(means)
    var_indices = select_vars(vars, dim)
    n_vars = length(var_indices)

    # Time axis
    t_axis = result.time !== nothing ? result.time : 1:h

    if n_vars > 1
        layout --> (n_vars, 1)
    end

    legend --> :topright
    xlabel --> "Forecast Horizon"

    for (subplot_idx, var_idx) in enumerate(var_indices)
        fc_mean = means[var_idx, :]
        fc_var = [covs[var_idx, var_idx, t] for t in 1:h]

        if band
            lower, upper = confidence_bands(fc_mean, fc_var, level)

            @series begin
                subplot := subplot_idx
                seriestype := :path
                primary := false
                linecolor := nothing
                fillcolor --> FORECAST_FILL
                fillalpha --> 0.4
                fillrange := lower
                label := ""
                t_axis, upper
            end
        end

        @series begin
            subplot := subplot_idx
            seriestype := :path
            linewidth --> 2
            linecolor --> FORECAST_COLOR
            linestyle --> :dash
            label --> "Forecast"
            ylabel --> "$ylabel_prefix $var_idx"
            title --> (subplot_idx == 1 ? plot_title : "")
            t_axis, fc_mean
        end
    end
end

# ============================================================================
# Comparison Recipe: Filtered vs Smoothed
# ============================================================================

"""
    @recipe for (KalmanFilterResult, SmootherResult)

Plot filtered and smoothed states together for comparison.

# Keywords
- `vars=:all`: Which state variables to plot
- `level=0.95`: Confidence level for bands (applied to smoother only)
- `band=true`: Show confidence bands for smoother

# Example
```julia
result = kalman_filter(p, y, a1, P1)
smooth = SmootherResult(p, y, a1, P1)
plot((result, smooth), vars=1)  # Compare state 1
```
"""
@recipe function f(results::Tuple{KalmanFilterResult, SmootherResult})
    filt, smooth = results

    vars = get(plotattributes, :vars, :all)
    level = get(plotattributes, :level, 0.95)
    band = get(plotattributes, :band, true)

    m, n = size(filt.att)
    var_indices = select_vars(vars, m)
    n_vars = length(var_indices)

    # Time axis
    t_axis = smooth.time !== nothing ? smooth.time : 1:n

    if n_vars > 1
        layout --> (n_vars, 1)
    end

    legend --> :topright
    xlabel --> "Time"
    title --> "Filtered vs Smoothed States"

    for (subplot_idx, var_idx) in enumerate(var_indices)
        # Smoother confidence band (if requested)
        if band
            smooth_mean = smooth.alpha[var_idx, :]
            smooth_var = [smooth.V[var_idx, var_idx, t] for t in 1:n]
            lower, upper = confidence_bands(smooth_mean, smooth_var, level)

            @series begin
                subplot := subplot_idx
                seriestype := :path
                primary := false
                linecolor := nothing
                fillcolor --> SMOOTHER_FILL
                fillalpha --> 0.3
                fillrange := lower
                label := ""
                t_axis, upper
            end
        end

        # Filtered
        @series begin
            subplot := subplot_idx
            seriestype := :path
            linewidth --> 1.5
            linecolor --> FILTER_COLOR
            linestyle --> :dash
            label --> "Filtered"
            t_axis, filt.att[var_idx, :]
        end

        # Smoothed
        @series begin
            subplot := subplot_idx
            seriestype := :path
            linewidth --> 2
            linecolor --> SMOOTHER_COLOR
            label --> "Smoothed"
            ylabel --> "State $var_idx"
            t_axis, smooth.alpha[var_idx, :]
        end
    end
end

# ============================================================================
# Observable Prediction Recipe (One-step-ahead predictions)
# ============================================================================

"""
    ObservablePlot

Marker type for plotting observable predictions instead of states.

Use with KalmanFilterResult to plot one-step-ahead predictions:
```julia
plot(result, ObservablePlot)
```
"""
struct ObservablePlot end

"""
    @recipe for (KalmanFilterResult, Type{ObservablePlot})

Plot one-step-ahead predictions of observables with confidence bands.

# Keywords
- `vars=:all`: Which observable variables to plot
- `level=0.95`: Confidence level for bands
- `band=true`: Show confidence bands
- `actual=nothing`: Actual observations matrix to overlay (p × n)

# Examples
```julia
result = kalman_filter(p, y, a1, P1)
plot(result, ObservablePlot)                # One-step-ahead predictions
plot(result, ObservablePlot, actual=y)      # With actual data overlay
```
"""
@recipe function f(result::KalmanFilterResult, ::Type{ObservablePlot})
    vars = get(plotattributes, :vars, :all)
    level = get(plotattributes, :level, 0.95)
    band = get(plotattributes, :band, true)
    actual = get(plotattributes, :actual, nothing)
    time_axis = get(plotattributes, :time, nothing)

    # One-step-ahead predictions: yhat_t = Z * a_t (predicted state before update)
    Z = result.p.Z
    at = result.at
    Pt = result.Pt
    H = result.p.H

    _, n = size(at)
    p_obs = size(Z, 1)

    # Compute predictions
    yhat = Z * at  # p × n

    var_indices = select_vars(vars, p_obs)
    n_vars = length(var_indices)

    t_axis = time_axis !== nothing ? time_axis : 1:n

    if n_vars > 1
        layout --> (n_vars, 1)
    end

    legend --> :topright
    xlabel --> "Time"

    for (subplot_idx, var_idx) in enumerate(var_indices)
        pred_mean = yhat[var_idx, :]

        # Variance of prediction: Z[i,:] * P_t * Z[i,:]' + H[i,i]
        if band
            pred_var = zeros(n)
            z_row = Z[var_idx, :]
            h_var = H[var_idx, var_idx]
            for t in 1:n
                pred_var[t] = dot(z_row, Pt[:, :, t] * z_row) + h_var
            end

            lower, upper = confidence_bands(pred_mean, pred_var, level)

            @series begin
                subplot := subplot_idx
                seriestype := :path
                primary := false
                linecolor := nothing
                fillcolor --> FILTER_FILL
                fillalpha --> 0.4
                fillrange := lower
                label := ""
                t_axis, upper
            end
        end

        # Prediction line
        @series begin
            subplot := subplot_idx
            seriestype := :path
            linewidth --> 2
            linecolor --> FILTER_COLOR
            label --> "Prediction"
            ylabel --> "Observable $var_idx"
            title --> (subplot_idx == 1 ? "One-Step-Ahead Predictions" : "")
            t_axis, pred_mean
        end

        # Actual observations overlay (if provided)
        if actual !== nothing
            @series begin
                subplot := subplot_idx
                seriestype := :scatter
                markersize --> 3
                markercolor --> :black
                markerstrokewidth --> 0
                label --> "Actual"
                t_axis, actual[var_idx, :]
            end
        end
    end
end
