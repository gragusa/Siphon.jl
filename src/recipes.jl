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

# Standard normal quantile: the `z` with `P(Z ≤ z) = p` for `Z ~ N(0, 1)`.
function _quantile_normal(p::Real)
    0 < p < 1 || throw(DomainError(p, "the normal quantile requires 0 < p < 1"))
    return -sqrt(oftype(float(p), 2)) * erfcinv(2 * p)
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
    z = _band_quantile(level)
    std = sqrt.(var)
    lower = mean .- z .* std
    upper = mean .+ z .* std
    return lower, upper
end

"""
    confidence_bands(mean::AbstractMatrix, cov::AbstractArray{<:Real,3}, level::Real=0.95)

Compute confidence bands for matrix of means and 3D array of covariances.

Extracts diagonal variances from covariance matrices.

# Arguments
- `mean`: Matrix of point estimates (m × n)
- `cov`: 3D array of covariances (m × m × n)
- `level`: Confidence level in (0, 1), default 0.95

# Returns
- `(lower, upper)`: Tuple of matrices (m × n) with lower and upper bounds
"""
function confidence_bands(
        mean::AbstractMatrix, cov::AbstractArray{<:Real, 3}, level::Real = 0.95)
    m, n = size(mean)
    size(cov) == (m, m, n) ||
        throw(DimensionMismatch("cov must be ($m, $m, $n) for a ($m, $n) mean, got $(size(cov))"))
    z = _band_quantile(level)
    T = promote_type(eltype(mean), eltype(cov), typeof(z))
    lower = similar(mean, T)
    upper = similar(mean, T)

    @inbounds for t in 1:n
        for i in 1:m
            std_i = sqrt(cov[i, i, t])
            lower[i, t] = mean[i, t] - z * std_i
            upper[i, t] = mean[i, t] + z * std_i
        end
    end

    return lower, upper
end

function _band_quantile(level::Real)
    0 < level < 1 || throw(DomainError(level, "confidence level must be in (0, 1)"))
    return _quantile_normal((1 + level) / 2)
end

# Indices selected by a `vars` plot keyword: `:all`/`nothing` (every index), an
# integer, a vector of integers or a range, each checked against `1:n_vars`.
function _select_vars(vars, n_vars::Int)
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
# Smoother and forecast results
# ============================================================================

"""
    SmootherResult(ws::AbstractFilterWorkspace)

Copy the smoother output of `ws` into a [`SmootherResult`](@ref). Needs a prior
`kalman_smoother!(ws)` run. The cross-lag covariances are
copied when that run computed them. Later runs on `ws` don't change the result.

# Example
```julia
ws = KalmanWorkspace(Z, H, T, R, Q, a1, P1, n)
kalman_filter_and_smooth!(ws, y)
smooth = SmootherResult(ws)
plot(smooth)
```
"""
function SmootherResult(ws::AbstractFilterWorkspace)
    store = _filter_store(_require_smoothed_cov(ws, :SmootherResult))
    Pcross = store.crosscov ? copy(store.Pcross) : nothing
    return SmootherResult(copy(store.αs), copy(store.Vs), Pcross)
end

"""
    SmootherResult(model::StateSpaceModel)

Copy the smoothed states and covariances of a fitted model into a
[`SmootherResult`](@ref). The smoother runs first if it has not yet.
"""
function SmootherResult(model::StateSpaceModel)
    return SmootherResult(copy(smoothed_states(model)), copy(smoothed_states_cov(model)),
        nothing)
end

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
    var_indices = _select_vars(vars, m)
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
        states = smoothed_states(ws)
        covs = smoothed_states_cov(ws)
        plot_title = "Smoothed States"
        line_color = SMOOTHER_COLOR
        fill_color = SMOOTHER_FILL
    elseif what === :filtered
        states = filtered_states(ws)
        covs = filtered_states_cov(ws)
        plot_title = "Filtered States"
        line_color = FILTER_COLOR
        fill_color = FILTER_FILL
    else  # :predicted
        states = predicted_states(ws)
        covs = predicted_states_cov(ws)
        plot_title = "Predicted States"
        line_color = PREDICTED_COLOR
        fill_color = PREDICTED_FILL
    end

    m, n = size(states)
    var_indices = _select_vars(vars, m)
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

- `time=nothing`: Custom time axis (default: 1:n)

# Examples
```julia
smooth = kalman_smoother(p, y, a1, P1)
plot(smooth)                       # All smoothed states
plot(smooth, vars=1, level=0.90)   # State 1 with 90% CI
```
"""
@recipe function f(result::SmootherResult)
    vars = get(plotattributes, :vars, :all)
    level = get(plotattributes, :level, 0.95)
    band = get(plotattributes, :band, true)
    time_axis = get(plotattributes, :time, nothing)

    alpha = result.αs
    V = result.Vs

    m, n = size(alpha)
    var_indices = _select_vars(vars, m)
    n_vars = length(var_indices)

    # Time axis
    t_axis = time_axis !== nothing ? time_axis : 1:n

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
- `time=nothing`: Custom time axis (default: `forecast_periods(fc)`)

# Examples
```julia
fc = forecast(model, 12)
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
    var_indices = _select_vars(vars, dim)
    n_vars = length(var_indices)

    time_axis = get(plotattributes, :time, nothing)
    t_axis = time_axis !== nothing ? time_axis : forecast_periods(result)

    if n_vars > 1
        layout --> (n_vars, 1)
    end

    legend --> :topright
    xlabel --> "Time"

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
- `time=nothing`: Custom time axis (default: 1:n)

# Example
```julia
plot(kalman_filter_and_smooth(p, y, a1, P1), vars=1)  # Compare state 1
```
"""
@recipe function f(results::Tuple{KalmanFilterResult, SmootherResult})
    filt, smooth = results

    vars = get(plotattributes, :vars, :all)
    level = get(plotattributes, :level, 0.95)
    band = get(plotattributes, :band, true)
    time_axis = get(plotattributes, :time, nothing)

    m, n = size(filt.att)
    var_indices = _select_vars(vars, m)
    n_vars = length(var_indices)

    # Time axis
    t_axis = time_axis !== nothing ? time_axis : 1:n

    if n_vars > 1
        layout --> (n_vars, 1)
    end

    legend --> :topright
    xlabel --> "Time"
    title --> "Filtered vs Smoothed States"

    for (subplot_idx, var_idx) in enumerate(var_indices)
        # Smoother confidence band (if requested)
        if band
            smooth_mean = smooth.αs[var_idx, :]
            smooth_var = [smooth.Vs[var_idx, var_idx, t] for t in 1:n]
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
            t_axis, smooth.αs[var_idx, :]
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

    var_indices = _select_vars(vars, p_obs)
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
