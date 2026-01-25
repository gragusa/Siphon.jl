# Visualization

Siphon.jl provides plotting recipes via [RecipesBase.jl](https://github.com/JuliaPlots/RecipesBase.jl) for visualizing Kalman filter and smoother outputs with confidence bands. These recipes work with any Plots.jl backend.

## Quick Start (Recommended: KalmanWorkspace)

The recommended approach uses `KalmanWorkspace` for zero-allocation, in-place computation:

```julia
using Plots
using Siphon

# Create workspace and run filter + smoother
ws = KalmanWorkspace(Z, H, T, R, Q, a1, P1, n)
kalman_filter!(ws, y)
kalman_smoother!(ws)

# Plot smoothed states with 95% confidence bands
plot(ws)
```

## Plotting with KalmanWorkspace

The `KalmanWorkspace` recipe is the primary plotting interface. It supports filtered, predicted, and smoothed states:

```julia
# Create and run
ws = KalmanWorkspace(Z, H, T, R, Q, a1, P1, n)
kalman_filter!(ws, y)
kalman_smoother!(ws)

# Plot smoothed states (default)
plot(ws)
plot(ws, what=:smoothed)

# Plot filtered or predicted states
plot(ws, what=:filtered)
plot(ws, what=:predicted)

# Customize
plot(ws, vars=1, level=0.90)     # Single state with 90% CI
plot(ws, vars=[1,3])             # Multiple states
plot(ws, band=false)             # No confidence bands
plot(ws, time=dates)             # Custom time axis
```

### Keywords

| Keyword | Default | Description |
|---------|---------|-------------|
| `vars` | `:all` | Variables to plot: Int, Vector, Range, or `:all` |
| `level` | `0.95` | Confidence level for bands (0 < level < 1) |
| `what` | `:smoothed` | State type: `:smoothed`, `:filtered`, or `:predicted` |
| `band` | `true` | Whether to show confidence bands |
| `time` | `nothing` | Custom time axis (default: 1:n) |

## Functional API (Alternative)

For simpler cases or when AD compatibility is needed, you can use the functional API:

```julia
# Define model and run filter
p = KFParms(Z, H, T, R, Q)
result = kalman_filter(p, y, a1, P1)

# Plot filtered states with 95% confidence bands
plot(result)
plot(result, vars=1)             # State 1 only
plot(result, vars=[1,3])         # States 1 and 3
plot(result, level=0.90)         # 90% CI
plot(result, band=false)         # No confidence bands
plot(result, filtered=false)     # Predicted states instead of filtered
```

## Plotting Smoothed States

The `SmootherResult` wrapper enables plotting smoothed states from any source:

```julia
# Method 1: From KalmanWorkspace (recommended)
ws = KalmanWorkspace(Z, H, T, R, Q, a1, P1, n)
kalman_filter!(ws, y)
kalman_smoother!(ws)
smooth = SmootherResult(ws)
plot(smooth)

# Method 2: Convenience constructor (runs filter + smoother internally)
p = KFParms(Z, H, T, R, Q)
smooth = SmootherResult(p, y, a1, P1)
plot(smooth, vars=1, level=0.90)

# Method 3: Wrap existing smoother NamedTuple output
filt = kalman_filter(p, y, a1, P1)
nt = kalman_smoother(p.Z, p.T, filt.at, filt.Pt, filt.vt, filt.Ft)
smooth = SmootherResult(nt)
plot(smooth)
```

### Keywords

| Keyword | Default | Description |
|---------|---------|-------------|
| `vars` | `:all` | Variables to plot |
| `level` | `0.95` | Confidence level for bands |
| `band` | `true` | Whether to show confidence bands |

## Plotting Forecasts

Wrap forecast output in `ForecastResult`:

```julia
# Run forecast
fc_nt = forecast(spec, θ, y, 24)  # 24-step ahead forecast
fc = ForecastResult(fc_nt)

# Plot forecasted observations (default)
plot(fc)
plot(fc, what=:observations)

# Plot forecasted states
plot(fc, what=:states)
plot(fc, what=:states, vars=1)
```

### Keywords

| Keyword | Default | Description |
|---------|---------|-------------|
| `vars` | `:all` | Variables to plot |
| `level` | `0.95` | Confidence level for bands |
| `what` | `:observations` | Plot `:observations` or `:states` |
| `band` | `true` | Whether to show confidence bands |

## Comparing Filtered vs Smoothed

Plot both filtered and smoothed states together:

```julia
result = kalman_filter(p, y, a1, P1)
smooth = SmootherResult(p, y, a1, P1)

# Compare on same plot
plot((result, smooth), vars=1)
plot((result, smooth), vars=[1,2], band=true)
```

The comparison shows filtered states as dashed blue lines and smoothed states as solid green lines, with optional confidence bands around the smoother.

## Plotting Observable Predictions

Plot one-step-ahead predictions of observables (y_hat_t = Z * a_t):

```julia
result = kalman_filter(p, y, a1, P1)

# One-step-ahead predictions
plot(result, ObservablePlot)

# With actual observations overlay
plot(result, ObservablePlot, actual=y)

# Select specific observables
plot(result, ObservablePlot, vars=1, level=0.90)
```

## Complete Example: Nile River Flow

```julia
using Plots
using Siphon
using DelimitedFiles

# Load Nile data
nile = readdlm("Nile.csv", ',', Float64)
y = reshape(nile[:, 1], 1, :)
n = size(y, 2)

# Local level model parameters (MLE estimates)
Z = [1.0;;]       # Observation matrix
H = [15099.0;;]   # Observation variance
T = [1.0;;]       # Transition matrix
R = [1.0;;]       # Selection matrix
Q = [1469.0;;]    # State variance
a1 = [0.0]        # Initial state mean
P1 = [1e7;;]      # Initial state variance (diffuse)

# Create workspace and run filter + smoother
ws = KalmanWorkspace(Z, H, T, R, Q, a1, P1, n)
kalman_filter!(ws, y)
kalman_smoother!(ws)

# Plot smoothed states (recommended)
p1 = plot(ws, what=:smoothed, title="Smoothed Level",
          xlabel="Year", ylabel="Flow")

# Plot filtered states
p2 = plot(ws, what=:filtered, title="Filtered Level",
          xlabel="Year", ylabel="Flow")

# Plot predicted states
p3 = plot(ws, what=:predicted, title="Predicted Level",
          xlabel="Year", ylabel="Flow")

# For comparison with functional API
result = kalman_filter(KFParms(Z, H, T, R, Q), y, a1, P1)
smooth = SmootherResult(ws)  # Create from workspace

# Compare filtered vs smoothed
p4 = plot((result, smooth), title="Filtered vs Smoothed",
          xlabel="Year", ylabel="Flow")

# Observable predictions with actual data
p5 = plot(result, ObservablePlot, actual=y,
          title="One-Step-Ahead Predictions",
          xlabel="Year", ylabel="Flow")

# Combine into single figure
plot(p1, p2, p4, p5, layout=(2,2), size=(1000, 800))
```

## Customizing Plots

Since recipes use RecipesBase.jl, you can combine them with standard Plots.jl attributes:

```julia
plot(result,
    vars=1,
    level=0.95,
    title="Custom Title",
    xlabel="Time",
    ylabel="Value",
    linewidth=3,
    legend=:bottomright,
    size=(800, 400)
)
```

## Color Scheme

The default color scheme uses consistent colors across plot types:

- **Filtered states**: Steel blue (line) with light blue (bands)
- **Smoothed states**: Dark green (line) with light green (bands)
- **Forecasts**: Dark orange (dashed line) with light orange (bands)
- **Predicted states**: Purple (line) with light purple (bands)

## API Reference

### Core Types

- `KalmanWorkspace`: In-place filter/smoother workspace (recommended)
- `KalmanFilterResult`: Functional API filter output
- `SmootherResult`: Wrapper for smoother output
- `ForecastResult`: Wrapper for forecast output
- `ObservablePlot`: Marker type for observable prediction plots

### In-place Functions

- `kalman_filter!`: Run filter in-place
- `kalman_smoother!`: Run smoother in-place
- `filter_and_smooth!`: Run both in-place

### Accessor Functions

- `smoothed_states`: Get smoothed state means
- `filtered_states`: Get filtered state means
- `predicted_states`: Get predicted state means
- `variances_smoothed_states`: Get smoothed state covariances
- `loglikelihood`: Get log-likelihood

### Helper Functions

- `confidence_bands`: Compute confidence interval bounds
- `select_vars`: Parse variable selection keywords
