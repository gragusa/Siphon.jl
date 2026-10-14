# Core Functions

This page documents the core Kalman filtering and smoothing functions.

## Types

```@docs
KFParms
```

## Kalman Filter

### Log-Likelihood

```@docs
kalman_loglik
```

### Full Filter Output

```@docs
kalman_filter
KalmanFilterResult
```

### Exact Diffuse Initialization

Supplying `P1_inf` to `kalman_filter` selects the exact diffuse filter, which
returns a `DiffuseFilterResult`.

```@docs
DiffuseFilterResult
diffuse_period
diffuse_flags
diffuse_covariances
```

## Kalman Smoother

```@docs
kalman_smoother
kalman_filter_and_smooth
SmootherResult
```

## Disturbance Smoother

```@docs
disturbance_smoother
disturbance_smoother!
DisturbanceSmootherResult
smoothed_obs_disturbances
smoothed_obs_disturbances_cov
smoothed_state_disturbances
smoothed_state_disturbances_cov
auxiliary_residuals
```

## Prediction and Forecasting

```@docs
predict
predicted_observations
forecast
forecast_paths
ForecastResult
forecast_observations
forecast_observations_cov
forecast_states
forecast_states_cov
forecast_periods
forecast_interval
```

## Missing Data Utilities

```@docs
missing_to_nan
nan_to_missing
ismissing_period
isincomplete_period
count_missing_periods
count_incomplete_periods
```

## Plotting

With Plots.jl loaded, filter, smoother and forecast results have plot recipes;
see the Visualization tutorial.

```@docs
ObservablePlot
Siphon.confidence_bands
```

## StaticArrays

```@docs
Siphon.STATIC_THRESHOLD
```
