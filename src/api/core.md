# Core Functions

This page documents the core Kalman filtering and smoothing functions.

## Types

```@docs
KFParms
KFParms_static
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

## Kalman Smoother

```@docs
kalman_smoother
kalman_filter_and_smooth
SmootherResult
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

## StaticArrays Utilities

```@docs
STATIC_THRESHOLD
to_static_if_small
```
