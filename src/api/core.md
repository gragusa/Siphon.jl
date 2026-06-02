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
kalman_loglik_scalar
```

### Full Filter Output

```@docs
kalman_filter
kalman_filter_scalar
```

## Kalman Smoother

```@docs
kalman_smoother
kalman_smoother_scalar
kalman_filter_and_smooth
```

## Prediction and Forecasting

```@docs
predict
forecast
forecast_paths
```

## Missing Data Utilities

```@docs
missing_to_nan
nan_to_missing
count_missing
ismissing_obs
```

## StaticArrays Utilities

```@docs
STATIC_THRESHOLD
to_static_if_small
```
