# Extended Kalman Filter (EKF)

API reference for Siphon.jl's EKF extension. See the
[EKF tutorial](../tutorials/ekf.md) for usage examples and a worked
shadow-rate-DNS fit.

## Measurement Protocol

```@docs
Siphon.AbstractEKFMeasurement
Siphon.measurement
Siphon.measurement!
Siphon.measurement_jacobian
Siphon.measurement_jacobian!
```

## Jacobian Modes

```@docs
Siphon.AbstractEKFJacobianMode
Siphon.ADJacobian
Siphon.AnalyticJacobian
Siphon.FiniteDiffJacobian
```

## Functional API

```@docs
Siphon.EKFParms
Siphon.EKFFilterResult
Siphon.ekf_loglik
Siphon.ekf_filter
```

### Accessors on `EKFFilterResult`

```@docs
Siphon.predicted_observations
Siphon.measurement_jacobians
```

The standard accessors (`predicted_states`, `filtered_states`, `prediction_errors`,
etc.) also work on `EKFFilterResult`.

## In-place API

```@docs
Siphon.EKFWorkspace
Siphon.ekf_filter!
Siphon.ekf_smoother!
Siphon.ekf_filter_and_smooth!
```

## DSL

```@docs
Siphon.DSL.EKFSpec
Siphon.DSL.MeasurementExpr
Siphon.DSL.custom_ekf
Siphon.DSL.build_measurement
Siphon.DSL.build_ekfparms
Siphon.DSL.build_nonlinear_state_space
```

## Estimation

```@docs
Siphon.DSL.EKFLogDensity
Siphon.DSL.optimize_ekf
```
