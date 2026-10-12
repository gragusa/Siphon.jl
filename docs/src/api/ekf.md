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
```

The extended Kalman filter is selected by passing an `EKFParms` to the generic
functions: `kalman_loglik(p, y, a1, P1)` and `kalman_filter(p, y, a1, P1)`, documented
on the [Core Functions](core.md) page.

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
```

An `EKFWorkspace` runs with the same in-place functions as a `KalmanWorkspace`:
`kalman_filter!(ws, y)`, `kalman_smoother!(ws; crosscov)` and
`kalman_filter_and_smooth!(ws, y; crosscov)`.

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
```

`ssm_loglik(spec, θ, y)` and `optimize_ssm(spec, y)` accept an `EKFSpec` as well as an
`SSMSpec`; see [Optimization](optimization.md).
