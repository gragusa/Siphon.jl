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
LinearMeasurement
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
functions: `kalman_loglik(p, y, a1, P1)`, `kalman_filter(p, y, a1, P1)`,
`kalman_smoother` and `kalman_filter_and_smooth`, documented on the
[Core Functions](core.md) page.

### Accessors on `EKFFilterResult`

```@docs
Siphon.measurement_jacobians
```

`predicted_observations(r)` returns the predicted measurements `h(aₜ, t)`; see the
[Core Functions](core.md) page.

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
Siphon.EKFSpec
Siphon.MeasurementExpr
Siphon.custom_ekf
Siphon.build_measurement
Siphon.build_ekfparms
Siphon.build_nonlinear_state_space
```

## Estimation

`ssm_loglik(spec, θ, y)`, `optimize_ssm(spec, y)` and `SSMLogDensity(spec, y; prior)`
accept an `EKFSpec` as well as an `SSMSpec`; see [Optimization](optimization.md).

`StateSpaceModel(spec::EKFSpec, n)` supports `fit!(MLE(), model, y)`,
`kalman_filter!(model, y)`, the filter and smoother accessors, `forecast(model, h)` and
the StatsAPI accessors (`coef`, `vcov`, `stderror`, `aic`, …). `fit!(EM(), …)`,
`disturbance_smoother(model)` and `obs_matrix(model)` throw; `system_matrices(model)`
returns the `EKFParms`.
