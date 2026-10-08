# Conversion Guide

This page lists the changes that break code written for Siphon 0.1 and shows the
replacement for each. There are no deprecation shims: the old names are gone.

## Accessors

Filter results (`KalmanFilterResult`, `DiffuseFilterResult`, `EKFFilterResult`,
`KalmanFilterResultScalar`) are now subtypes of `AbstractFilterResult`. The in-place
workspaces (`KalmanWorkspace`, `DiffuseKalmanWorkspace`, `EKFWorkspace`) are subtypes of
`AbstractFilterWorkspace`. One set of accessor names covers results, workspaces and fitted
models.

Covariance accessors end in `_cov`:

| Siphon 0.1 | Now |
|:-----------|:----|
| `variances_predicted_states(x)` | `predicted_states_cov(x)` |
| `variances_filtered_states(x)` | `filtered_states_cov(x)` |
| `variances_prediction_errors(x)` | `prediction_errors_cov(x)` |
| `variances_smoothed_states(x)` | `smoothed_states_cov(x)` |
| `variances_forecast_states(fc)` | `forecast_states_cov(fc)` |
| `variances_forecast_observations(fc)` | `forecast_observations_cov(fc)` |
| `Siphon.innovations(ws)` | `prediction_errors(ws)` |
| `Siphon.innovation_covs(ws)` | `prediction_errors_cov(ws)` |

`parameters` and `system_matrices` now mean the same thing on every type:

| Siphon 0.1 | Now |
|:-----------|:----|
| `parameters(result)` (returned the `KFParms`) | `system_matrices(result)` |
| `system_matrices(model)` returned a `NamedTuple` `(Z, H, T, R, Q)` | returns a `KFParms` with the same fields |

`parameters(model)` still returns the fitted θ as a NamedTuple. Filter results and
workspaces have no θ, so `parameters` is not defined on them.

Code that reads fields, such as `system_matrices(model).Z`, keeps working. Code that
iterated over the NamedTuple or called `keys` on it must read the fields by name.

New accessors:

- `obs_matrix`, `obs_cov`, `transition_matrix`, `selection_matrix`, `state_cov` and
  `system_matrices` work on workspaces as well as on results and models.
- `initial_state(x)` and `initial_state_cov(x)` return the a₁ and P₁ used by the filter.
- `crosslag_covs(ws)` (now exported) returns the smoothed cross-lag covariances.
- `kalman_gains` works on a fitted `StateSpaceModel`, as the other filter accessors
  already did.

`system_matrices` of a model or result can be passed straight to the functional filter:

```julia
p = system_matrices(model)
filt = kalman_filter(p, y, initial_state(model), initial_state_cov(model))
```

On the extended Kalman filter, `system_matrices` returns the `EKFParms`, and
`obs_matrix` throws, because the measurement is nonlinear. Use
`measurement_jacobians` instead.

## Workspace setters

The unexported single-matrix setters were renamed. Under their old names they collided
with the EM M-step functions:

| Siphon 0.1 | Now |
|:-----------|:----|
| `Siphon.update_Z!(ws, Z)` | `Siphon.set_Z!(ws, Z)` |
| `Siphon.update_H!(ws, H)` | `Siphon.set_H!(ws, H)` |
| `Siphon.update_T!(ws, T)` | `Siphon.set_T!(ws, T)` |
| `Siphon.update_R!(ws, R)` | `Siphon.set_R!(ws, R)` |
| `Siphon.update_Q!(ws, Q)` | `Siphon.set_Q!(ws, Q)` |

`set_params!` and `update_params!` are unchanged.
