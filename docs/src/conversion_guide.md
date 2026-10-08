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

`set_params!` is unchanged. `update_params!` takes the transition matrix as `T`, on
`KalmanWorkspace` and `EKFWorkspace` alike; the `Tmat` keyword is gone.

## Failures and errors

A filter now stops at the first period whose innovation covariance is not positive
definite, and every path reports this the same way:

| Situation | Siphon 0.1 | Now |
|:----------|:-----------|:----|
| `kalman_filter!(ws, y)`, `ekf_filter!(ws, y)` or `kalman_filter_diffuse!` on such a period | threw `PosDefException` | returns `-Inf`; the workspace accessors then throw |
| `kalman_filter(p, y, a1, P1)` and the other functional filters | returned `-Inf` plus arrays holding uninitialised memory | return a result with `loglikelihood(r) == -Inf`; its state and innovation accessors throw, and the arrays past the failure hold `NaN` |
| `kalman_filter_scalar` and the static one-observable filter | threw `DomainError` | return a failed result, as above |
| `kalman_filter_and_smooth!` (formerly `filter_and_smooth!`), `ekf_filter_and_smooth!` | threw | return `-Inf` and skip the smoother |
| `kalman_smoother`, `predict`, `forecast`, `forecast_paths` on a failed filter | ran on garbage | throw `ArgumentError` |
| EM (`fit!(EM(), …)`, `em_estimate!`) whose filter fails | threw `PosDefException` | stops, returns `converged = false` and warns |

Check `loglikelihood(x) == -Inf` (or `isfinite`) rather than catching an exception.

Accessors refuse to return storage that holds no output:

| Call | Siphon 0.1 | Now |
|:-----|:-----------|:----|
| a filter accessor on a workspace before its first run | returned uninitialised memory | throws `ArgumentError` naming `kalman_filter!(ws, y)` (`ekf_filter!(ws, y)` for the EKF) |
| `smoothed_states`, `smoothed_states_cov` on a workspace before `kalman_smoother!` | returned stale or uninitialised memory | throws `ArgumentError`; a new filter run also invalidates the smoother output |
| `crosslag_covs(ws)` after `kalman_smoother!(ws; crosscov = false)` | returned stale memory | throws `ArgumentError` |
| `kalman_smoother!(ws)` before the filter has run | ran on uninitialised memory | throws `ArgumentError` |
| `loglikelihood(model)` on `StateSpaceModel(spec, θ, n)` before `kalman_filter!(model, y)` | returned `-Inf` | throws `ArgumentError` |

`KalmanWorkspace(p, m, r, n)` and the other dimension-only constructors now fill their
arrays with zeros.

Shape errors are `DimensionMismatch`:

| Call | Siphon 0.1 | Now |
|:-----|:-----------|:----|
| `kalman_filter!`, `ekf_filter!`, `kalman_filter_diffuse!` with `y` of the wrong size | `AssertionError` | `DimensionMismatch` |
| `kalman_filter!(model, y)`, `kalman_loglik(model, y)`, `fit!(…, model, y)` with `y` of the wrong size | `AssertionError` | `DimensionMismatch` |
| `EKFWorkspace(...)` with inconsistent matrices | `AssertionError` | `DimensionMismatch` |
| `KFParms(Z, H, T, R, Q)` with inconsistent matrices | accepted | `DimensionMismatch` |
| `set_params!`, `set_initial!`, `update_params!`, `set_Z!`…`set_Q!`, `set_initial_diffuse!` with wrongly sized matrices | copied element by element (or threw `BoundsError`) | `DimensionMismatch` |
| `diag_free(names; init = v)` with `length(v) != length(names)` | `AssertionError` | `DimensionMismatch` |

A template given `diffuse = :foo` (anything other than `true`, `false` or `:exact`) throws
an `ArgumentError` instead of a `TypeError`.

## Convergence warnings

Estimation that ends without converging now warns. Pass `warn = false` to silence it:

| Function | Warns when |
|:---------|:-----------|
| `fit!(EM(), model, y)`, `em_estimate!`, `fit!(EM(), ::DynamicFactorModel, y)` | `maxiter` is reached, or the filter fails at the current parameters |
| `optimize_ssm`, `optimize_ekf`, `fit!(MLE(), model, y)` | the optimiser's return code is not `Success` |
| `optimize_ssm_with_stderr` | as `optimize_ssm`; it already warned, and now accepts `warn` |

`isconverged(model)` and the `converged` fields are unchanged.

## Missing data

Every filter, smoother, forecasting function and estimator now accepts observations
with `missing` entries (for example a `Matrix{Union{Missing, Float64}}`) and treats them
like `NaN`. Converting with `missing_to_nan` first is no longer necessary.

The period counts are split by meaning:

| Siphon 0.1 | Now |
|:-----------|:----|
| `count_missing(y)` (periods with at least one missing value) | `count_incomplete_periods(y)` |
| `ismissing_obs(y_t::AbstractVector)` (any value missing) | `isincomplete_period(y_t)` |
| `ismissing_obs(x::Real)` | `isnan(x)` |
| (none) | `count_missing_periods(y)` and `ismissing_period(y_t)`: periods with no value, the ones `missing_mask` marks |

`missing_to_nan` and `nan_to_missing` keep the element type instead of forcing `Float64`:
`Union{Missing, Float32}` becomes `Float32`, and `nan_to_missing` returns
`Union{Missing, eltype(y)}`. Integer input still becomes `Float64`, which can hold `NaN`.
`missing_to_nan` of an array that is already floating point returns it unchanged.

### Innovations in series positions

`prediction_errors`, `prediction_errors_cov` and `kalman_gains` now index by series in
every period:

| Quantity in a period where only some rows are observed | Siphon 0.1 | Now |
|:----------|:-----------|:----|
| `prediction_errors(x)[:, t]` | the innovations of the observed rows packed into the leading rows, then `NaN` | row `i` is the innovation of series `i`; `NaN` where series `i` is unobserved |
| `prediction_errors_cov(x)[:, :, t]` | packed into the leading block | rows and columns by series; `NaN` for unobserved series |
| `kalman_gains(x)[:, :, t]` | packed into the leading columns | column `i` for series `i`; zero for unobserved series |

In a period with no observed row, `prediction_errors_cov` is now `NaN`. Before, the
functional filter reported `Z P Z' + H` there and the in-place filter reported `NaN`.

These three accessors now return new arrays on workspaces as well as on results. The
packed layout remains in the fields `vt`, `Ft` and `Kt`, together with `observed_rows`.

## Smoothing

The smoothers return one type, `SmootherResult`, read through the accessors
`smoothed_states`, `smoothed_states_cov` and `crosslag_covs`. The cross-lag covariances
are requested with `crosscov`, which defaults to `false` everywhere.

| Siphon 0.1 | Now |
|:-----------|:----|
| `kalman_smoother(Z, T, at, Pt, vt, Ft; compute_crosscov)` returned `(alpha, V, P_crosslag)` | `kalman_smoother(kalman_filter(p, y, a1, P1); crosscov)` returns a `SmootherResult` |
| `kalman_smoother(result, Z, T)` | `kalman_smoother(result)`: the system matrices come from `result` |
| `kalman_smoother(p, y, a1, P1)` returned `(α, V)` | returns a `SmootherResult` |
| `kalman_smoother_scalar(...)` returned a tuple of vectors | returns a `SmootherResult` with one state (1 × n) |
| `kalman_filter_and_smooth(p, y, a1, P1)` returned `(loglik, a_filtered, P_filtered, alpha_smooth, V_smooth)` | returns the tuple `(filter, smoother)` of a `KalmanFilterResult` and a `SmootherResult` |
| `compute_crosscov = true` (functional) | `crosscov = true` |
| `kalman_smoother!(ws)`, `ekf_smoother!(ws)` computed the cross-lag covariances by default | pass `crosscov = true` |
| `filter_and_smooth!(ws, y)` | `kalman_filter_and_smooth!(ws, y)`; pass `crosscov = true` for `crosslag_covs(ws)` |
| `SmootherResult(nt)` wrapping the smoother's NamedTuple | not needed: `kalman_smoother` returns a `SmootherResult` |
| `SmootherResult(p, y, a1, P1)` | `kalman_smoother(p, y, a1, P1)` |
| `SmootherResult(ws)` aliased the workspace storage | copies it, so later runs on `ws` don't change it |
| `SmootherResult(…; time = t)`, then `plot(sr)` | `plot(sr; time = t)` |
| `sr.alpha`, `sr.V`, `result.alpha_smooth`, `nt.P_crosslag` | `smoothed_states(sr)`, `smoothed_states_cov(sr)`, `crosslag_covs(sr)` |

```julia
# Siphon 0.1
filt = kalman_filter(p, y, a1, P1)
sm = kalman_smoother(p.Z, p.T, filt.at, filt.Pt, filt.vt, filt.Ft; compute_crosscov = true)
sm.alpha, sm.P_crosslag

# Now
filt, sm = kalman_filter_and_smooth(p, y, a1, P1; crosscov = true)
smoothed_states(sm), crosslag_covs(sm)
plot((filt, sm))
```

New conversions:

- `KalmanFilterResult(ws)` copies the filter output of a `KalmanWorkspace` into a result,
  which `kalman_smoother` and the plot recipes accept.
- `SmootherResult(model)` copies the smoothed states of a fitted `StateSpaceModel`.
- `kalman_smoother` on an exact diffuse result (`DiffuseFilterResult`) throws an
  `ArgumentError`. Before, it was a `MethodError`.
