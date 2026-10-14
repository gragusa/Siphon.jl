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
| `kalman_filter!(ws, y)` on such a period, for any workspace (Kalman, diffuse or EKF) | threw `PosDefException` | returns `-Inf`; the workspace accessors then throw |
| `kalman_filter(p, y, a1, P1)` and the other functional filters | returned `-Inf` plus arrays holding uninitialised memory | return a result with `loglikelihood(r) == -Inf`; its state and innovation accessors throw, and the arrays past the failure hold `NaN` |
| the scalar filter and the static one-observable filter | threw `DomainError` | return a failed result, as above |
| `kalman_filter_and_smooth!` (formerly `filter_and_smooth!` and `ekf_filter_and_smooth!`) | threw | return `-Inf` and skip the smoother |
| `kalman_smoother`, `predict`, `forecast`, `forecast_paths` on a failed filter | ran on garbage | throw `ArgumentError` |
| EM (`fit!(EM(), …)`, `em_estimate!`) whose filter fails | threw `PosDefException` | stops, returns `converged = false` and warns |

Check `loglikelihood(x) == -Inf` (or `isfinite`) rather than catching an exception.

Accessors refuse to return storage that holds no output:

| Call | Siphon 0.1 | Now |
|:-----|:-----------|:----|
| a filter accessor on a workspace before its first run | returned uninitialised memory | throws `ArgumentError` naming `kalman_filter!(ws, y)` |
| `smoothed_states`, `smoothed_states_cov` on a workspace before `kalman_smoother!` | returned stale or uninitialised memory | throws `ArgumentError`; a new filter run also invalidates the smoother output |
| `crosslag_covs(ws)` after `kalman_smoother!(ws; crosscov = false)` | returned stale memory | throws `ArgumentError` |
| `kalman_smoother!(ws)` before the filter has run | ran on uninitialised memory | throws `ArgumentError` |
| `loglikelihood(model)` on `StateSpaceModel(spec, θ, n)` before `kalman_filter!(model, y)` | returned `-Inf` | throws `ArgumentError` |

`KalmanWorkspace(p, m, r, n)` and the other dimension-only constructors now fill their
arrays with zeros.

Shape errors are `DimensionMismatch`:

| Call | Siphon 0.1 | Now |
|:-----|:-----------|:----|
| `kalman_filter!` on any workspace with `y` of the wrong size | `AssertionError` | `DimensionMismatch` |
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

## Choosing a filter variant

The filter variant follows from the arguments. The specialised names are gone.

### Extended Kalman filter

An `EKFParms`, `EKFWorkspace` or `EKFSpec` selects the extended Kalman filter in the
generic functions:

| Siphon 0.1 | Now |
|:-----------|:----|
| `ekf_loglik(p::EKFParms, y, a1, P1)` | `kalman_loglik(p, y, a1, P1)` |
| `ekf_filter(p::EKFParms, y, a1, P1)` | `kalman_filter(p, y, a1, P1)` (still an `EKFFilterResult`) |
| `ekf_filter!(ws::EKFWorkspace, y)` | `kalman_filter!(ws, y)` |
| `ekf_smoother!(ws)` | `kalman_smoother!(ws)` |
| `ekf_filter_and_smooth!(ws, y)` | `kalman_filter_and_smooth!(ws, y)` |
| `Siphon.ekf_loglik(spec::EKFSpec, θ, y)` | `ssm_loglik(spec, θ, y)` |
| `optimize_ekf(spec::EKFSpec, y; …)` | `optimize_ssm(spec, y; …)`, same keywords and return value |

Error messages on an unrun EKF workspace now name `kalman_filter!(ws, y)` and
`kalman_smoother!(ws)`.

New for the extended Kalman filter: `kalman_smoother(::EKFFilterResult)` and
`kalman_filter_and_smooth(p::EKFParms, …)` smooth in the functional API (before, only
the in-place workspace could). `EKFWorkspace` accepts `FiniteDiffJacobian()`, which
used to throw. Siphon now exports `LinearMeasurement`, so code that defines its own
type of that name, as the EKF tutorial used to, should use the built-in one or rename
its type.

### Models with one state and one series

A `KFParms` whose fields are numbers selects the scalar kernel. It returns an ordinary
`KalmanFilterResult` with one-row arrays, so the accessors, `kalman_smoother` and the
plot recipes work on it:

| Siphon 0.1 | Now |
|:-----------|:----|
| `kalman_loglik_scalar(Z, H, T, R, Q, a1, P1, y)` | `kalman_loglik(KFParms(Z, H, T, R, Q), y, a1, P1)` |
| `kalman_filter_scalar(Z, H, T, R, Q, a1, P1, y)` returned a `KalmanFilterResultScalar` of vectors | `kalman_filter(KFParms(Z, H, T, R, Q), y, a1, P1)` returns a `KalmanFilterResult` (`at` is 1 × n, `Pt` 1 × 1 × n) that also stores the system and the gains |
| `kalman_smoother_scalar(Z, T, at, Pt, vt, Ft)` | `kalman_smoother(kalman_filter(KFParms(Z, H, T, R, Q), y, a1, P1))` |

`y` may be a vector or a 1 × n matrix; `a1` and `P1` are numbers. `KalmanFilterResultScalar`
is gone.

### Exact diffuse initialisation

Supplying the diffuse part `P1_inf` of the initial covariance selects exact diffuse
initialisation, as before: the 5-argument `kalman_loglik(p, y, a1, P1_star, P1_inf)` and
`kalman_filter(p, y, a1, P1_star, P1_inf)`, or a `DiffuseKalmanWorkspace` built with
`P1_inf`. In the DSL it is `diffuse = :exact`. The in-place diffuse filter is now only
`kalman_filter!(ws::DiffuseKalmanWorkspace, y)`; the unexported
`Siphon.kalman_filter_diffuse!` is gone.

Exact diffuse smoothing is new: `kalman_smoother(result::DiffuseFilterResult)`,
`kalman_smoother!(ws::DiffuseKalmanWorkspace)` (and so `kalman_filter_and_smooth!`)
and `smoothed_states(model)` for a model with `diffuse = :exact` used to throw.
`DiffuseFilterResult(ws)` copies the output of a `DiffuseKalmanWorkspace`. A
`DiffuseFilterResult` has two new fields, `diffuse_ended` and `tol`, after
`diffuse_flag`; `forecast` on a result whose diffuse period ends in the last period
now works instead of throwing.

## Prediction and forecasting

Every `forecast` method returns a `ForecastResult`. It holds the observation forecasts
`yhat` (p × h) with covariances `F`, the state forecasts `a` (m × h) with covariances
`P`, and `origin`, the last sample period. Before, `forecast` returned a `NamedTuple`
`(yhat, a, P, F)` on a `KFParms`, spec or `StateSpaceModel` and a
`DynamicFactorModelForecast` on a `DynamicFactorModel`, and `ForecastResult` was a
separate plotting wrapper.

| Siphon 0.1 | Now |
|:-----------|:----|
| `fc = forecast(…)`, then `fc.yhat`, `fc.a`, `fc.P`, `fc.F` | unchanged field names; prefer `forecast_observations(fc)`, `forecast_states(fc)`, `forecast_states_cov(fc)`, `forecast_observations_cov(fc)` |
| `ForecastResult(forecast(…))`, `ForecastResult(nt; time = 101:112)` | `forecast(…)` is already a `ForecastResult`; pass `time` to `plot` |
| `fc.time` | `forecast_periods(fc)`, which is `n+1:n+h` |
| `plot(fc)` with x axis `1:h` | x axis `forecast_periods(fc)`, or `plot(fc; time = …)` |
| `forecast(dfm, h).obs_mean`, `.obs_cov` | `forecast_observations(fc)`, `forecast_observations_cov(fc)` |
| `forecast(dfm, h).state_mean`, `.state_cov` | `forecast_states(fc)`, `forecast_states_cov(fc)` |
| `forecast(dfm, h).factor_mean`, `.factor_cov` | `forecast_states(fc)[1:k, :]`, `forecast_states_cov(fc)[1:k, 1:k, :]` |
| `forecast(dfm, h).h` | `size(forecast_observations(fc), 2)` |
| `Siphon.forecast_interval(fc, α)` (DFM only) | `forecast_interval(fc, 1 - α)`, exported, for every forecast |
| `Siphon.factor_forecast_interval(fc, α)` | `forecast_interval(fc, 1 - α; what = :states)`, rows `1:k` |

New:

- `forecast(result, h)` forecasts from any filter result (`KalmanFilterResult`,
  `DiffuseFilterResult`, `EKFFilterResult`) without running the filter again. An exact
  diffuse result whose diffuse period lasts the whole sample throws an `ArgumentError`.
- `forecast_paths(result, h, n_paths)` does the same for simulated paths.

`predict` returns the fitted values ŷₜ|ₜ₋₁ = Z aₜ (p × n) as a matrix:

| Siphon 0.1 | Now |
|:-----------|:----|
| `predict(…).yhat` | `predict(…)` |
| `predict(…).at`, `.Pt`, `.att`, `.Ptt`, `.vt`, `.Ft`, `.loglik`, `.missing_mask` | `filt = kalman_filter(…)`, then the filter accessors; `predicted_observations(filt)` gives the same matrix as `predict` |

`predicted_observations`, previously defined only for the extended Kalman filter, now
works on `KalmanFilterResult` and `DiffuseFilterResult` too.

`forecast_paths` takes an `rng` keyword, keeps the element type of the system instead of
returning `Float64`, and accepts singular covariances (a state shock of reduced rank, or
`H = 0`), which made it throw before.

## StaticArrays

Every function that builds a linear system from a spec takes one keyword,
`static = :auto | :on | :off`:

- `build_linear_state_space`, `objective_function` and `ssm_loglik`;
- `optimize_ssm`, `optimize_ssm_with_stderr` and `SSMLogDensity`;
- the spec methods of `predict`, `forecast` and `forecast_paths`;
- `fit!(MLE(), model, y)`, `kalman_loglik(model, y)` and `kalman_filter!(model, y)`.

`:auto` uses StaticArrays when the numbers of states, observed series and shocks are all
at most `STATIC_THRESHOLD` (13). The conversion covers all system matrices and the
initial state, or none of them. The choice affects speed. Results agree up to rounding
error, which an ill-conditioned innovation covariance (for example under a large
approximate diffuse `P1`) can amplify.

| Siphon 0.1 | Now |
|:-----------|:----|
| `use_static = true` (DSL functions, `SSMLogDensity`) | `static = :auto` (the default) |
| `use_static = false` | `static = :off` |
| `fit!(MLE(), …; static, static_threshold = k)` | `fit!(MLE(), …; static)`; the threshold is fixed at `STATIC_THRESHOLD` |
| `EKFLogDensity(spec, y; use_static)`, `optimize_ssm(::EKFSpec, y; use_static)`, `build_nonlinear_state_space(…; use_static)` | no keyword: the extended Kalman filter has no StaticArrays specialization |
| `KFParms_static(Z, H, T, R, Q)` | `build_linear_state_space(spec, θ, y).p`, or `KFParms(SMatrix{m, n}(Z), …)` |
| `to_static_if_small(x)` | `SMatrix{m, n}(x)` / `SVector{n}(x)` from StaticArrays |
| `select_backend(n_states, n_obs, backend)` | removed |
| `backend(model)`, `model.backend` | removed; `isfitted(model)` reports whether the model was fitted |

Changed behavior:

- `ssm_loglik`, `objective_function`, `kalman_loglik(model, y)` and
  `kalman_filter!(model, y)` now default to `:auto`. Previously `ssm_loglik` and
  `objective_function` always used plain arrays.
- The `:auto` rule counts shocks too. `kalman_loglik(model, y)` and
  `kalman_filter!(model, y)` previously ignored them.
- A model with one dimension above the threshold no longer gets a mix of static and plain
  arrays.
- `static = :on` with exact diffuse initialization (`diffuse = :exact`) throws an
  `ArgumentError`. Previously it fell back to plain arrays without saying so.

## EM estimation

`fit!(EM(), model, y)` is the EM entry point for a spec, and `profile_em_ssm` is built on
it. The workspace layer underneath is public but not exported: `Siphon.EMWorkspace`,
`Siphon.em_estimate!`, `Siphon.compute_sufficient_stats!` and `Siphon.set_em_masks!`.

| Siphon 0.1 | Now |
|:-----------|:----|
| `EMWorkspace`, `em_estimate!`, `compute_sufficient_stats!` (exported) | `Siphon.EMWorkspace`, `Siphon.em_estimate!`, `Siphon.compute_sufficient_stats!` |
| `Siphon._set_em_masks_from_spec!(em_ws, spec)` | `Siphon.set_em_masks!(em_ws, spec)` |
| `Siphon.DSL._em_general_ssm_full_cov(Z, T, R, H, Q, y, a1, P1; Z_free, T_free, H_free, Q_free, …)` | `Siphon.em_estimate!` on a `KalmanWorkspace` with the masks set on `Siphon.EMWorkspace(kf_ws)` |
| `Siphon.DSL.em_ssm_diagonal(…)` | as above, with diagonal `H_free`/`Q_free` and `em_ws.H_diag_only = true` |
| `Siphon.DSL._em_local_level(spec, y)` | `fit!(EM(), StateSpaceModel(spec, n), y)` |
| `setup_dfm_workspaces`, `extract_loadings`, `extract_factors` | `DynamicFactorModel` with `loadings` and `factors` |
| `EMResult` fields `theta`, `theta_vec`, `smoothed_states`, `smoothed_cov` (from the DSL EMs) | `EMResult` holds `Z`, `H`, `T`, `R`, `Q`, `a1`, `P1`; parameters come from `parameters(model)` or `ProfileEMResult.θ` |
| `profile_em_ssm(…; tol_ll)` | `profile_em_ssm(…; tol)`, the relative change in the log-likelihood as in `fit!(EM(), …)` |
| `ProfileEMResult.em_result::NamedTuple` | `ProfileEMResult.em_result::EMResult` |
| `verbose = true` printed with `println` | logs with `@info` |

Changed behavior:

- `fit!(EM(), …)` throws an `ArgumentError` for parameters that EM cannot update: those
  inside a `MatrixExpr` (such as `λ` of `dns_model`), in `R`, or in an expression.
  Previously they stayed at their initial values without notice.
- EM keeps cells that share a parameter equal, for example `T_structure = :ar1` and
  `H_structure = :scalar` in `dns_model`.
- EM keeps fixed cells of `H` fixed. Previously it overwrote them.
- With full covariance `Q` or `H` (`cov_free`, `Q_structure = :full`), `profile_em_ssm`
  returns the correlation parameters of the estimated matrix. Previously it returned 0,
  so `ssm_loglik(spec, result.θ, y)` did not reproduce `result.loglik`.
- `profile_em_ssm` uses the initial state of the spec. Previously it capped the diagonal
  of `P1` at 1e4.
- `profile_em_ssm` needs a value in `fixed` for every other parameter that EM cannot
  update, for example `fixed = (λ2 = 0.03,)` for Svensson loadings with `λ_param = :λ1`.
- When some free cells of `T` (or `Z`) differ between rows and `Q` (or `H`) is not
  diagonal, the M-step weights the rows by the inverse covariance. Diagonal `T` with a
  full `Q`, the `dns_model` default, now gives the exact M-step.
- The `Z` and `H` updates sum over the observed periods only, which removes a bias when
  whole periods are missing.
- `em_estimate!` throws an `ArgumentError` for periods with some but not all series
  missing, as `fit!(EM(), …)` already did.

## Dynamic factor models

| Siphon 0.1 | Now |
|:-----------|:----|
| `innovation_cov(model)` | `factor_innovation_cov(model)`, or `state_cov(model)` |
| `DynamicFactorModelSpec` (exported) | `Siphon.DynamicFactorModelSpec`; read it as `model.spec` |
| `dynamic_factor(n, k; obs_lags = p)` | `dynamic_factor(n, k; loading_lags = p)`, the keyword of `DynamicFactorModel` |
| `DynamicFactorModel(…; identification = …)` (documented, never implemented) | no replacement; see "Identification" in the dynamic factor tutorial |
| `fit!(EM(), ::DynamicFactorModel, y; verbose = true)` printed a header with `println` | logs with `@info` |

Changed behavior:

- `DynamicFactorModel` has the accessors of the other models: `filtered_states`,
  `smoothed_states`, `predicted_states` and their `_cov` versions, `crosslag_covs`,
  `prediction_errors`, `prediction_errors_cov`, `kalman_gains`, `missing_mask`,
  `observed_mask`, `system_matrices` (and `obs_matrix` … `state_cov`), `initial_state`,
  `initial_state_cov` and `parameters`. `factors_cov`, `filtered_factors` and
  `filtered_factors_cov` are exported.
- `loadings`, `var_coefficients`, `factor_innovation_cov`, `ar_coefficients` and
  `idiosyncratic_variances` return copies. Previously some returned the model's own
  arrays, so mutating the result changed the model.
- `ar_coefficients` returns an empty vector when `error_lags = 0`. Previously it returned
  `[0.0]`.
- `DynamicFactorModel(…; rng)` takes the random number generator for the starting
  loadings. Previously it always drew from the global RNG.

## Module layout

The `Siphon.DSL` submodule is gone. Model specification (templates, `custom_ssm`,
matrix helpers, transformations, `optimize_ssm`, the EKF specification types) now lives
in `Siphon` itself, with the same exported names.

| Siphon 0.1 | Now |
|:-----------|:----|
| `using Siphon.DSL` | `using Siphon` |
| `Siphon.DSL.name` | `Siphon.name` |
| `Siphon.DSL.build_kfparms`, `Siphon.DSL.build_initial_state`, `Siphon.DSL.param_index` | `Siphon.build_kfparms`, `Siphon.build_initial_state`, `Siphon.param_index` (not exported) |
| A matrix element given as an `Expr` in an `SSMMatrixSpec` | a `MatrixExpr` for the whole matrix |
| Any object with `params`, `builder` and `dims` fields passed as a matrix to `custom_ssm` | a `MatrixExpr` |

## Free parameters and matrix helpers

`FreeParam` takes its arguments in the same order as `SSMParameter`, and every
field is optional. A parameter's starting value must lie strictly inside its
bounds. A name used in several cells must have the same settings each time.

| Siphon 0.1 | Now |
|:-----------|:----|
| `FreeParam(name, init, lower, upper)` | `FreeParam(name, lower, upper, init)`, or the keyword form |
| `FreeParam(:v; lower = 0.0)` started at `0.0` (on the bound) | starts at `1.0`; a default inside the bounds |
| `@P(:x, 1.0)`, `Siphon.P` | `FreeParam(:x; init = 1.0)` |
| The same name with different settings in two cells (the first one won) | `ArgumentError` |
| `SSMParameter(:x; lower = 0.0)` (init `0.0`) | init `1.0`, marked as a default |
| `identity_mat(n)` | `I` in `custom_ssm`, or `Matrix(1.0I, n, n)` |
| `zeros_mat(m, n)`, `ones_mat(m, n)` | `zeros(m, n)`, `ones(m, n)` |
| `diag_fixed(v)`, `diag_fixed(x, n)` | `Diagonal(v)`, `Diagonal(fill(x, n))` |
| `companion_mat(n, prefix)` | `companion_free(n, prefix)` |
| `cov_free(n, prefix; init_σ)`, `CovFree.init_σ` | `cov_free(n, prefix; init_sd)`, `CovFree.init_sd` |

## Parameter names

Variances keep the `var_` prefix. Standard deviations and correlation
parameters of a full covariance are named `<prefix>_sd_<i>` and `<prefix>_corr_<i>`.

| Siphon 0.1 | Now |
|:-----------|:----|
| `cov_free(n, :Q)`: `Q_σ_1`, … | `Q_sd_1`, … |
| `dynamic_factor(…; correlated_errors = true)`: `H_σ_1`, … | `H_sd_1`, … |
| `dns_model(…; Q_structure = :full)`: `Q_σ_L`, `Q_σ_S`, `Q_σ_C` | `Q_sd_L`, `Q_sd_S`, `Q_sd_C` |
| `dns_model(…; Q_structure = :diagonal)`: `Q_L`, `Q_S`, `Q_C` | `var_L`, `var_S`, `var_C` |
| `dns_model(…; H_structure = :diagonal)`: `H_1`, … | `var_obs_1`, … |
| `dns_model(…; H_structure = :scalar)`: `H` | `var_obs` |

`SSMSpec` and `EKFSpec` now share the supertype `AbstractSSMSpec`. In
`custom_ekf`, the measurement's output length must match the size of `H`.

## Template keywords

Every parameter keyword of a template takes a number, which fixes the parameter,
or a `FreeParam`, which estimates it. Fields left unset in the `FreeParam` take
the template's defaults, so `FreeParam(init = v)` changes only the starting
value. A keyword that covers several parameters takes one value for all of them,
or a vector with one value per parameter. All `*_init`, `λ_lower` and `λ_upper`
keywords are gone.

| Siphon 0.1 | Now |
|:-----------|:----|
| `local_level(var_obs = :free)` | `local_level()` or `local_level(var_obs = FreeParam())` |
| `local_level(var_obs = (init = 225.0,))` | `local_level(var_obs = FreeParam(init = 225.0))` |
| `local_level(var_obs = (init = 225.0, lower = 1.0))` | `local_level(var_obs = FreeParam(lower = 1.0, init = 225.0))` |
| `local_level(var_obs = 225.0)` (fixed) | unchanged |
| `arma(p, q; ar_init = v, ma_init = w, var_init = s)` | `arma(p, q; ar = [FreeParam(init = v[1]), …], ma = [FreeParam(init = w[1]), …], var = FreeParam(init = s))` |
| *(not possible)* | `arma(2, 0; ar = [FreeParam(), 0.0])` fixes φ2 at 0 |
| `dynamic_factor(…; loadings_init = a, ar_init = b)` | `dynamic_factor(…; loadings = FreeParam(init = a), ar = FreeParam(init = b))` |
| `dynamic_factor(…; var_obs_init = v, var_factor_init = w)` | `dynamic_factor(…; var_obs = FreeParam(init = v), var_factor = FreeParam(init = w))` |
| `dns_model(…; λ_init = a, λ_lower = l, λ_upper = u)` | `dns_model(…; λ = FreeParam(lower = l, upper = u, init = a))` |
| *(not possible)* | `dns_model(…; λ = 0.0609)` fixes λ, and `Z` is then fixed |
| `dns_model(…; T_init = a, H_init = b, Q_init = c)` | `dns_model(…; T = FreeParam(init = a), H = FreeParam(init = b), Q = FreeParam(init = c))` |

Two starting values differ from Siphon 0.1:

- In `dynamic_factor`, a `FreeParam(init = a)` for `loadings` or `ar` starts
  every member at `a`. The old `loadings_init` and `ar_init` were divided by the
  lag (plus one, for loadings). Leave `init` unset to keep the per-lag defaults.
- In `dns_model`, `T = FreeParam(init = a)` with `T_structure = :full` starts all
  nine entries at `a`; the old `T_init` set only the diagonal. Pass a vector of
  nine values to set the diagonal and off-diagonal entries separately.

`dns_model` variances left unset are now rescaled to the data by maximum
likelihood, as in the other templates.

## Priors

Priors are now univariate building blocks combined by parameter name with
`ParameterPriors`. Each one applies to the parameter as estimated, so a prior on
`var_obs` is a prior on the variance. With Distributions.jl loaded, any
univariate distribution can take the place of a built-in prior.
`SSMLogDensity` checks that each name in a `ParameterPriors` is a parameter
of the spec, and parameters without a prior get a flat one.

| Siphon 0.1 | Now |
|:-----------|:----|
| `NormalPrior((a = μa, b = μb), (a = σa, b = σb))` | `ParameterPriors(a = NormalPrior(μa, σa), b = NormalPrior(μb, σb))` |
| `NormalPriorVec(μ, σ)` | `ParameterPriors` with one `NormalPrior` per name |
| `InverseGammaPrior(α, β, (:σ_obs,))`: density of σ² at a standard deviation σ | `ParameterPriors(var_obs = InverseGammaPrior(α, β))`: density of the variance itself |
| `CompositePrior(p1, p2)` | unchanged; mixed families fit in one `ParameterPriors` |
| *(not possible)* | `using Distributions; ParameterPriors(var_obs = InverseGamma(2, 1), ρ = Beta(2, 2))` |
| `SSMLogDensity(spec, y; prior = nothing)` | `SSMLogDensity(spec, y)` or `prior = FlatPrior()` |
| `EKFLogDensity(spec, y)` | `SSMLogDensity(spec, y; prior)`, which now accepts an `EKFSpec` |

All priors are subtypes of `AbstractPrior`. `SSMLogDensity` also accepts a plain
function `θ -> log p(θ)` as `prior`.

## Element types and constructors

The element type of a workspace or model is a type parameter everywhere. Without
one, `Float64` is used, or, for constructors that take system matrices, the
floating-point type that holds all of them.

| Siphon 0.1 | Now |
|:-----------|:----|
| `StateSpaceModel(spec, n; T = Float32)` | `StateSpaceModel{Float32}(spec, n)` |
| `StateSpaceModel(spec, θ, n; T = Float32)` | `StateSpaceModel{Float32}(spec, θ, n)` |
| `DynamicFactorModel(N, k, n; T = Float32, …)` | `DynamicFactorModel{Float32}(N, k, n; …)` |
| `Siphon.EMWorkspace(p, m, r, n, Float32)` | `Siphon.EMWorkspace{Float32}(p, m, r, n)` |
| `SmootherWorkspace(m, n, Float32)` | `SmootherWorkspace{Float32}(m, n)` |
| `KalmanWorkspace(Z, …)` with an integer `Z`: a `KalmanWorkspace{Int}` | a `KalmanWorkspace{Float64}`; the type holds every argument, not only `Z` |
| `EKFWorkspace(meas, mode, H, …)`: always `Float64` | the floating-point type of `H`, `Tmat`, `R`, `Q`, `a1`, `P1`; `EKFWorkspace{T}(…)` forces one |

`KalmanWorkspace{T}(Z, H, T, R, Q, a1, P1, n)`, `KalmanWorkspace{T}(kf::KFParms, a1, P1, n)`
and `DiffuseKalmanWorkspace{T}(…)` set the element type explicitly.

`simulation_smoother` takes the number of draws as a positional argument, so its
return type no longer depends on a keyword value:

| Siphon 0.1 | Now |
|:-----------|:----|
| `simulation_smoother(ws, y)`: `m × n` matrix | unchanged |
| `simulation_smoother(ws, y; n_draws = k)`: `m × n` when `k == 1`, else `m × n × k` | `simulation_smoother(ws, y, k)`: always `m × n × k` |

`Siphon.EMWorkspace`, `SimulationSmootherWorkspace` and `simulation_smoother` throw
an `ArgumentError` for a `DiffuseKalmanWorkspace`: EM needs the cross-lag covariances
and the simulation smoother draws the initial state from `N(a1, P1)`, neither of which
is available under exact diffuse initialization. Use a `KalmanWorkspace` with a large
`P1`.
