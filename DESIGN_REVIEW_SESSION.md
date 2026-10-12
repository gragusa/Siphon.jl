# Session Handoff — 2026-10-12

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-017: unify-forecast-output
- Every `forecast` method now returns one `ForecastResult` (fields `yhat`, `F`, `a`, `P`, `origin`). This covers `KFParms`, spec, `StateSpaceModel`, DFM and the new `forecast(::AbstractFilterResult, h)`.
- The `forecast_*` accessors and the new `forecast_periods` and `forecast_interval` work on it, and the plot's time axis is `n+1:n+h`.
- `predict` returns the fitted-value matrix.
- `forecast_paths` gained `rng`, `psd_factor` and element-type promotion.

## Key decisions / shim choices
- No shims; there is a new conversion-guide section, "Prediction and forecasting".
- `predict` → matrix ŷ. This was the implementer's call and is recorded in Decisions as unconfirmed by the author. `predicted_observations` gained linear/diffuse methods.
- The EKF forecasts through the measurement function and its Jacobian (`_measurement_mean`/`_measurement_jacobian` hooks in `predict.jl`).
- `DynamicFactorModelForecast`, the DFM-only `forecast_interval` and `factor_forecast_interval` are removed; factors are the first k states.

## State of the codebase
- Files modified:
  - src: Siphon.jl, predict.jl, inplace.jl, recipes.jl, api/core.md
  - test: api_coverage, recipe, dfm, ekf, failure, dsl, missing_input, partial_missing
  - docs/src: conversion_guide.md, api/core.md, api/ekf.md, tutorials/visualization.md, tutorials/dynamic_factor.md
- Test suite: pass (`Pkg.test()`), 2 broken pins unchanged, Aqua 10/10.
- Ambiguity count: 0 (no change).
- Staged but uncommitted: yes.

## Cluster status
- api-unification: 4 of 5 complete (CHUNK-018 remains).

## Next chunk
CHUNK-018: unify-static-backend-controls. Replace `KFParms_static`/`to_static_if_small`/`use_static`/`static=`/`select_backend` with one keyword, one vocabulary and one gating rule.

## Watch out for
- The spec-level `predict`/`forecast`/`forecast_paths` still take `use_static::Bool`; CHUNK-018 must migrate them.
- `forecast` on a `DiffuseFilterResult` with `d == n` throws, including the edge case where the diffuse period ends exactly in period n (Open Question).
- `ForecastResult` is now defined in `predict.jl`, which is included before `recipes.jl`; `forecast_interval` calls `confidence_bands` from recipes.jl at runtime.
- Don't run `Pkg.test()` while editing `src/` or `test/`. Don't `Pkg.develop` into `test/`: it creates a stray `test/Project.toml` and rewrites the tracked `test/Manifest.toml`.
- The `no-plan-comments.py` hook still errors after every Write/Edit; the edits apply.
