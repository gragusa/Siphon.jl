# Session Handoff — 2026-10-08

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-016: dispatch-based-variant-selection
- **EKF:** it now runs through the generic names. `kalman_loglik`/`kalman_filter` take an `EKFParms`; `kalman_filter!`, `kalman_smoother!` and `kalman_filter_and_smooth!` take an `EKFWorkspace`; `ssm_loglik`/`optimize_ssm` take an `EKFSpec`. All `ekf_*` functions and `optimize_ekf` are removed.
- **Scalar:** a `KFParms` of numbers dispatches to the scalar kernel and returns a 1-row `KalmanFilterResult`. `KalmanFilterResultScalar` and the `_scalar` functions are removed.
- **Exact diffuse:** selected by supplying `P1_inf`, which was the author's choice this session (see Decisions). `kalman_filter_diffuse!` was folded into `kalman_filter!(::DiffuseKalmanWorkspace, y)`.

## Key decisions / shim choices
- No shims. A "Choosing a filter variant" section was added to the conversion guide.
- Diffuse mechanism: pass `P1_inf` (functional 5-arg form, `DiffuseKalmanWorkspace`, `diffuse = :exact` in the DSL). Initial-state objects and a keyword were rejected.
- `optimize_ekf` was folded into `optimize_ssm(::EKFSpec)`. It isn't an `ekf_*` name, but the same decision applies.
- The scalar smoother kernel was deleted. `kalman_smoother(result)` runs the matrix kernel on 1 × 1 matrices (`_as_matrix`), so `crosscov` works for scalar systems too.
- `kalman_filter_and_smooth!` is a single method on `AbstractFilterWorkspace`.

## State of the codebase
- Files modified:
  - src: Siphon.jl, accessors.jl, types.jl, filter_ad.jl, smoother_ad.jl, ekf_ad.jl, ekf_types.jl, ekf_inplace.jl, inplace.jl, recipes.jl, dsl/{dsl.jl, ekf_codegen.jl, ekf_optim.jl, optimization.jl}, api/core.md
  - test: variant_dispatch_tests.jl (new), runtests.jl, dsl_tests.jl, ekf_tests.jl, failure_tests.jl, missing_input_tests.jl, partial_missing_tests.jl, accessor_tests.jl
  - docs/src: conversion_guide.md, api/core.md, api/ekf.md, tutorials/ekf.md
- Test suite: pass (`Pkg.test()`), 2 broken pins unchanged, Aqua 10/10.
- Ambiguity count: 0 (no change).
- Staged but uncommitted: yes.

## Cluster status
- api-unification: 3 of 5 complete (CHUNK-017, 018 remain).

## Next chunk
CHUNK-017: unify-forecast-output.
- One `forecast` return type, carrying the `forecast_*` accessors and intervals, with the time axis `n+1:n+h`.
- `forecast(::AbstractFilterResult, h)`, merging `_ssm_forecast_from_state`.
- A fix for the `predict` docstring/shape mismatch.
- `forecast_paths` gains `rng`, uses `psd_factor`, and keeps the element type.
- `ForecastResult`'s `time` field should become a plot attribute, as on `SmootherResult`.

## Watch out for
- `kalman_loglik`/`kalman_filter` now carry linear, static, scalar (`_ScalarKFParms`, `y` a vector or 1-row matrix, `a1`/`P1` numbers), 5-arg diffuse and EKF methods. Adding a method with loose argument types risks ambiguities; ambiguities are 0 now.
- A scalar `KalmanFilterResult` has `p.Z` etc. as numbers. Code reading `result.p.Z` as a matrix must use `_as_matrix` (`smoother_ad.jl`). This matters for a `forecast(::AbstractFilterResult, h)` in CHUNK-017.
- `predict`/`forecast` still take only matrix `KFParms`; scalar systems are not routed there.
- `SmootherWorkspace` is still a dead export (CHUNK-030). It was briefly deleted by mistake this session, then restored.
- `api/ekf.md` now describes the generic functions in prose; see the Open Question on `@docs` duplicates (CHUNK-032). The docs build was not run.
- Don't run `Pkg.test()` while editing `src/` or `test/`.
- The `no-plan-comments.py` PostToolUse hook still errors after every Write/Edit (the file is missing); the edits apply.
