# Session Handoff — 2026-10-08

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-015: unify-filter-smoother-interfaces
- Every smoother path now returns `SmootherResult` (fields `αs`, `Vs`, `Pcross` or `nothing`; read it with `smoothed_states`/`smoothed_states_cov`/`crosslag_covs`).
- `kalman_smoother(result)` reads the system from the filter result.
- `kalman_filter_and_smooth` returns `(KalmanFilterResult, SmootherResult)`.
- `filter_and_smooth!` is now `kalman_filter_and_smooth!`.
- `crosscov` (default `false`) is the single keyword.
- New: `KalmanFilterResult(ws)`, copying `SmootherResult(ws)`, and `SmootherResult(model)`.
- `update_params!` takes `T` only.

## Key decisions / shim choices
- No shims. A "Smoothing" section was added to the conversion guide.
- The array method `kalman_smoother(Z, T, at, Pt, vt, Ft)` is now the internal `_kalman_smoother`, which also returns a `SmootherResult`. Internal users: the model smoother, the static EM, and the DSL EMs (imported in `dsl.jl`).
- The `SmootherResult` fields `p` and `time` were dropped; `time` is a plot attribute.
- The `crosscov` default changed from `true` to `false` for the in-place smoothers. Every internal EM call passes `crosscov = true` explicitly.
- `kalman_smoother(::DiffuseFilterResult)` throws an `ArgumentError` until the diffuse smoother exists.

## State of the codebase
- Files modified:
  - src: smoother_ad.jl, recipes.jl, inplace.jl, ekf_inplace.jl, simsmooth.jl, Siphon.jl, dsl/{dsl.jl, em.jl}, api/core.md
  - test: smoother_interface_tests.jl (new), runtests.jl, dsl_tests.jl, em_tests.jl, recipe_tests.jl, partial_missing_tests.jl, failure_tests.jl, missing_input_tests.jl, simsmooth_tests.jl, accessor_tests.jl, api_coverage_tests.jl
  - docs/src: conversion_guide.md, api/core.md, tutorials/{visualization, estimation_methods, custom_models, dynamic_factor, ekf}.md
  - examples: dns_estimation.jl, dns_estimation_dsl.jl
- Test suite: pass (`Pkg.test()`), 2 broken pins unchanged, Aqua 10/10.
- Ambiguity count: 0 (no change).
- Staged but uncommitted: yes.

## Cluster status
- api-unification: 2 of 5 complete (CHUNK-016, 017, 018 remain).

## Next chunk
CHUNK-016: dispatch-based-variant-selection.
- The EKF uses the generic names (`kalman_filter(::EKFParms)`, `kalman_filter!(::EKFWorkspace)`, `kalman_smoother!(::EKFWorkspace)`, `kalman_filter_and_smooth!(::EKFWorkspace)`), and the `ekf_*` names are removed.
- One diffuse-selection mechanism.
- A scalar `KFParms` dispatches to the scalar kernel; `kalman_smoother_scalar` and the other `_scalar` names are removed.
- The diffuse filter is exported or folded in.

## Watch out for
- `kalman_smoother!`/`kalman_filter_and_smooth!` default to `crosscov = false`. Any new EM-like caller that reads `Pcross` or calls `compute_sufficient_stats!` must pass `crosscov = true`; `compute_sufficient_stats!` throws otherwise.
- `kalman_smoother_scalar` returns a 1 × n `SmootherResult`. When CHUNK-016 folds the scalar path into `KFParms`, `kalman_smoother(result)` should cover the scalar result.
- `KalmanFilterResultScalar` doesn't store Z/T, so `kalman_smoother(::KalmanFilterResultScalar)` doesn't exist yet.
- `SmootherResult(ws)` is generic over `AbstractFilterWorkspace` and relies on the `αs`/`Vs`/`Pcross`/`crosscov` fields of `_filter_store(ws)`.
- The docs build (`docs/make.jl`) was not run.
- Don't run `Pkg.test()` in the background while editing `src/`. Don't run `Pkg.develop` in `test/`.
- The `no-plan-comments.py` PostToolUse hook still errors after every Write/Edit (the file is missing); the edits apply.
