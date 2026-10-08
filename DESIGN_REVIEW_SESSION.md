# Session Handoff — 2026-10-08

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-014: apply-missing-data-model
- Every public entry point that takes observations accepts `missing` entries, converted to NaN by `_nan_obs`.
- `count_missing`/`ismissing_obs` were split into "missing period" (no value) and "incomplete period" (some value absent) functions.
- `missing_to_nan`/`nan_to_missing` keep element types.
- `prediction_errors`, `prediction_errors_cov` and `kalman_gains` return copies in series positions; the stored layout stays packed.
- Every filter output now has `observed_mask`.

## Key decisions / shim choices
- No shims. A "Missing data" section was added to the conversion guide.
- New names: `ismissing_period`, `isincomplete_period`, `count_missing_periods`, `count_incomplete_periods`. They accept NaN or `missing`.
- The three innovation accessors now always return new arrays, including on workspaces (documented on `AbstractFilterWorkspace`).
- `prediction_errors_cov` is NaN for unobserved series, including fully missing periods. The functional filter's old `Z P Z' + H` there is no longer exposed.
- `_nan_obs` is dispatch-based, so plain real arrays pay nothing.

## State of the codebase
- Files modified:
  - src: missing.jl (new), Siphon.jl, accessors.jl, types.jl, ekf_types.jl, filter_ad.jl, ekf_ad.jl, smoother_ad.jl, predict.jl, inplace.jl, ekf_inplace.jl, simsmooth.jl, recipes.jl, dsl/{dsl.jl, bayesian.jl, codegen.jl, ekf_codegen.jl, ekf_optim.jl, optimization.jl}, api/core.md
  - test: missing_input_tests.jl (new), runtests.jl, api_coverage_tests.jl, partial_missing_tests.jl
  - docs/src: conversion_guide.md, api/core.md
- Test suite: pass (`Pkg.test()`). 2 broken pins unchanged, Aqua 10/10.
- Ambiguity count: 0 (no change).
- Staged but uncommitted: yes.

## Cluster status
- type-hierarchy: 2 of 2 complete.
- api-unification: 1 of 5 complete (CHUNK-015, 016, 017, 018 remain).

## Next chunk
CHUNK-015: unify-filter-smoother-interfaces.
- One smoother result type and one crosscov keyword.
- `kalman_smoother(result)` takes the system from the result.
- `kalman_filter_and_smooth!` naming.
- `KalmanFilterResult(ws)`, and a copying `SmootherResult(ws)`.
- The `update_params!` `T`/`Tmat` keyword.

## Watch out for
- A new public function that takes `y` must start with `y = _nan_obs(y)`, or `missing` input will reach the NaN-based kernels.
- Internal code needing the packed innovations must read the fields (`r.vt`, `ws.Ft`, …), not the accessors, which now unpack.
- `SmootherResult(ws)` still aliases `ws.αs`/`ws.Vs` through the accessors; CHUNK-015 makes it copy.
- Running `Pkg.test()` in the background while editing `src/` races the compile. Wait for the run to finish before editing, or re-run.
- The docs build (`docs/make.jl`) was not run.
- Don't run `Pkg.develop` in `test/`; use `Pkg.test()`.
- The `no-plan-comments.py` PostToolUse hook still errors after every Write (the file is missing); the edits apply.
