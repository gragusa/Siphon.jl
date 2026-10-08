# Session Handoff — 2026-10-08

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-013: apply-failure-model
The failure policy (i)–(v) is applied across the package:
- Filters stop at the first non-positive-definite F, return −Inf and never throw. They record the failing period: `failed_at` on results, `status` on workspaces, `filter_status` on `StateSpaceModel`.
- Accessors on unrun or failed outputs throw an `ArgumentError` that names the call to make or the failed period. Functional storage is NaN-filled, and dimension-only constructors zero-fill.
- Every `@assert` became a `DimensionMismatch`. `KFParms` and all setters validate shapes.
- EM and MLE warn on non-convergence (`warn = false` silences them). EM stops cleanly when its filter fails.

## Key decisions / shim choices
- No shims. The conversion guide gained "Failures and errors" and "Convergence warnings".
- One status encoding everywhere: `_NOT_RUN` = −1, 0 = ok, t > 0 = failed in period t.
- `loglikelihood` returns −Inf on a failed output; it throws only on a workspace or known-parameter model that has never run.
- The silencing keyword is `warn`, not `verbose`, because `verbose` already means "print progress".
- The external model keeps `fitted = true`; instead, `loglikelihood(model)` throws until `kalman_filter!(model, y)` runs.
- A failed E-step ends EM with `converged = false` and a warning. It does not throw.

## State of the codebase
- Files modified:
  - src: accessors.jl, types.jl, ekf_types.jl, filter_ad.jl, ekf_ad.jl, inplace.jl, ekf_inplace.jl, smoother_ad.jl, predict.jl, simsmooth.jl, recipes.jl, dsl/{optimization.jl, ekf_optim.jl, templates.jl, matrix_helpers.jl}
  - test: failure_tests.jl (new), runtests.jl, diffuse_tests.jl, dsl_tests.jl, partial_missing_tests.jl
  - docs/src/conversion_guide.md
- Test suite: pass (`Pkg.test()`). 2 broken pins unchanged, Aqua 10/10.
- Ambiguity count: 0 (no change).
- Staged but uncommitted: yes.

## Cluster status
- type-hierarchy: 2 of 2 complete (CHUNK-012, CHUNK-013). Cluster done.
- api-unification: 0 of 5 complete (CHUNK-014, 015, 016, 017, 018).

## Next chunk
CHUNK-014: apply-missing-data-model.
- Accept `missing` at entry points.
- Split the any/all counts (`count_missing`/`ismissing_obs` vs `missing_mask`).
- Expand public `prediction_errors` to series positions.
- Make `missing_to_nan`/`nan_to_missing` preserve the element type.

## Watch out for
- Any new code reading workspace storage should go through the accessors or `_output(x, :name)`, not `ws.att` etc., or it bypasses the run-status check. Internal kernels (EM M-steps, the smoother loops) still read fields directly after their callers have checked.
- A new filter run resets `smoothed`/`crosscov`. Anything that fills `αs`/`Vs` outside `kalman_smoother!` must set `ws.smoothed = true` itself.
- CHUNK-014 will change what `prediction_errors` returns for partial periods. The accessor is in `accessors.jl` and now calls `_output(x, :prediction_errors)`.
- The test log now has 8 "EM did not converge" warnings from tests that cap `maxiter` on purpose (see Open Questions).
- The docs build (`docs/make.jl`) was not run.
- Don't run `Pkg.develop` in `test/`; use `Pkg.test()`.
- The `no-plan-comments.py` PostToolUse hook still errors after every Write (the file is missing); the edits apply.
