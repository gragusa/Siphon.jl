# Session Handoff — 2026-10-14

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-028: add-disturbance-smoother
A new `DisturbanceSmootherResult` holds ε̂ₜ, η̂ₜ, their conditional covariances and the auxiliary residuals (Durbin & Koopman §4.5.3). It comes from `disturbance_smoother` on a `KalmanFilterResult`, on `(p, y, a1, P1)`, on a `KalmanWorkspace` and on a `StateSpaceModel`, and in place from `disturbance_smoother!(out, ws)`. The results match brute-force Gaussian conditioning to 1e-12, including partly and fully missing periods, a full H and the rank-revealing path.

## Key decisions / shim choices
- Separate result type and a standalone backward pass in `src/disturbance_smoother.jl`. It reruns the r/N recursion, so it needs only the filter. `KalmanWorkspace` is unchanged.
- `auxiliary_residuals(ds)` returns `(obs, state)` and is NaN where the estimator variance is exactly zero: unobserved periods, and ηₙ.
- Exact diffuse inputs throw an `ArgumentError` (new Open Question). There are no EKF or DFM methods.
- No conversion-guide entry, since this is a pure addition. The docs now cover it in `api/core.md` and a `getting_started.md` section, which makes the `index.md` claim true.

## State of the codebase
- Files modified: src/Siphon.jl (include and exports), src/show.jl (show method), test/runtests.jl, docs/src/api/core.md, docs/src/tutorials/getting_started.md, DESIGN_REVIEW_PLAN.md
- New files: src/disturbance_smoother.jl, test/disturbance_smoother_tests.jl
- Test suite: full `Pkg.test()` passes (includes the new `disturbance_smoother_tests.jl`, 47 tests)
- Ambiguity count: 0 (no change)
- Staged but uncommitted: no (all changes unstaged)

## Cluster status
- missing-operations: 2 of 3 complete (CHUNK-029 remains)

## Next chunk
CHUNK-029: add-ekf-smoother-and-model. A functional AD EKF smoother, StateSpaceModel/EKF model support (`fit!(MLE())`, `smoothed_states`, `forecast`), a built-in linear measurement type, and a decision on `FiniteDiffJacobian` (implement or drop). This is a large chunk, so consider splitting it.

## Watch out for
- The in-place disturbance smoother uses `ws.r_smooth`/`N_smooth`/`L_smooth` and the tmp scratch. Any future code that reads those after `kalman_smoother!` would be clobbered by it. Nothing does today.
- The `@allocated` test compares an n-period run against a 5n-period run, and asserts equal allocations rather than zero.
