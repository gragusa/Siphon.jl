# Session Handoff — 2026-10-12

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-018: unify-static-backend-controls
- **One keyword:** every spec-level entry point takes `static = :auto | :on | :off`, with `:auto` as the default.
- **One gating rule:** `_use_static` in types.jl checks max(n_states, n_obs, n_shocks) ≤ `STATIC_THRESHOLD`.
- **All-or-nothing conversion:** `_static_system` converts all seven arrays together.
- **Removed:** `KFParms_static`, `to_static_if_small`, `select_backend`, `static_threshold` and `backend(model)`/`model.backend`.

## Key decisions / shim choices
- No shims. There is a new "StaticArrays" section in the conversion guide.
- `backend(model)` was removed outright. This is the implementer's call and is recorded in Decisions as unconfirmed.
- The EKF lost its no-op `use_static` keyword instead of gaining `static`.
- `STATIC_THRESHOLD` stays exported; the package has no `public` declarations yet.
- `static = :on` with `diffuse = :exact` now throws. Before, it silently used plain arrays.

## State of the codebase
- Files modified:
  - src: types.jl, Siphon.jl, inplace.jl, predict.jl, dsl/{dsl,codegen,bayesian,optimization,ekf_optim,ekf_codegen}.jl, api/core.md
  - test: api_coverage, em, failure, missing_input, partial_missing, dsl
  - docs/src: conversion_guide.md, api/core.md, tutorials/estimation_methods.md
- Test suite: pass (`Pkg.test()` from the shell; the Julia MCP server failed to connect). 2 broken pins unchanged.
- Static and dynamic paths differ under ill-conditioned F (Open Question); the new test uses a well-conditioned `P1`.
- Ambiguity count: 0 (no change).
- Staged but uncommitted: changes are unstaged and uncommitted.

## Cluster status
- api-unification: 5 of 5 complete.

## Next chunk
CHUNK-019: consolidate-em. One `EMResult`, `fit!(EM())` canonical, `profile_em_ssm` rebuilt on it (fixes B9), and dead static-EM/DFM helpers deleted.

## Watch out for
- The dead static EM block in inplace.jl (`_ssm_fit_em_static!` and friends) was migrated to `static = :auto` only so that it compiles; CHUNK-019 deletes it.
- `ssm_loglik` now defaults to the static path for small models. Static and dynamic log-likelihoods differ at about 1e-13, so compare with `≈`, never `==`.
- A background check found that Julia 1.10 passes the suite with `julia = "1.10"` (Open Questions). Supporting 1.10 affects how CHUNK-031 declares `public`.
- Don't run `Pkg.test()` while editing `src/` or `test/`. Don't `Pkg.develop` into `test/`.
