# Session Handoff — 2026-10-14

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-032: sync-docs
The docs build is now strict (`checkdocs = :public`, no `warnonly`) and passes. Every export and `public` name is in an `@docs` block, with two new API pages (`api/inplace.md`, `api/dfm.md`). All docstring examples in `src/` and all tutorial code blocks were run against the current API, and the broken ones were fixed. `index.md` and the README now describe the package as it is.

## Key decisions / shim choices
- `checkdocs = :public`: internal docstrings do not need to be in the manual.
- Whole-binding `@docs` entries for `kalman_filter!`, `kalman_smoother!` and `kalman_loglik`. Method entries for parametric signatures do not match in Documenter.
- Tutorial code is still not executed by the docs build (plain ```julia blocks). Verification used a block-by-block runner in the scratchpad (not committed).
- Two package bugs found while running the tutorials became new chunks instead of being fixed here: CHUNK-042 (static kernel gives `-Inf` on the DNS example; `optimize_ssm` reports convergence at `-Inf`) and CHUNK-043 (EM on `arma(2, 0)` → `PosDefException` in the smoother).

## State of the codebase
- Files modified: docs/make.jl, docs/Manifest.toml, docs/src/index.md, docs/src/api/{core,dsl,optimization}.md, docs/src/api/{inplace,dfm}.md (new), docs/src/tutorials/{arma_models,custom_models,dynamic_factor,ekf,estimation_methods,getting_started,transformations}.md, README.md, src/dsl/codegen.jl, src/dsl/optimization.jl, src/ekf_codegen.jl (docstrings only)
- Test suite: full `Pkg.test()` passes (Aqua included); strict docs build passes
- Ambiguity count: 0 (no source code changes beyond docstrings)
- Staged but uncommitted: no (all unstaged)

## Cluster status
- release: 3 of 7 complete (CHUNK-030, 031, 032; remaining 042, 043, 033, 034)

## Next chunk
CHUNK-042: fix-static-kernel-numerics. The static multivariate filter returns `-Inf` where the dynamic path gives a finite log-likelihood (DNS, 8 maturities, P1 = 1e7). Also make `optimize_ssm` fail loudly when the objective is `-Inf`.

## Watch out for
- Reproducer for CHUNK-042: the simulation block plus the "Approach 1" block of `docs/src/tutorials/estimation_methods.md`; `ssm_loglik(spec, θ0, yields; static = :on)` is `-Inf`, `:off` is ≈ 2881.
- After CHUNK-042, rerun `estimation_methods.md` blocks 91/116/174/232. The quoted numbers in that tutorial were never checked against output.
- `docs/src/tutorials/ekf.md` quotes `loglik ≈ 2377`, `μ_L ≈ 3.7` from one run (seed 0); if the EKF numerics change, update them.
