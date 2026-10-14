# Session Handoff — 2026-10-14

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-030: remove-dead-exports-and-utilities
Removed `SmootherWorkspace`, `ParamExpr`, `numstates`/`nummeasur` and the stray `src/api/*.md`. Unexported `STATIC_THRESHOLD` and `confidence_bands`, and made `quantile_normal`/`select_vars` private (`_`-prefixed). `StateSpaceModel` lost `kf_workspace_allocated` and now holds a typed `const workspaces` field. Stale comments and docstrings were fixed, and the open-question items assigned to this chunk were picked up (`@info` for MLE verbose output, an EKF-aware fit hint, the ported EM benchmark).

## Key decisions / shim choices
- `block_diag` and `uses_exact_diffuse` stay exported (both are used; `block_diag` is a FreeParam-aware DSL helper).
- No shims. The conversion guide has a new "Removed names and utilities" section.
- DNS loading builders are still exported; CHUNK-031 should unexport them and declare them `public`.

## State of the codebase
- Files modified: src/Siphon.jl, src/types.jl, src/show.jl, src/recipes.jl, src/inplace.jl, src/em_spec.jl, src/dsl/expressions.jl, src/dsl/codegen.jl, ext/SiphonStatsBaseExt.jl, benchmark/em_benchmark.jl, test/{api_coverage,construction,recipe,em,ekf}_tests.jl, docs/src/{conversion_guide.md, api/core.md, api/dsl.md, tutorials/visualization.md, tutorials/estimation_methods.md}; deleted src/api/*.md
- Test suite: full `Pkg.test()` passes (4.3 min, Aqua included); new testset "Removed and unexported names" (11 tests)
- Ambiguity count: 0 (no change)
- Staged but uncommitted: no (the `src/api` deletions are staged through `git rm`; everything else is unstaged)

## Cluster status
- release: 1 of 5 complete

## Next chunk
CHUNK-031: annotate-public-api. Declare `public` for every user-facing unexported name. New from CHUNK-030: `STATIC_THRESHOLD`, `confidence_bands`, `StateSpaceModelWorkspaces`(?), and the DNS builders, which also need unexporting.

## Watch out for
- `public` needs Julia ≥ 1.11 or `Compat.@compat public`; the compat bound question (1.10 vs 1.12) is still open.
- Docstrings link `[`Siphon.STATIC_THRESHOLD`](@ref)`; the docs build hasn't been run (CHUNK-032).
- The R/MARSS half of `benchmark/em_benchmark.jl` was not run.
