# Session Handoff — 2026-10-08

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-012: introduce-abstract-supertypes-and-accessor-vocabulary (first breaking chunk)
Filter results subtype `AbstractFilterResult` and the in-place workspaces subtype `AbstractFilterWorkspace`. The accessors live once in `src/accessors.jl` and read through `_filter_store`. The `variances_*` names became `*_cov`. `system_matrices` returns a `KFParms`/`EKFParms` everywhere, `parameters` is θ on models only, and there are new `initial_state`/`initial_state_cov`. The single-matrix setters are now `set_Z!`…`set_Q!`. `docs/src/conversion_guide.md` was created.

## Key decisions / shim choices
- No shims (author: "forget about breaking").
- `system_matrices(model)` returns `KFParms` instead of a NamedTuple; field access is unchanged.
- `obs_matrix` … `state_cov` are untyped generics `f(x) = system_matrices(x).Z` etc. Any type with `system_matrices` gets them.
- `AbstractStateSpaceModel` owns `isfitted`/`isconverged`/`niterations`/`loglikelihood` via `_require_fitted` + `_fit_hint`.

## State of the codebase
- Files modified: src/{accessors.jl (new), Siphon.jl, types.jl, filter_ad.jl, ekf_types.jl, ekf_inplace.jl, inplace.jl, recipes.jl}; test/{accessor_tests.jl (new), runtests.jl, arma_tests.jl, diffuse_tests.jl, partial_missing_tests.jl, recipe_tests.jl, simsmooth_tests.jl}; docs/{make.jl, src/conversion_guide.md (new), src/api/optimization.md, src/tutorials/ekf.md, src/tutorials/visualization.md}
- Test suite: pass (`Pkg.test()`), 2 broken pins unchanged, Aqua 10/10
- Ambiguity count: 0 (no change)
- Staged but uncommitted: yes

## Cluster status
- type-hierarchy: 1 of 3 complete (CHUNK-013, CHUNK-014 remain; CHUNK-014 is api-unification but depends on 012)

## Next chunk
CHUNK-013: apply-failure-model. Apply policy (i)–(v): no uninitialised arrays, `DimensionMismatch` in place of `@assert`, accessors on unrun workspaces throw, a convergence `@warn`, and so on.

## Watch out for
- Every breaking chunk must add rows to `docs/src/conversion_guide.md`.
- Workspace accessors still return garbage before a run. CHUNK-013 should add a "has run" flag; the shared accessors in `accessors.jl` are the single place to check it (e.g. inside `_filter_store` or a wrapper).
- DFM still lacks `system_matrices`/filter accessors and keeps `innovation_cov` (CHUNK-020).
- The docs build (`docs/make.jl`) was not run this session.
- Don't run `Pkg.develop` in `test/`. It creates `test/Project.toml` and rewrites the tracked `test/Manifest.toml`. Use `Pkg.test()`.
- The `no-plan-comments.py` PostToolUse hook still errors after every Write (the file is missing); the edits apply.
