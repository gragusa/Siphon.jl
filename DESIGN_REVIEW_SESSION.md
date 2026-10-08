# Session Handoff — 2026-10-08

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-037: fix-dfm-ar-error-loadings
`DynamicFactorModel` with AR idiosyncratic errors no longer puts the errors in the state with `H = 0`, which made the EM loadings update a fixed point. The observation equation is now quasi-differenced (`X̃ₜ = λ(L)(1 − δ(L)L) fₜ + vₜ`, `H = diag(σ²)`), and the EM alternates loadings, δ and σ² (ECM). Forecasting with AR errors iterates an augmented state of factors plus the last r observations.

## Key decisions / shim choices
- With AR errors the likelihood is conditional on the first r observations. A period contributes only if it and its r predecessors are observed. Documented in the `fit!` docstring and the DFM tutorial.
- The shock dimension is now k for all DFMs (R = [I_k; 0], Q = Σ_η). The unused idiosyncratic shock columns were dropped.
- Internal helpers `compute_error_sufficient_stats!` and `update_ar_errors!` were replaced by `_update_dfm_ar!`. Workspace fields changed (all internal, none exported).

## State of the codebase
- Files modified: `src/inplace.jl`, `test/dfm_tests.jl`, `test/known_bugs_tests.jl`, `docs/src/tutorials/dynamic_factor.md`
- Test suite: pass (`Pkg.test()`); Known bugs shows 2 broken pins (profile_em_ssm θ, EM diagonal T)
- Ambiguity count: 0 (no change)
- Staged but uncommitted: yes

## Cluster status
- correctness: 11 of 11 complete

## Next chunk
CHUNK-012: introduce-abstract-supertypes-and-accessor-vocabulary. This is the first breaking chunk. It adds the abstract result and workspace types, one accessor vocabulary (`filtered_states_cov` style), `parameters` / `system_matrices` semantics, and renames the setter `update_*!` functions that collide with the EM M-step names. It must create `docs/src/conversion_guide.md` and add it to `docs/make.jl`.

## Watch out for
- CHUNK-012 renames the single-matrix setters `update_Z!` … `update_Q!` (inplace.jl ~505-519), which share names with the EM M-step functions the DFM still calls (`update_T!`, `update_Q!`, `update_Z!`, `update_H!`).
- The DFM tests use `Random.seed!(3)` before construction because the initial loadings use the global RNG. A CHUNK-022/026 `rng` keyword should keep that draw order or update the tests.
- New Open Questions: `compute_sufficient_stats!` has an S_αα vs S_yα period mismatch with missing periods (for CHUNK-019), and the DFM tutorial has a stray `include("src/inplace.jl")` (for CHUNK-032).
- The `no-plan-comments.py` PostToolUse hook in the user's settings points at a missing file; it errors after every Edit or Write, but the edits apply.
