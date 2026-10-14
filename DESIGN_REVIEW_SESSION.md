# Session Handoff — 2026-10-14

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-026: unify-construction-conventions
The element type of every workspace and model is now a type parameter (`StateSpaceModel{Float32}(spec, n)`, `EMWorkspace{T}(p, m, r, n)`, …). Constructors that take system arrays infer a floating-point type from all of them. `simulation_smoother` returns a matrix without a draw count and a 3-D array with one. The EM and simulation-smoother constructors give a clear error for a `DiffuseKalmanWorkspace`.

## Key decisions / shim choices
- Type parameter over keyword or positional `Type` argument (implementer's call, per the style guide's constructor cascade). No shims; the conversion guide has a new "Element types and constructors" section.
- `simulation_smoother(ws, y, n_draws)` is positional, following `rand(d)` / `rand(d, n)`.
- The `rng` item needed no change: every random function already took `rng`.
- `StateSpaceModel` with a non-Float64 element type was broken before (its `KalmanWorkspace` was always Float64) and now works through MLE and EM.

## State of the codebase
- Files modified: src/inplace.jl, src/ekf_inplace.jl, src/simsmooth.jl, src/types.jl, test/runtests.jl, test/simsmooth_tests.jl, test/companion_smoother_tests.jl, test/singular_innovation_tests.jl, docs/src/conversion_guide.md, DESIGN_REVIEW_PLAN.md
- New files: test/construction_tests.jl
- Test suite: full `Pkg.test()` passes (includes the new `construction_tests.jl`)
- Ambiguity count: 0 (no change)
- Staged but uncommitted: no (all changes unstaged)

## Cluster status
- composability-with-base: 3 of 3 complete

## Next chunk
CHUNK-027: add-diffuse-smoother. Exact diffuse smoothing (Durbin & Koopman §5.3) for `DiffuseFilterResult` and `DiffuseKalmanWorkspace`; remove the `det` allocation in `_safe_inverse_inplace!`; document `diffuse_flags == -1`. The smoother must read the packed `vt`/`Ft`/`Kt` blocks.

## Watch out for
- Open Questions lists diffuse items that belong in CHUNK-027: `DiffuseFilterResult(ws)` conversion, `SmootherResult(ws)` on a diffuse workspace, and a `diffuse_ended` flag for `forecast`.
- After CHUNK-027, consider letting `SimulationSmootherWorkspace`/`simulation_smoother` accept a `DiffuseKalmanWorkspace`; their `ArgumentError` messages and `test/construction_tests.jl` would change.
