# Session Handoff — 2026-10-14

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-027: add-diffuse-smoother
Exact diffuse smoothing now works in the functional API (`kalman_smoother(::DiffuseFilterResult)`), the in-place API (`kalman_smoother!` / `kalman_filter_and_smooth!` on `DiffuseKalmanWorkspace`), and the model API (`smoothed_states(model)` with `diffuse = :exact`). One backward routine in `src/diffuse_smoother.jl` handles every diffuse step type by processing rows one at a time. It matches a BigFloat large-κ oracle to ~1e-15. `DiffuseFilterResult(ws)` was added, and the in-place diffuse filter and smoother are allocation-free.

## Key decisions / shim choices
- Univariate (row-by-row) backward pass for all diffuse periods, rebuilt from the stored predicted Pstar/Pinf. It uses the exact κ-expansion coefficients, including the L⁽²⁾ terms in N⁽²⁾.
- `DiffuseFilterResult` gains `diffuse_ended` and `tol` fields. `DiffuseKalmanWorkspace` gains `Pinf_store` (m × m × n) and `smoother_scratch`.
- `crosscov = true` throws under exact diffuse. EM and the simulation smoother still reject `DiffuseKalmanWorkspace`, with corrected reasons.
- The model API smooths by rerunning the functional diffuse filter on `model.data`.
- No shims, per the stated values. The conversion guide has a paragraph on the new capability and the new fields.

## State of the codebase
- Files modified: src/Siphon.jl, src/filter_ad.jl, src/smoother_ad.jl, src/inplace.jl, src/types.jl, src/predict.jl, src/simsmooth.jl, test/runtests.jl, test/smoother_interface_tests.jl, test/dsl_tests.jl, docs/src/conversion_guide.md, docs/src/tutorials/initial_state.md, docs/src/tutorials/getting_started.md, DESIGN_REVIEW_PLAN.md
- New files: src/diffuse_smoother.jl, test/diffuse_smoother_tests.jl
- Test suite: full `Pkg.test()` passes (includes the new `diffuse_smoother_tests.jl`)
- Ambiguity count: 0 (no change)
- Staged but uncommitted: no (all changes unstaged)

## Cluster status
- missing-operations: 1 of 3 complete (CHUNK-028 and CHUNK-029 remain)

## Next chunk
CHUNK-028: add-disturbance-smoother. Smoothed ε̂, η̂ and their variances (D&K §4.5) in the functional and in-place APIs. The r/N recursions in `_rts_step` and `_kalman_smoother_cov!` are the natural places to hook in (ε̂ₜ = H Fₜ⁻¹vₜ − H Kₜ' rₜ, η̂ₜ = Q R' rₜ).

## Watch out for
- A disturbance smoother under exact diffuse would need the r⁽⁰⁾/r⁽¹⁾ quantities from `_diffuse_smoother_backward!`. Either restrict it to non-diffuse results first, or extend that routine.
- `_diffuse_step_flag!` keeps the scale-dependent `|det Finf| > tol` criterion (see Open Questions).
- Revise cannot redefine structs. After this chunk, restart the MCP session before testing.
