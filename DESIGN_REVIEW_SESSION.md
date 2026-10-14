# Session Handoff — 2026-10-14

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-029: add-ekf-smoother-and-model (first half; the model part was split into CHUNK-041)
The extended Kalman filter now smooths in the functional API: `kalman_smoother(::EKFFilterResult)`, plus `kalman_smoother`/`kalman_filter_and_smooth` on an `EKFParms`. It reuses `_kalman_smoother`, which now also takes a per-period Z array. `LinearMeasurement(Z)` is a built-in, exported measurement type. `EKFWorkspace` supports `FiniteDiffJacobian()` without allocating.

## Key decisions / shim choices
- Author's decisions: generalize `StateSpaceModel` for EKF specs, rather than adding a separate model type, and split the chunk (both recorded under Decisions → CHUNK-029).
- The EKF smoother methods live in `src/ekf_ad.jl`, because `smoother_ad.jl` loads before the EKF types.
- The in-place finite-difference step matches the functional path, so the first-period Jacobians are bit-identical.
- No shims. The conversion guide notes the new capabilities and the possible `LinearMeasurement` name clash.

## State of the codebase
- Files modified: src/smoother_ad.jl, src/ekf_ad.jl, src/ekf_types.jl, src/ekf_inplace.jl, src/Siphon.jl, test/ekf_tests.jl, test/accessor_tests.jl, docs/src/tutorials/ekf.md, docs/src/api/ekf.md, docs/src/conversion_guide.md, DESIGN_REVIEW_PLAN.md
- Test suite: full `Pkg.test()` passes (25 new EKF tests)
- Ambiguity count: 0 (no change)
- Staged but uncommitted: no (all changes unstaged)

## Cluster status
- missing-operations: 3 of 4 complete (CHUNK-041 remains)

## Next chunk
CHUNK-041: ekf-statespacemodel. `StateSpaceModel{T, S<:AbstractSSMSpec}` holding an `EKFSpec`, with `fit!(MLE())`, `kalman_filter!(model, y)`, `smoothed_states[_cov]`, `forecast(model, h)` and the StatsAPI accessors. `fit!(EM())` on an EKF spec throws.

## Watch out for
- `StateSpaceModel` has the field `spec::SSMSpec`, and many methods in `inplace.jl` (~3360–4100) assume `build_kfparms`, `observed_mask` and the stored `Kt`. Survey them with `grep -n "model::StateSpaceModel" src/*.jl` before choosing between dispatching on the spec and a function barrier.
- Changing the `StateSpaceModel` struct needs an MCP session restart (Revise cannot redefine structs).
- `disturbance_smoother(model)` and the DFM paths assume a linear spec. They should throw a clear error for EKF specs.
