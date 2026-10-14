# Session Handoff — 2026-10-14

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-041: ekf-statespacemodel
`StateSpaceModel` is now `StateSpaceModel{T, S <: AbstractSSMSpec}` and accepts an `EKFSpec`. It supports `fit!(MLE())`, `kalman_filter!`/`kalman_loglik`, the filter and smoother accessors, `forecast` and the StatsAPI accessors, including `vcov`/`stderror`. `fit!(EM())`, `obs_matrix` and `disturbance_smoother` throw clear `ArgumentError`s for EKF models.

## Key decisions / shim choices
- Spec dispatch is in four small helpers in `inplace.jl` (`_model_parms`, `_model_filter`, `_model_loglik`, `_model_optimize`), not in branches inside each method.
- EKF smoothing reruns the functional EKF filter on the stored data, because the model does not keep the per-period Jacobians. This is the same approach as the exact diffuse path.
- For EKF models, `static = :on` throws, and `:auto`/`:off` are accepted.
- The EKF constructor stores `nothing` in `workspaces_ref` (no EM workspace).
- `_param_vcov` now takes any `AbstractSSMSpec`.
- No shims. The conversion guide notes the new type parameter: `typeof(model) == StateSpaceModel{Float64}` is now false, but dispatch and `isa` on `StateSpaceModel{Float64}` still work.

## State of the codebase
- Files modified: src/inplace.jl, src/em_spec.jl, src/disturbance_smoother.jl, src/dsl/optimization.jl, test/ekf_tests.jl, docs/src/tutorials/ekf.md, docs/src/api/ekf.md, docs/src/conversion_guide.md, DESIGN_REVIEW_PLAN.md
- Test suite: full `Pkg.test()` passes (4.3 min; 36 new tests in 3 new EKF testsets)
- Ambiguity count: 0 (no change)
- Staged but uncommitted: no (all changes unstaged)

## Cluster status
- missing-operations: 4 of 4 complete
- release: 0 of 5 complete

## Next chunk
CHUNK-030: remove-dead-exports-and-utilities. Removes dead exports (`SmootherWorkspace`, `select_backend`, `ParamExpr`, …), the `kf_workspace_allocated` field, and stale comments and docstrings. Breaking; the conversion guide must be updated in the same commit.

## Watch out for
- The new testset "EKF DSL — StateSpaceModel matches the linear spec" takes about 40 s, almost all of it compilation (the EKF MLE and the nested-AD Hessian for an 8-parameter model). If suite time matters, use a smaller model there.
- The `_require_filter`/`_fit_hint` messages still suggest `fit!(EM(), …)` to EKF models (Open Questions).
- Removing `kf_workspace_allocated` in CHUNK-030 means editing `_empty_model` (positional field list) and needs an MCP restart. `workspaces_ref` is still a `Ref{Any}`.
