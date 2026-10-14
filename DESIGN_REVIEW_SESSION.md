# Session Handoff — 2026-10-14

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-042: fix-static-kernel-numerics, and CHUNK-043: fix-em-arma-smoother-posdef (one session, at the author's request)

- **CHUNK-042:** The static `kalman_loglik` and both `kalman_filter` kernels now use Cholesky solves in place of `inv(chol(F))`, and they symmetrize the filtered covariance. Under an ill-conditioned F they agree with the dynamic `kalman_loglik`. Maximum likelihood now throws when the log-likelihood at the starting values is not finite, and never reports convergence at `-Inf`.
- **CHUNK-043:** The lag-one cross covariance in the in-place, EKF in-place and functional smoothers now uses `(I − P_t N_{t−1}) T P_{t−1|t−1}`, with no inverse. `fit!(EM())` throws for `T` parameters in rows whose state has no shock (for example, `φ2…` in `arma(p, 0)` with p > 1), because EM cannot move them.

## Key decisions / shim choices
- The cross-lag identity is D&K (2012) §4.7. It equals the old `V_t J'` whenever `P_t` is invertible, which the MARSS EM validation tests confirm.
- EM supports `arma(1, 0)` only; higher AR orders need MLE (recorded in the CHUNK-043 notes, ARMA tutorial updated).
- The `J_smooth` scratch fields of `KalmanWorkspace` and `EKFWorkspace` were removed (internal).

## State of the codebase
- Files modified: src/filter_ad.jl, src/dsl/optimization.jl, src/inplace.jl, src/ekf_inplace.jl, src/smoother_ad.jl, src/em_spec.jl, test/api_coverage_tests.jl, test/failure_tests.jl, test/em_tests.jl, docs/src/tutorials/arma_models.md, DESIGN_REVIEW_PLAN.md
- Test suite: full `Pkg.test()` passes (Aqua included)
- Ambiguity count: 0 (no change)
- Staged but uncommitted: no (all unstaged)

## Cluster status
- release: 5 of 7 complete (remaining CHUNK-033, CHUNK-034)

## Next chunk
CHUNK-033: finalize-conversion-guide. Complete `docs/src/conversion_guide.md`, ordered by workflow, and link it from the README.

## Watch out for
- The conversion guide should mention two behavior changes. `optimize_ssm`/`fit!(MLE())` now throw on a non-finite starting log-likelihood. `fit!(EM())` now rejects `arma(p, 0)` for p > 1 and any `T` parameter in a row without a shock.
- Open questions added: the exact diffuse kernels and `_rts_step` still invert F explicitly; and a lagged-observation layout for `arma(p, 0)` would let EM estimate pure AR models.
