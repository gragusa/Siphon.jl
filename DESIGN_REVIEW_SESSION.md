# Session Handoff — 2026-10-08

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-038: fix-exact-diffuse-update-sign
The Finf-invertible step of the exact diffuse filters had the wrong sign on the `K∞ F*` term of K⁽¹⁾ and on the `K∞ F∞ K⁽¹⁾'` term of the P* update. That made P* asymmetric and wrong from the second diffuse step on. All three filters now follow Durbin & Koopman (2012, §5.2), and the filtered covariance stored during a diffuse step is now the correct κ → ∞ limit in both APIs.

## Key decisions / shim choices
- The in-place filter stores `K⁽⁰⁾` (the gain that updates the state) in `Kt` during the diffuse period, matching the functional filter. Before, it stored K⁽¹⁾.
- The in-place flag-1 branch is written around `A = P∞ Z' F∞⁻¹`. It predicts through `P*ₜ|ₜ`, which is algebraically identical to the K⁽⁰⁾/K⁽¹⁾ recursion. The functional filter keeps the explicit DK recursion, so the two forms cross-check each other in the tests.
- `Ptt` during the diffuse period holds only the finite part `P*ₜ|ₜ`. The diffuse part of the filtered covariance is not stored (the predicted one is, in `Pinf_store`).

## State of the codebase
- Files modified: `src/filter_ad.jl`, `src/inplace.jl`, `src/types.jl`, `test/diffuse_tests.jl`, `test/known_bugs_tests.jl`
- Test suite: pass (`Pkg.test()`); Known bugs shows 4 broken assertions (DFM AR-error loadings, profile_em_ssm θ, EM diagonal T, singular nonzero Finf)
- Ambiguity count: 0 (no change)
- Staged but uncommitted: yes (src and test files)

## Cluster status
- correctness: 9 of 11 complete (037 and 039 remain)

## Next chunk
CHUNK-039: exact-diffuse-singular-finf. Univariate (sequential) treatment of periods where `Finf` is singular but nonzero, or at least throw instead of returning a wrong likelihood.
(CHUNK-037, fix-dfm-ar-error-loadings, is also ready.)

## Watch out for
- The univariate treatment in CHUNK-039 needs diagonal H, or an LDL transform of H_t per period (`H[rows,rows]` for partial periods). The non-diagonal H used in `partial_missing_tests.jl` exercises this.
- After CHUNK-039, a partial period inside a multi-step diffuse period (m = 2) becomes testable; add one, since the CHUNK-036 tests stuck to d = 1.
- The "exact diffuse with a singular nonzero Finf" pin is the last diffuse entry in `known_bugs_tests.jl`.
- The `no-plan-comments.py` PostToolUse hook in the user's settings points at a missing file; it errors after every Edit or Write, but the edits apply.
