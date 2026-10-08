# Session Handoff — 2026-10-08

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-039: exact-diffuse-singular-finf
A diffuse period where `Finf = Z_t Pinf Z_t'` is singular but nonzero is now processed one row at a time (Koopman & Durbin 2000), and is marked with the new flag `2`. Before, these periods took the Fstar step, so the diffuse period never ended and the likelihood was wrong. All three diffuse filters share one non-allocating kernel, `_univariate_diffuse_update!` in `src/filter_ad.jl`, which decorrelates a non-diagonal `H_t` by its unit-LDL factor first.

## Key decisions / shim choices
- New flag value `2`. Flag `0` now means only "Finf is zero". Classification is in `_diffuse_step_flag`; `_safe_inverse` was deleted.
- A flag-2 step stores `Ft = Finf`, `Kt = T G` (the gain with `a_filt = a + G v`) and `Ptt` = the filtered Pstar, so the predict identities hold in every period.
- Likelihood convention: like the flag-1 step, diffuse rows drop their `log Finf,ᵢ` and 2π terms.
- The multivariate flag-1 step is unchanged (the two forms are not cross-checked against each other directly, but both match the κ-limit).

## State of the codebase
- Files modified: `src/filter_ad.jl`, `src/inplace.jl`, `src/types.jl`, `test/diffuse_tests.jl`, `test/known_bugs_tests.jl`
- Test suite: pass (`Pkg.test()`); Known bugs shows 3 broken pins (DFM AR-error loadings, profile_em_ssm θ, EM diagonal T)
- Ambiguity count: 0 (no change)
- Staged but uncommitted: yes (src and test files)

## Cluster status
- correctness: 10 of 11 complete (CHUNK-037 remains)

## Next chunk
CHUNK-037: fix-dfm-ar-error-loadings — with `error_lags > 0` the DFM sets H = 0 and the EM loadings update is a fixed point. Reparameterise (quasi-difference the observation equation) so EM can move Λ.

## Watch out for
- Float64 approximate-diffuse references at κ = 1e8 lose precision through cancellation when several rows load on one diffuse direction. The new tests use BigFloat at κ = 1e20 (`setprecision(BigFloat, 256)`); the functional filters accept BigFloat `KFParms`.
- The flag-2 failure path (`Fstar,ᵢ < −tol`) returns `-Inf`; the in-place filter returns early and leaves the remaining periods unwritten. CHUNK-013 (failure model) should cover this.
- The `no-plan-comments.py` PostToolUse hook in the user's settings points at a missing file; it errors after every Edit or Write, but the edits apply.
