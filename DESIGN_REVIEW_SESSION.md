# Session Handoff — 2026-10-08

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-036: partial-periods-in-exact-diffuse-filters
The three exact diffuse filters (`kalman_loglik_diffuse`, `kalman_filter_diffuse`, `kalman_filter_diffuse!`) no longer reject partially observed periods. Each period runs on the reduced system of its observed rows, the likelihood constant counts observed scalars, and innovations are stored packed in leading rows with an `observed_mask`, the same layout as the non-diffuse filters. `DiffuseFilterResult` gained an `observed_mask` field, and the in-place filter was rewritten on the `kalman_filter!(::KalmanWorkspace)` pattern.

## Key decisions / shim choices
- `DiffuseFilterResult` has a new last positional field, `observed_mask`; there is no compatibility constructor (the only constructor call is in `filter_ad.jl`).
- In-place: the Finf-singular diffuse step and the post-diffuse step share one branch, because the algebra is the same plus `Pinf = T Pinf T'`. `M = Pstar Z' F⁻¹` is formed from `Pstar Z'` directly, because `Pstar` can be asymmetric (bug below).
- Kept the existing per-filter conventions: the functional filter stores Kinf in `Kt` during the diffuse period and the in-place one stores Kstar. The functional `Ptt` during diffuse uses the Finf⁻¹ formula; the in-place one stores `Pstar`.
- Two pre-existing bugs were found and **not fixed**; they are scoped as CHUNK-038 (sign error) and CHUNK-039 (singular nonzero Finf), both pinned with `@test_broken` in `known_bugs_tests.jl`.

## State of the codebase
- Files modified: `src/filter_ad.jl`, `src/inplace.jl`, `src/types.jl`, `test/partial_missing_tests.jl`, `test/known_bugs_tests.jl`
- Test suite: pass (`Pkg.test()`); Known bugs shows 6 broken assertions: DFM AR-error loadings, profile_em_ssm θ, EM diagonal T, plus 3 from the two new diffuse pins
- Ambiguity count: 0 (no change)
- Staged but uncommitted: yes (src and test files; the plan and handoff are left unstaged)

## Cluster status
- correctness: 8 of 11 complete (006, 007, 035, 008, 009, 010, 011, 036 done; 037, 038, 039 remain)

## Next chunk
CHUNK-038: fix-exact-diffuse-update-sign. Flip the sign of the `Kinf F*` term in Kstar and of the `Kinf Finf Kstar'` term in the Pstar update, in all three diffuse filters, then flip the pin.
(CHUNK-037, fix-dfm-ar-error-loadings, is also ready.)

## Watch out for
- After CHUNK-038, `kalman_loglik` for exact-diffuse specs with d ≥ 2 changes value (e.g. `local_linear_trend(diffuse=:exact)`). Check `diffuse_tests.jl` and any MARSS/KFAS pins. The Nile pin (d = 1) must not move.
- The new partial-missing tests deliberately use d = 1 setups. CHUNK-039 should add one with a partial period inside a multi-step diffuse period.
- `custom_ssm` cannot express exact diffuse; the test builds `Siphon.DSL.SSMSpec(..., P1_inf, base.matrix_exprs)` by hand. CHUNK-022 may want a `P1_inf` keyword.
- The `no-plan-comments.py` PostToolUse hook in the user's settings points at a missing file; it errors after every Edit, but the edits apply.
