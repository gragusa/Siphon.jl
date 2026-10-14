# Session Handoff — 2026-10-14

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.2.0

## What was just completed
CHUNK-033: finalize-conversion-guide, and CHUNK-034: version-bump (one session, at the author's request)

- **CHUNK-033:** `docs/src/conversion_guide.md` is ordered by workflow. It opens with before/after snippets for the common workflows and has a "Results that change" section for the bug fixes that change numbers without breaking code. The README links the guide.
- **CHUNK-034:** `Project.toml` is at 0.2.0.

## Key decisions / shim choices
- No CHANGELOG; the conversion guide is the release note for 0.1 → 0.2.
- The "0.1" sides of the workflow snippets come from the guide's tables; only the "Now" sides were executed.

## State of the codebase
- Files modified: Project.toml, docs/src/conversion_guide.md, README.md, DESIGN_REVIEW_PLAN.md, DESIGN_REVIEW_SESSION.md
- Test suite: full `Pkg.test()` passes (Aqua 10/10); strict docs build passes
- Ambiguity count: 0 (no change from baseline)
- Staged but uncommitted: no (all unstaged)

## Cluster status
- release: 7 of 7 complete

## Next chunk
None. Every chunk is complete or dropped.

## Watch out for
- The tag and Julia registry registration for 0.2.0 are the author's actions.
- A docs build rewrites the tracked files under `docs/build/`; run `git checkout -- docs/build` afterwards (see Open Questions).
- The Open Questions list holds possible follow-up work (explicit inverses in the exact diffuse kernels and `_rts_step`, the lagged-observation `arma(p, 0)` layout for EM, the scale-invariant diffuse criterion, and others).
