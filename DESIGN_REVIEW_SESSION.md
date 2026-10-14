# Session Handoff — 2026-10-14

## Plan
DESIGN_REVIEW_PLAN.md — Siphon 0.1.0

## What was just completed
CHUNK-031: annotate-public-api
Set the compat bound to `julia = "1.13"`, since the package targets 1.13+ only. Declared 33 unexported user-facing names `public` with the `public` keyword (block at the end of `src/Siphon.jl`). The DNS loading builders are no longer exported and became public. Added docstrings for the newly public names that lacked them.

## Key decisions / shim choices
- Author's decision: Julia 1.13+ only; do not test on 1.10 (recorded in Decisions).
- Internal names are not renamed to `_name` (divergence recorded in the chunk notes).
- No shims; the conversion guide notes the Julia 1.13 requirement, what `public` means, and the DNS builder move.

## State of the codebase
- Files modified: Project.toml, src/Siphon.jl, src/inplace.jl, test/api_coverage_tests.jl, test/dsl_tests.jl, test/dns_marss_validation_tests.jl, docs/src/conversion_guide.md, docs/src/tutorials/custom_models.md
- Test suite: full `Pkg.test()` passes on Julia 1.13.0 (4.3 min, Aqua included); the new public-names testset has 132 checks
- Ambiguity count: 0 (no change)
- Staged but uncommitted: no (all unstaged)

## Cluster status
- release: 2 of 5 complete

## Next chunk
CHUNK-032: sync-docs. Every export and `public` name in an `@docs` block, remove `warnonly = [:missing_docs]`, fix the broken docstring examples and tutorials, update index.md and the README.

## Watch out for
- The newly public `set_H!`…`set_Q!` docstrings link `[`set_Z!`](@ref Siphon.set_Z!)`; check that it resolves in the docs build.
- `docs/src/tutorials/custom_models.md:332` still calls `Siphon.build_dns_loadings(maturities, :λ; λ_init = …)`. Check that the signature matches in CHUNK-032.
