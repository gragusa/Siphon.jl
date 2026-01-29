using Test
using Aqua
using Siphon

@testset "Aqua.jl" begin
    Aqua.test_all(
        Siphon;
        ambiguities = true,
        unbound_args = true,
        undefined_exports = true,
        stale_deps = true,
        deps_compat = true,
        persistent_tasks = false  # Disabled per user request
    )
end
