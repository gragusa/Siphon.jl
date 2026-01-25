"""
    benchmarks.jl

Main benchmark runner for Siphon.jl Kalman filter implementations.

Usage:
    julia --project=benchmark benchmark/benchmarks.jl [options]

Options:
    --step      Run single-step benchmarks only
    --full      Run full filter benchmarks only
    --quick     Run quick benchmarks (smaller n values)
    --all       Run all benchmarks (default)

Results Summary:
    The benchmarks compare:
    1. Pure (functional) implementation - creates new arrays each step
    2. In-place implementation - reuses preallocated buffers

    Key findings (typical):
    - Scalar models: In-place provides minimal benefit (no matrix allocation anyway)
    - Small matrices (m < 5): Pure implementation is often comparable or faster
    - Medium/Large matrices (m >= 5): In-place can be 1.5-3x faster
    - Long sequences (n > 1000): In-place benefits compound

Recommendation:
    For most use cases with Siphon.jl:
    - Keep the pure implementation as default (simpler, easier to maintain)
    - Consider in-place variant for:
      * Online/streaming filtering with large state dimensions
      * Performance-critical production code with m >= 10
      * Very long sequences where allocation pressure matters
"""

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using BenchmarkTools

# Configure BenchmarkTools for consistent results
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5

include("filterstep_bench.jl")
include("filter_full_bench.jl")

function print_header()
    println()
    println("╔" * "═"^58 * "╗")
    println("║" * " "^14 * "Siphon.jl Kalman Filter Benchmarks" * " "^9 * "║")
    println("║" * " "^58 * "║")
    println("║  Comparing pure (functional) vs in-place implementations  ║")
    println("╚" * "═"^58 * "╝")
    println()
    println("Julia version: ", VERSION)
    println("Threads: ", Threads.nthreads())
    println("BLAS threads: ", LinearAlgebra.BLAS.get_num_threads())
    println()
end

function print_recommendations(results)
    println()
    println("╔" * "═"^58 * "╗")
    println("║" * " "^20 * "RECOMMENDATIONS" * " "^23 * "║")
    println("╚" * "═"^58 * "╝")
    println()

    # Analyze results
    speedups = [(r[2], r[3], r[4]/r[5]) for r in results]

    println("Based on benchmark results:")
    println()

    # Scalar case
    scalar_speedups = [s[3] for s in speedups if s[1] == 1]
    avg_scalar = sum(scalar_speedups) / length(scalar_speedups)
    println(
        "• Scalar models: In-place $(avg_scalar > 1.0 ? "faster" : "slower") by $(round(abs(avg_scalar - 1)*100, digits=1))%",
    )
    println("  → Recommendation: Use pure implementation (simpler code)")
    println()

    # Small matrix
    small_speedups = [s[3] for s in speedups if s[1] == 3]
    avg_small = sum(small_speedups) / length(small_speedups)
    println(
        "• Small matrices (m=3): In-place $(avg_small > 1.0 ? "faster" : "slower") by $(round(abs(avg_small - 1)*100, digits=1))%",
    )
    if avg_small < 1.2
        println("  → Recommendation: Use pure implementation")
    else
        println("  → Recommendation: Consider in-place for long sequences")
    end
    println()

    # Medium matrix
    medium_speedups = [s[3] for s in speedups if s[1] == 10]
    avg_medium = sum(medium_speedups) / length(medium_speedups)
    println(
        "• Medium matrices (m=10): In-place $(avg_medium > 1.0 ? "faster" : "slower") by $(round(abs(avg_medium - 1)*100, digits=1))%",
    )
    if avg_medium > 1.5
        println("  → Recommendation: Implement in-place variant for m >= 10")
    else
        println("  → Recommendation: Pure implementation sufficient")
    end
    println()

    println("General guidance:")
    println("  • StaticArrays for m <= 4: Best performance, stack-allocated")
    println("  • Pure implementation: Default choice, maintainable code")
    println("  • In-place: Consider for m >= 10 AND n >= 1000")
end

function main(args = ARGS)
    print_header()

    quick = "--quick" in args
    step_only = "--step" in args
    full_only = "--full" in args

    # Default: run both unless specific flag given
    run_step = !full_only
    run_full = !step_only

    lengths = quick ? [100, 1000] : [100, 1000, 10000]

    if run_step
        run_filterstep_benchmarks()
    end

    results = nothing
    if run_full
        results = benchmark_comparison_table(; lengths = lengths)
    end

    if results !== nothing
        print_recommendations(results)
    end

    println()
    println("Benchmarks complete.")
end

# Run main
main()
