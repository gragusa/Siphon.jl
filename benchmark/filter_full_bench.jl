"""
    filter_full_bench.jl

Benchmarks for full Kalman filter over sequences.
Compares pure vs in-place implementations across different sequence lengths.
"""

using BenchmarkTools
using LinearAlgebra
using Random
using Statistics

include("data_generators.jl")
include("implementations.jl")

# ============================================
# Full filter benchmarks
# ============================================

function benchmark_filter_scalar(; lengths=[100, 1000, 10000])
    println("\n" * "="^60)
    println("SCALAR FULL FILTER BENCHMARKS")
    println("="^60)

    parms = scalar_local_level()
    rng = benchmark_rng()

    for n in lengths
        y = simulate_scalar(n; rng=rng)

        println("\n--- n = $n ---")

        print("Pure:     ")
        t_pure = @benchmark filter_scalar_pure($y, $(parms.Z), $(parms.H),
                                                $(parms.T), $(parms.R),
                                                $(parms.Q), $(parms.a1), $(parms.P1))
        println("$(median(t_pure.times) / 1e6) ms (median)")

        print("In-place: ")
        t_inplace = @benchmark filter_scalar_inplace($y, $(parms.Z), $(parms.H),
                                                      $(parms.T), $(parms.R),
                                                      $(parms.Q), $(parms.a1), $(parms.P1))
        println("$(median(t_inplace.times) / 1e6) ms (median)")

        speedup = median(t_pure.times) / median(t_inplace.times)
        println("Speedup: $(round(speedup, digits=2))x")
    end
end

function benchmark_filter_small(; lengths=[100, 1000, 10000])
    println("\n" * "="^60)
    println("SMALL MATRIX (3x3) FULL FILTER BENCHMARKS")
    println("="^60)

    parms = small_matrix()
    rng = benchmark_rng()

    for n in lengths
        y = simulate_small_matrix(n; rng=rng)

        println("\n--- n = $n ---")

        print("Pure:     ")
        t_pure = @benchmark filter_pure($y, $(parms.Z), $(parms.H),
                                         $(parms.T), $(parms.R),
                                         $(parms.Q), $(parms.a1), $(parms.P1))
        println("$(median(t_pure.times) / 1e6) ms (median)")

        print("In-place: ")
        t_inplace = @benchmark filter_inplace($y, $(parms.Z), $(parms.H),
                                               $(parms.T), $(parms.R),
                                               $(parms.Q), $(parms.a1), $(parms.P1))
        println("$(median(t_inplace.times) / 1e6) ms (median)")

        speedup = median(t_pure.times) / median(t_inplace.times)
        println("Speedup: $(round(speedup, digits=2))x")

        # Memory comparison
        alloc_pure = @allocated filter_pure(y, parms.Z, parms.H, parms.T, parms.R, parms.Q, parms.a1, parms.P1)
        alloc_inplace = @allocated filter_inplace(y, parms.Z, parms.H, parms.T, parms.R, parms.Q, parms.a1, parms.P1)
        println("Memory - Pure: $(alloc_pure ÷ 1024) KB, In-place: $(alloc_inplace ÷ 1024) KB")
    end
end

function benchmark_filter_medium(; lengths=[100, 1000, 10000])
    println("\n" * "="^60)
    println("MEDIUM MATRIX (10x10, 3 obs) FULL FILTER BENCHMARKS")
    println("="^60)

    rng = benchmark_rng()

    for n in lengths
        y, parms = simulate_medium_matrix(n; rng=rng)

        println("\n--- n = $n ---")

        print("Pure:     ")
        t_pure = @benchmark filter_pure($y, $(parms.Z), $(parms.H),
                                         $(parms.T), $(parms.R),
                                         $(parms.Q), $(parms.a1), $(parms.P1))
        println("$(median(t_pure.times) / 1e6) ms (median)")

        print("In-place: ")
        t_inplace = @benchmark filter_inplace($y, $(parms.Z), $(parms.H),
                                               $(parms.T), $(parms.R),
                                               $(parms.Q), $(parms.a1), $(parms.P1))
        println("$(median(t_inplace.times) / 1e6) ms (median)")

        speedup = median(t_pure.times) / median(t_inplace.times)
        println("Speedup: $(round(speedup, digits=2))x")

        # Memory comparison
        alloc_pure = @allocated filter_pure(y, parms.Z, parms.H, parms.T, parms.R, parms.Q, parms.a1, parms.P1)
        alloc_inplace = @allocated filter_inplace(y, parms.Z, parms.H, parms.T, parms.R, parms.Q, parms.a1, parms.P1)
        println("Memory - Pure: $(alloc_pure ÷ 1024) KB, In-place: $(alloc_inplace ÷ 1024) KB")
    end
end

function benchmark_comparison_table(; lengths=[100, 1000, 10000])
    println("\n" * "="^60)
    println("COMPARISON SUMMARY TABLE")
    println("="^60)

    results = []

    # Scalar
    parms_scalar = scalar_local_level()
    for n in lengths
        rng = benchmark_rng(n)
        y = simulate_scalar(n; rng=rng)

        # Warm up
        filter_scalar_pure(y, parms_scalar.Z, parms_scalar.H, parms_scalar.T, parms_scalar.R, parms_scalar.Q, parms_scalar.a1, parms_scalar.P1)
        filter_scalar_inplace(y, parms_scalar.Z, parms_scalar.H, parms_scalar.T, parms_scalar.R, parms_scalar.Q, parms_scalar.a1, parms_scalar.P1)

        t_pure = @belapsed filter_scalar_pure($y, $(parms_scalar.Z), $(parms_scalar.H),
                                               $(parms_scalar.T), $(parms_scalar.R),
                                               $(parms_scalar.Q), $(parms_scalar.a1), $(parms_scalar.P1))
        t_inplace = @belapsed filter_scalar_inplace($y, $(parms_scalar.Z), $(parms_scalar.H),
                                                     $(parms_scalar.T), $(parms_scalar.R),
                                                     $(parms_scalar.Q), $(parms_scalar.a1), $(parms_scalar.P1))
        push!(results, ("scalar", 1, n, t_pure * 1e9, t_inplace * 1e9))  # Convert to ns
    end

    # Small matrix
    parms_small = small_matrix()
    for n in lengths
        rng = benchmark_rng(1000 + n)
        y = simulate_small_matrix(n; rng=rng)

        # Warm up
        filter_pure(y, parms_small.Z, parms_small.H, parms_small.T, parms_small.R, parms_small.Q, parms_small.a1, parms_small.P1)
        filter_inplace(y, parms_small.Z, parms_small.H, parms_small.T, parms_small.R, parms_small.Q, parms_small.a1, parms_small.P1)

        t_pure = @belapsed filter_pure($y, $(parms_small.Z), $(parms_small.H),
                                        $(parms_small.T), $(parms_small.R),
                                        $(parms_small.Q), $(parms_small.a1), $(parms_small.P1))
        t_inplace = @belapsed filter_inplace($y, $(parms_small.Z), $(parms_small.H),
                                              $(parms_small.T), $(parms_small.R),
                                              $(parms_small.Q), $(parms_small.a1), $(parms_small.P1))
        push!(results, ("small", 3, n, t_pure * 1e9, t_inplace * 1e9))
    end

    # Medium matrix - generate parameters once with fixed seed
    rng_med = benchmark_rng(123)
    parms_med = medium_matrix(; rng=rng_med)
    for n in lengths
        rng_y = benchmark_rng(2000 + n)
        y = randn(rng_y, 3, n)  # Just use random data with fixed parameters

        # Warm up
        filter_pure(y, parms_med.Z, parms_med.H, parms_med.T, parms_med.R, parms_med.Q, parms_med.a1, parms_med.P1)
        filter_inplace(y, parms_med.Z, parms_med.H, parms_med.T, parms_med.R, parms_med.Q, parms_med.a1, parms_med.P1)

        t_pure = @belapsed filter_pure($y, $(parms_med.Z), $(parms_med.H),
                                        $(parms_med.T), $(parms_med.R),
                                        $(parms_med.Q), $(parms_med.a1), $(parms_med.P1))
        t_inplace = @belapsed filter_inplace($y, $(parms_med.Z), $(parms_med.H),
                                              $(parms_med.T), $(parms_med.R),
                                              $(parms_med.Q), $(parms_med.a1), $(parms_med.P1))
        push!(results, ("medium", 10, n, t_pure * 1e9, t_inplace * 1e9))
    end

    # Print table
    println("\n| Model  | States | n     | Pure (ms) | In-place (ms) | Speedup |")
    println("|--------|--------|-------|-----------|---------------|---------|")
    for (model, m, n, t_pure, t_inplace) in results
        speedup = t_pure / t_inplace
        println("| $(rpad(model, 6)) | $(lpad(m, 6)) | $(lpad(n, 5)) | $(lpad(round(t_pure/1e6, digits=3), 9)) | $(lpad(round(t_inplace/1e6, digits=3), 13)) | $(lpad(round(speedup, digits=2), 7)) |")
    end

    return results
end

function run_filter_benchmarks()
    println("\n" * "#"^60)
    println("# KALMAN FILTER FULL SEQUENCE BENCHMARKS")
    println("#"^60)

    benchmark_filter_scalar()
    benchmark_filter_small()
    benchmark_filter_medium()

    results = benchmark_comparison_table()

    println("\n" * "="^60)
    println("FULL FILTER BENCHMARKS COMPLETE")
    println("="^60)

    return results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_filter_benchmarks()
end
