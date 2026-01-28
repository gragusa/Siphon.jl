"""
    filterstep_bench.jl

Benchmarks for single Kalman filter step operations.
Compares pure vs in-place implementations.
"""

using BenchmarkTools
using LinearAlgebra
using StaticArrays
using Random

include("data_generators.jl")
include("implementations.jl")

# ============================================
# Single-step benchmarks
# ============================================

function benchmark_filterstep_scalar()
    println("\n" * "="^60)
    println("SCALAR FILTER STEP BENCHMARKS")
    println("="^60)

    parms = scalar_local_level()

    # Non-diffuse state
    a = 100.0
    P = 1000.0

    y = 105.0

    println("\n--- Pure scalar step ---")
    @btime filterstep_scalar_pure(
        $a,
        $P,
        $(parms.Z),
        $(parms.H),
        $(parms.T),
        $(parms.R),
        $(parms.Q),
        $y
    )
end

function benchmark_filterstep_small()
    println("\n" * "="^60)
    println("SMALL MATRIX (3x3) FILTER STEP BENCHMARKS")
    println("="^60)

    parms = small_matrix()
    m = 3

    a = randn(m)
    P = Matrix(1.0I, m, m) * 100

    y = randn(1)

    println("\n--- Pure matrix step ---")
    @btime filterstep_pure(
        $a,
        $P,
        $(parms.Z),
        $(parms.H),
        $(parms.T),
        $(parms.R),
        $(parms.Q),
        $y
    )

    # In-place
    a_out = similar(a)
    P_out = similar(P)
    cache = FilterCache(m, 1, m)

    println("\n--- In-place matrix step ---")
    @btime filterstep_inplace!(
        $a_out,
        $P_out,
        $a,
        $P,
        $(parms.Z),
        $(parms.H),
        $(parms.T),
        $(parms.R),
        $(parms.Q),
        $y,
        $cache
    )
end

function benchmark_filterstep_medium()
    println("\n" * "="^60)
    println("MEDIUM MATRIX (10x10) FILTER STEP BENCHMARKS")
    println("="^60)

    rng = benchmark_rng()
    parms = medium_matrix(; rng = rng)
    m, p, r = 10, 3, 5

    a = randn(m)
    P = Matrix(1.0I, m, m) * 100

    y = randn(p)

    println("\n--- Pure matrix step ---")
    @btime filterstep_pure(
        $a,
        $P,
        $(parms.Z),
        $(parms.H),
        $(parms.T),
        $(parms.R),
        $(parms.Q),
        $y
    )

    # In-place
    a_out = similar(a)
    P_out = similar(P)
    cache = FilterCache(m, p, r)

    println("\n--- In-place matrix step ---")
    @btime filterstep_inplace!(
        $a_out,
        $P_out,
        $a,
        $P,
        $(parms.Z),
        $(parms.H),
        $(parms.T),
        $(parms.R),
        $(parms.Q),
        $y,
        $cache
    )
end

function benchmark_filterstep_static()
    println("\n" * "="^60)
    println("STATIC ARRAYS (3x3) FILTER STEP BENCHMARKS")
    println("="^60)

    parms = small_static()

    a = SVector{3}(randn(3))
    P = SMatrix{3, 3}(Matrix(1.0I, 3, 3) * 100)

    y = SVector{1}(randn(1))

    # StaticArrays version (uses regular pure implementation but with static types)
    println("\n--- StaticArrays step ---")
    @btime begin
        v = $y - $(parms.Z) * $a
        F = $(parms.Z) * $P * $(parms.Z)' + $(parms.H)
        Finv = inv(F)
        K = $(parms.T) * $P * $(parms.Z)' * Finv
        L = $(parms.T) - K * $(parms.Z)
        a_new = $(parms.T) * $a + K * v
        P_new = $(parms.T) * $P * L' + $(parms.R) * $(parms.Q) * $(parms.R)'
        (a_new, P_new, v, F, Finv)
    end
end

function run_filterstep_benchmarks()
    println("\n" * "#"^60)
    println("# KALMAN FILTER SINGLE-STEP BENCHMARKS")
    println("#"^60)

    benchmark_filterstep_scalar()
    benchmark_filterstep_small()
    benchmark_filterstep_medium()
    benchmark_filterstep_static()

    println("\n" * "="^60)
    println("SINGLE-STEP BENCHMARKS COMPLETE")
    println("="^60)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_filterstep_benchmarks()
end
