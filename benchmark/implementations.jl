"""
    implementations.jl

In-place Kalman filter implementation for benchmark comparison.
This implements filterstep! that reuses preallocated buffers.
"""

using LinearAlgebra

# ============================================
# Cache for in-place operations
# ============================================

"""
Preallocated workspace for in-place Kalman filter operations.
"""
struct FilterCache{T}
    # Intermediate vectors
    v::Vector{T}      # innovation (p)
    Za::Vector{T}     # Z*a (p)
    Ta::Vector{T}     # T*a (m)
    Kv::Vector{T}     # K*v (m)

    # Intermediate matrices
    PZt::Matrix{T}    # P*Z' (m x p)
    ZPZt::Matrix{T}   # Z*P*Z' (p x p)
    F::Matrix{T}      # F = Z*P*Z' + H (p x p)
    Finv::Matrix{T}   # F⁻¹ (p x p)
    K::Matrix{T}      # Kalman gain T*P*Z'*F⁻¹ (m x p)
    L::Matrix{T}      # T - K*Z (m x m)
    TPLt::Matrix{T}   # T*P*L' (m x m)
    RQRt::Matrix{T}   # R*Q*R' (m x m)
    TMP1::Matrix{T}   # temporary (m x p)
    TMP2::Matrix{T}   # temporary (m x m)
    RQ::Matrix{T}     # R*Q (m x r)
end

"""
Create a FilterCache with appropriate dimensions.
"""
function FilterCache(m::Int, p::Int, r::Int, ::Type{T} = Float64) where {T}
    FilterCache{T}(
        Vector{T}(undef, p),      # v
        Vector{T}(undef, p),      # Za
        Vector{T}(undef, m),      # Ta
        Vector{T}(undef, m),      # Kv
        Matrix{T}(undef, m, p),   # PZt
        Matrix{T}(undef, p, p),   # ZPZt
        Matrix{T}(undef, p, p),   # F
        Matrix{T}(undef, p, p),   # Finv
        Matrix{T}(undef, m, p),   # K
        Matrix{T}(undef, m, m),   # L
        Matrix{T}(undef, m, m),   # TPLt
        Matrix{T}(undef, m, m),   # RQRt
        Matrix{T}(undef, m, p),   # TMP1
        Matrix{T}(undef, m, m),   # TMP2
        Matrix{T}(undef, m, r)   # RQ
    )
end

# ============================================
# Pure implementation (for comparison)
# ============================================

"""
Pure functional Kalman filter step.
Returns new state and covariance without modifying inputs.
"""
function filterstep_pure(
        a::AbstractVector,
        P::AbstractMatrix,
        Z::AbstractMatrix,
        H::AbstractMatrix,
        T::AbstractMatrix,
        R::AbstractMatrix,
        Q::AbstractMatrix,
        y::AbstractVector
)
    v = y - Z * a
    F = Z * P * Z' + H
    Finv = inv(F)
    K = T * P * Z' * Finv
    L = T - K * Z
    a_new = T * a + K * v
    P_new = T * P * L' + R * Q * R'
    return (a_new, P_new, v, F, Finv)
end

# ============================================
# In-place implementation
# ============================================

"""
In-place Kalman filter step using preallocated cache.
Modifies a_out and P_out in place.
"""
function filterstep_inplace!(
        a_out::AbstractVector,
        P_out::AbstractMatrix,
        a::AbstractVector,
        P::AbstractMatrix,
        Z::AbstractMatrix,
        H::AbstractMatrix,
        T::AbstractMatrix,
        R::AbstractMatrix,
        Q::AbstractMatrix,
        y::AbstractVector,
        cache::FilterCache
)
    m = length(a)
    p = length(y)

    # v = y - Z*a
    mul!(cache.Za, Z, a)
    @inbounds @simd for i in 1:p
        cache.v[i] = y[i] - cache.Za[i]
    end

    # PZ' -> PZt
    mul!(cache.PZt, P, Z')

    # Z*P*Z' -> ZPZt
    mul!(cache.ZPZt, Z, cache.PZt)

    # F = Z*P*Z' + H
    @inbounds @simd for i in eachindex(cache.F)
        cache.F[i] = cache.ZPZt[i] + H[i]
    end

    # Finv = inv(F)
    copyto!(cache.Finv, inv(cache.F))

    # K = T*P*Z'*Finv
    mul!(cache.TMP1, T, cache.PZt)
    mul!(cache.K, cache.TMP1, cache.Finv)

    # L = T - K*Z
    mul!(cache.L, cache.K, Z)
    @inbounds @simd for i in eachindex(cache.L)
        cache.L[i] = T[i] - cache.L[i]
    end

    # a_out = T*a + K*v
    mul!(cache.Ta, T, a)
    mul!(cache.Kv, cache.K, cache.v)
    @inbounds @simd for i in 1:m
        a_out[i] = cache.Ta[i] + cache.Kv[i]
    end

    # P_out = T*P*L' + R*Q*R'
    mul!(cache.TMP2, T, P)
    mul!(cache.TPLt, cache.TMP2, cache.L')
    mul!(cache.RQ, R, Q)
    mul!(cache.RQRt, cache.RQ, R')
    @inbounds @simd for i in eachindex(P_out)
        P_out[i] = cache.TPLt[i] + cache.RQRt[i]
    end

    return (cache.v, cache.F, cache.Finv)
end

# ============================================
# Full filter implementations
# ============================================

"""
Run full Kalman filter using pure implementation.
Returns vectors of filtered states and covariances.
"""
function filter_pure(y::AbstractMatrix, Z, H, T, R, Q, a1, P1)
    p, n = size(y)
    m = length(a1)

    # Storage
    a_filt = Vector{Vector{Float64}}(undef, n + 1)
    P_filt = Vector{Matrix{Float64}}(undef, n + 1)

    a_filt[1] = copy(a1)
    P_filt[1] = copy(P1)

    loglik = 0.0

    for t in 1:n
        a, P, v, F, Finv = filterstep_pure(a_filt[t], P_filt[t], Z, H, T, R, Q, y[:, t])
        a_filt[t + 1] = a
        P_filt[t + 1] = P
        loglik += -0.5 * (p * log(2π) + logdet(F) + v' * Finv * v)
    end

    return (a_filt, P_filt, loglik)
end

"""
Run full Kalman filter using in-place implementation.
Preallocates all storage upfront.
"""
function filter_inplace(y::AbstractMatrix, Z, H, T, R, Q, a1, P1)
    p, n = size(y)
    m = length(a1)
    r = size(Q, 1)

    # Preallocate storage
    a_filt = Matrix{Float64}(undef, m, n + 1)
    P_filt = Array{Float64}(undef, m, m, n + 1)

    a_filt[:, 1] = a1
    P_filt[:, :, 1] = P1

    # Working buffers for current/next state
    a_curr = Vector{Float64}(undef, m)
    P_curr = Matrix{Float64}(undef, m, m)
    a_next = Vector{Float64}(undef, m)
    P_next = Matrix{Float64}(undef, m, m)

    copy!(a_curr, a1)
    copy!(P_curr, P1)

    # Create cache
    cache = FilterCache(m, p, r, Float64)

    loglik = 0.0

    for t in 1:n
        v, F,
        Finv = filterstep_inplace!(
            a_next,
            P_next,
            a_curr,
            P_curr,
            Z,
            H,
            T,
            R,
            Q,
            view(y, :, t),
            cache
        )
        a_filt[:, t + 1] = a_next
        P_filt[:, :, t + 1] = P_next

        loglik += -0.5 * (p * log(2π) + logdet(F) + v' * Finv * v)

        # Swap buffers
        a_curr, a_next = a_next, a_curr
        P_curr, P_next = P_next, P_curr
    end

    return (a_filt, P_filt, loglik)
end

# ============================================
# Scalar specializations
# ============================================

"""
Pure scalar filter step.
"""
function filterstep_scalar_pure(
        a::Float64,
        P::Float64,
        Z::Float64,
        H::Float64,
        T::Float64,
        R::Float64,
        Q::Float64,
        y::Float64
)
    v = y - Z * a
    F = Z * P * Z + H
    Finv = 1.0 / F
    K = T * P * Z * Finv
    L = T - K * Z
    a_new = T * a + K * v
    P_new = T * P * L + R * Q * R
    return (a_new, P_new, v, F, Finv)
end

"""
Run full scalar filter (pure).
"""
function filter_scalar_pure(y::AbstractVector, Z, H, T, R, Q, a1, P1)
    n = length(y)

    a_filt = Vector{Float64}(undef, n + 1)
    P_filt = Vector{Float64}(undef, n + 1)

    a_filt[1] = a1
    P_filt[1] = P1

    loglik = 0.0

    for t in 1:n
        a, P, v, F, Finv = filterstep_scalar_pure(a_filt[t], P_filt[t], Z, H, T, R, Q, y[t])
        a_filt[t + 1] = a
        P_filt[t + 1] = P
        loglik += -0.5 * (log(2π) + log(F) + v^2 * Finv)
    end

    return (a_filt, P_filt, loglik)
end

"""
Run full scalar filter (in-place - note: scalar case has minimal benefit).
"""
function filter_scalar_inplace(y::AbstractVector, Z, H, T, R, Q, a1, P1)
    n = length(y)

    a_filt = Vector{Float64}(undef, n + 1)
    P_filt = Vector{Float64}(undef, n + 1)

    a_filt[1] = a1
    P_filt[1] = P1

    a_curr = a1
    P_curr = P1

    loglik = 0.0

    @inbounds for t in 1:n
        v = y[t] - Z * a_curr
        F = Z * P_curr * Z + H
        Finv = 1.0 / F
        K = T * P_curr * Z * Finv
        L = T - K * Z

        a_curr = T * a_curr + K * v
        P_curr = T * P_curr * L + R * Q * R

        a_filt[t + 1] = a_curr
        P_filt[t + 1] = P_curr

        loglik += -0.5 * (log(2π) + log(F) + v^2 * Finv)
    end

    return (a_filt, P_filt, loglik)
end
