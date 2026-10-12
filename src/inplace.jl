"""
    inplace.jl

In-place Kalman filter, smoother, and EM algorithm for large state-space models.

This module provides zero-allocation implementations designed for:
- Dynamic factor models with many observables (p >> m)
- EM algorithm requiring repeated filter/smoother iterations
- Large-scale applications where allocation overhead matters

# Contents

## KalmanWorkspace (lines ~25-340)
Pre-allocated workspace for in-place Kalman filter and smoother operations.
- `KalmanWorkspace{T}` struct with all storage
- `kalman_filter!` - in-place filter with missing data handling
- `kalman_smoother!` - in-place RTS smoother with cross-lag covariances
- `kalman_filter_and_smooth!` - combined operation

## EMWorkspace (lines ~890-1400)
EM algorithm infrastructure:
- `EMWorkspace{T}` - sufficient statistics storage
- `compute_sufficient_stats!` - E-step computation
- `update_Z!`, `update_H!`, `update_T!`, `update_Q!` - M-step updates
- `em_estimate!` - main EM algorithm

## StateSpaceModel (lines ~1400-2600)
High-level model interface:
- `StateSpaceModel{T}` - mutable model container
- `fit!(MLE(), model, y)` - MLE via numerical optimization
- `fit!(EM(), model, y)` - EM algorithm
- Accessors: `parameters`, `filtered_states`, `smoothed_states`, etc.
- `forecast(model, h)` - h-step ahead forecasting

## DynamicFactorModel (lines ~2600-end)
Specialized dynamic factor model:
- `DynamicFactorModel{T}` - DFM with optional AR errors
- `DynamicFactorModelSpec` - model specification
- EM estimation with AR error dynamics
- Factor extraction and forecasting
"""

# Note: LinearAlgebra is already imported by the main module

# DSL types (SSMSpec, SSMParameter, etc.) are available from the DSL submodule
# which is loaded before this file

# ============================================
# KalmanWorkspace - Pre-allocated storage
# ============================================

"""
    KalmanWorkspace{T<:Real}

Pre-allocated workspace for in-place Kalman filter and smoother operations.

Stores all matrices, vectors, and scratch space needed for filtering and smoothing,
enabling zero-allocation iterations after initial construction.

# Fields

## Dimensions
- `obs_dim`: Number of observables (p)
- `state_dim`: Number of states (m)
- `shock_dim`: Number of state shocks (r)
- `n_obs`: Number of time periods (n)

## Parameters (owned copies)
- `Z`: Observation matrix (p × m)
- `H`: Observation noise covariance (p × p)
- `T`: State transition matrix (m × m)
- `R`: State noise selection matrix (m × r)
- `Q`: State noise covariance (r × r)
- `a1`: Initial state mean (m)
- `P1`: Initial state covariance (m × m)

## Precomputed
- `RQR`: R * Q * R' (m × m), updated when R or Q change

## Filter storage
- `at`: Predicted states E[αₜ|y₁:ₜ₋₁] (m × n)
- `Pt`: Predicted covariances (m × m × n)
- `att`: Filtered states E[αₜ|y₁:ₜ] (m × n)
- `Ptt`: Filtered covariances (m × m × n)
- `vt`: Innovations yₜ - Z*aₜ (p × n)
- `Ft`: Innovation covariances (p × p × n)
- `Ft_L`: Lower-triangular Cholesky factor of Ft (p × p × n). On the
  rank-revealing path it instead holds `G`, the pseudo-inverse square root with
  `Fₜ⁺ = G G'`, in its leading `pₜ × Ft_rank[t]` block
- `Ft_rank`: Rank of Ft per period (n). On the Cholesky path this is the number
  of observed rows pₜ; on the rank-revealing path it is the numerical rank
- `Kt`: Kalman gains (m × p × n)
- `missing_mask`: BitVector, true where period t has no observed row at all
- `observed_mask`: BitMatrix (p × n), true where yₜ[i] is observed
- `n_observed`: Vector{Int} (n), the number of observed rows pₜ in each period
- `obs_rows`: Matrix{Int} (p × n), the observed row indices of each period in the
  first `n_observed[t]` entries of column t

When a period is partially observed the filter works on the reduced system formed
by those rows, so `vt`, `Ft`, `Ft_L` and `Kt` carry pₜ-sized blocks in their
leading rows and columns for that period; entries beyond pₜ are `NaN` for `vt` and
`Ft` and zero for `Ft_L` and `Kt`. Use `observed_rows(ws, t)` to recover which
original row each leading position corresponds to. The accessors `prediction_errors`,
`prediction_errors_cov` and `kalman_gains` return copies in series positions.

## Smoother storage
- `αs`: Smoothed states E[αₜ|y₁:ₙ] (m × n)
- `Vs`: Smoothed covariances (m × m × n)
- `Pcross`: Cross-lag covariances Cov[αₜ₊₁,αₜ|y₁:ₙ] (m × m × n-1)

## Scratch space (temporaries for BLAS operations)
- Various tmp_* matrices and vectors

## Scalars
- `loglik`: Log-likelihood value
- `n_obs_valid`: Count of non-missing observations
"""
mutable struct KalmanWorkspace{T <: Real} <: AbstractFilterWorkspace
    # Dimensions
    obs_dim::Int      # p
    state_dim::Int    # m
    shock_dim::Int    # r
    n_times::Int      # n

    # Parameters (owned copies)
    Z::Matrix{T}      # p × m
    H::Matrix{T}      # p × p
    Tmat::Matrix{T}   # m × m (named Tmat to avoid conflict with type T)
    R::Matrix{T}      # m × r
    Q::Matrix{T}      # r × r
    a1::Vector{T}     # m
    P1::Matrix{T}     # m × m

    # Precomputed
    RQR::Matrix{T}    # m × m: R * Q * R'

    # Filter storage
    at::Matrix{T}         # m × n: predicted states
    Pt::Array{T, 3}        # m × m × n: predicted covariances
    att::Matrix{T}        # m × n: filtered states
    Ptt::Array{T, 3}       # m × m × n: filtered covariances
    vt::Matrix{T}         # p × n: innovations
    Ft::Array{T, 3}        # p × p × n: innovation covariances
    Ft_L::Array{T, 3}      # p × p × n: Cholesky factors (lower tri)
    Kt::Array{T, 3}        # m × p × n: Kalman gains
    missing_mask::BitVector
    observed_mask::BitMatrix  # p × n: true where y[i,t] is observed
    n_observed::Vector{Int}   # n: number of observed rows per period
    obs_rows::Matrix{Int}     # p × n: observed row indices, first n_observed[t] entries valid

    # Smoother storage
    αs::Matrix{T}         # m × n: smoothed states
    Vs::Array{T, 3}        # m × m × n: smoothed covariances
    Pcross::Array{T, 3}    # m × m × (n-1): cross-lag covariances

    # Scratch space
    tmp_mm1::Matrix{T}    # m × m
    tmp_mm2::Matrix{T}    # m × m
    tmp_mm3::Matrix{T}    # m × m
    tmp_pp1::Matrix{T}    # p × p
    tmp_pp2::Matrix{T}    # p × p
    tmp_mp::Matrix{T}     # m × p
    tmp_pm::Matrix{T}     # p × m
    tmp_pm2::Matrix{T}    # p × m
    tmp_pm3::Matrix{T}    # p × m
    tmp_mr::Matrix{T}     # m × r
    tmp_m1::Vector{T}     # m
    tmp_m2::Vector{T}     # m
    tmp_p1::Vector{T}     # p
    tmp_p2::Vector{T}     # p

    # Smoother scratch
    r_smooth::Vector{T}   # m: smoother recursion vector
    N_smooth::Matrix{T}   # m × m: smoother recursion matrix
    L_smooth::Matrix{T}   # m × m: T - K*Z
    J_smooth::Matrix{T}   # m × m: for cross-cov computation

    # Scalars
    loglik::T
    n_obs_valid::Int
    n_obs_rows::Int

    # Declared block width of a companion transition, or 0 for a general `Tmat`.
    # Set only by `set_companion_structure!`; never inferred from `Tmat`.
    companion_block::Int

    # Rank-revealing measurement update. Set only by `set_rank_revealing!`.
    rank_revealing::Bool
    rank_rtol::T
    support_rtol::T
    Ft_rank::Vector{Int}   # n: rank of Fₜ used in the measurement update

    # Run status: _NOT_RUN before the first filter run and while a run is in progress,
    # 0 after a successful one, and t > 0 when the last run stopped in period t on a
    # non-positive-definite F.
    status::Int
    smoothed::Bool        # the smoother has run on the current filter output
    smoothed_cov::Bool    # ... and filled Vs
    crosscov::Bool        # ... and filled Pcross
end

"""
    KalmanWorkspace{T}(p::Int, m::Int, r::Int, n::Int) where T

Create workspace with specified dimensions. Parameters must be set via `set_params!`.

# Arguments
- `p`: Number of observables
- `m`: Number of states
- `r`: Number of state shocks
- `n`: Number of time periods
"""
function KalmanWorkspace{T}(p::Int, m::Int, r::Int, n::Int) where {T <: Real}
    KalmanWorkspace{T}(
        # Dimensions
        p,
        m,
        r,
        n,

        # Parameters (zero until set_params! is called)
        zeros(T, p, m),      # Z
        zeros(T, p, p),      # H
        zeros(T, m, m),      # Tmat
        zeros(T, m, r),      # R
        zeros(T, r, r),      # Q
        zeros(T, m),         # a1
        zeros(T, m, m),      # P1

        # Precomputed
        zeros(T, m, m),      # RQR

        # Filter storage
        zeros(T, m, n),      # at
        zeros(T, m, m, n),  # Pt
        zeros(T, m, n),      # att
        zeros(T, m, m, n),  # Ptt
        zeros(T, p, n),      # vt
        zeros(T, p, p, n),  # Ft
        zeros(T, p, p, n),  # Ft_L
        zeros(T, m, p, n),  # Kt
        falses(n),         # missing_mask
        falses(p, n),      # observed_mask
        zeros(Int, n),       # n_observed
        zeros(Int, p, n),    # obs_rows

        # Smoother storage
        zeros(T, m, n),      # αs
        zeros(T, m, m, n),  # Vs
        zeros(T, m, m, max(n-1, 1)),  # Pcross

        # Scratch space
        zeros(T, m, m),      # tmp_mm1
        zeros(T, m, m),      # tmp_mm2
        zeros(T, m, m),      # tmp_mm3
        zeros(T, p, p),      # tmp_pp1
        zeros(T, p, p),      # tmp_pp2
        zeros(T, m, p),      # tmp_mp
        zeros(T, p, m),      # tmp_pm
        zeros(T, p, m),      # tmp_pm2
        zeros(T, p, m),      # tmp_pm3
        zeros(T, m, r),      # tmp_mr
        zeros(T, m),         # tmp_m1
        zeros(T, m),         # tmp_m2
        zeros(T, p),         # tmp_p1
        zeros(T, p),         # tmp_p2

        # Smoother scratch
        zeros(T, m),         # r_smooth
        zeros(T, m, m),      # N_smooth
        zeros(T, m, m),      # L_smooth
        zeros(T, m, m),      # J_smooth

        # Scalars
        zero(T),                     # loglik
        0,                           # n_obs_valid
        0,                           # n_obs_rows
        0,                           # companion_block

        # Rank-revealing declaration
        false,                       # rank_revealing
        sqrt(eps(T)),                # rank_rtol
        sqrt(eps(T)),                # support_rtol
        zeros(Int, n),               # Ft_rank
        _NOT_RUN,                    # status
        false,                       # smoothed
        false,                       # smoothed_cov
        false                        # crosscov
    )
end

# Convenience constructor inferring T from inputs
KalmanWorkspace(p::Int, m::Int, r::Int, n::Int) = KalmanWorkspace{Float64}(p, m, r, n)

"""
    KalmanWorkspace(Z, H, T, R, Q, a1, P1, n::Int)

Create workspace from parameter matrices and set parameters.
"""
function KalmanWorkspace(
        Z::AbstractMatrix{T},
        H::AbstractMatrix,
        Tmat::AbstractMatrix,
        R::AbstractMatrix,
        Q::AbstractMatrix,
        a1::AbstractVector,
        P1::AbstractMatrix,
        n::Int
) where {T}
    p, m = size(Z)
    r = size(Q, 1)

    ws = KalmanWorkspace{T}(p, m, r, n)
    set_params!(ws, Z, H, Tmat, R, Q)
    set_initial!(ws, a1, P1)

    return ws
end

"""
    KalmanWorkspace(kf::KFParms, n::Int)

Create workspace from KFParms.
"""
function KalmanWorkspace(kf::KFParms, a1::AbstractVector, P1::AbstractMatrix, n::Int)
    KalmanWorkspace(kf.Z, kf.H, kf.T, kf.R, kf.Q, a1, P1, n)
end

# ============================================
# Accessor methods for KalmanWorkspace
# ============================================

# The shared accessors are in accessors.jl.
system_matrices(ws::KalmanWorkspace) = KFParms(ws.Z, ws.H, ws.Tmat, ws.R, ws.Q)

"""
    KalmanFilterResult(ws::KalmanWorkspace) -> KalmanFilterResult

Copy the filter output of `ws`, with its system matrices, into a `KalmanFilterResult`.
Later runs on `ws` don't change the result, which supports everything a result of
`kalman_filter` does, such as `kalman_smoother(result)`.

Throws an `ArgumentError` if no filter has run on `ws`. After a failed run the result is
a failed result: its log-likelihood is `-Inf` and its state accessors throw.
"""
function KalmanFilterResult(ws::KalmanWorkspace)
    ws.status == _NOT_RUN && _require_output(ws, :KalmanFilterResult)
    p = KFParms(copy(ws.Z), copy(ws.H), copy(ws.Tmat), copy(ws.R), copy(ws.Q))
    return KalmanFilterResult(p, ws.loglik, copy(ws.at), copy(ws.Pt), copy(ws.att),
        copy(ws.Ptt), copy(ws.vt), copy(ws.Ft), copy(ws.Kt), copy(ws.missing_mask),
        copy(ws.observed_mask), ws.status)
end

"""
    observed_mask(ws::KalmanWorkspace) -> BitMatrix

Return the `p × n` mask that is true where an individual observation was present.
`missing_mask(ws)[t]` is true exactly when column `t` of this mask is all false.
"""
observed_mask(ws::KalmanWorkspace) = _require_output(ws, :observed_mask).observed_mask

"""
    set_companion_structure!(ws::KalmanWorkspace, block::Integer) -> ws

Declare that `ws.Tmat` is block-companion with `block × block` blocks, so that the
filter may use the structured form of `T P T'`.

A block-companion transition of state dimension `m = block * lags` has an
arbitrary leading `block × m` row and a shifted identity below it:

    T = [ A₁ A₂ … A_lags ;  I 0 … 0 ]

`T P T'` is then a dense product only against that leading row; the rest is a copy
of `P`'s leading blocks. The filter falls back to the general product when `block`
is 0.

The claim is checked against `Tmat`: the subdiagonal must be exactly the shifted
identity, and a transition that is not of this form raises rather than being
filtered with the wrong recursion. It is rechecked on every filter pass, so a
sampler that redraws only the leading block row keeps the declaration without
restating it, while one that installs a transition of another shape raises.

Pass `block = 0` to return to the general path.

# Example
```julia
set_params!(ws, Z, H, Φ, R, Q)          # Φ companion for an n-variable VAR
set_companion_structure!(ws, n)
```
"""
function set_companion_structure!(ws::KalmanWorkspace{T}, block::Integer) where {T}
    m = ws.state_dim
    if block == 0
        ws.companion_block = 0
        return ws
    end
    block > 0 || throw(ArgumentError("block must be non-negative, got $block"))
    m % block == 0 || throw(ArgumentError(
        "state dimension $m is not a multiple of the block width $block"))

    # Rows below the leading block must be the shift [I 0].
    k = m - block
    @inbounds for j in 1:m, i in 1:k

        want = (j == i) ? one(T) : zero(T)
        ws.Tmat[block + i, j] == want || throw(ArgumentError(
            "transition is not block-companion with block width $block: " *
            "Tmat[$(block + i), $j] is $(ws.Tmat[block + i, j]), expected $want"))
    end

    ws.companion_block = block
    return ws
end

"""
    companion_block(ws::KalmanWorkspace) -> Int

Return the declared companion block width, or 0 when the transition is treated as
a general matrix.
"""
companion_block(ws::KalmanWorkspace) = ws.companion_block

"""
    set_rank_revealing!(ws::KalmanWorkspace, on::Bool; rank_rtol, support_rtol) -> ws

Declare whether the measurement update factors the innovation covariance `Fₜ`
with a rank-revealing method instead of a Cholesky decomposition.

With `on = false`, the default, `Fₜ` is assumed positive definite, and a period
where its Cholesky factorization fails ends the run with a log-likelihood of
`-Inf`. With `on = true` the filter takes a symmetric eigendecomposition of `Fₜ`,
keeps the eigenvalues above `rank_rtol` times the largest, and updates on that
range: a singular `Fₜ` is then a supported case rather than a failure. `Fₜ` is singular whenever the state
covariance is rank-deficient in the observed directions and the observation noise
does not fill them in — a state space driven by fewer shocks than states and
observed without measurement error, for instance.

The observation must still lie on the support. When the component of the
innovation `vₜ` orthogonal to the range of `Fₜ` exceeds `support_rtol` times the
scale of the period's observed quantities, the observation cannot have come from
the model and the filter raises [`InnovationSupportError`](@ref).

The log-likelihood on this path is the density of the singular Gaussian on its
support: `log|Fₜ|` is the sum of the retained eigenvalues' logarithms, and the
Gaussian constant counts the rank rather than the number of observed rows.

Implemented for `KalmanWorkspace`; a `DiffuseKalmanWorkspace` raises when its
base workspace carries the declaration.

# Example
```julia
set_params!(ws, Z, H, Tmat, R, Q)       # rank-deficient R Q R', H = 0
set_rank_revealing!(ws, true)
```

See also: [`rank_revealing`](@ref), [`innovation_ranks`](@ref).
"""
function set_rank_revealing!(
        ws::KalmanWorkspace{T}, on::Bool;
        rank_rtol::Real = sqrt(eps(T)), support_rtol::Real = sqrt(eps(T))) where {T}
    isfinite(rank_rtol) && rank_rtol >= 0 || throw(ArgumentError(
        "rank_rtol must be finite and non-negative, got $rank_rtol"))
    isfinite(support_rtol) && support_rtol >= 0 || throw(ArgumentError(
        "support_rtol must be finite and non-negative, got $support_rtol"))

    ws.rank_revealing = on
    ws.rank_rtol = rank_rtol
    ws.support_rtol = support_rtol
    return ws
end

"""
    rank_revealing(ws::KalmanWorkspace) -> Bool

Return whether the measurement update uses the rank-revealing factorization of
`Fₜ`. See [`set_rank_revealing!`](@ref).
"""
rank_revealing(ws::KalmanWorkspace) = ws.rank_revealing

"""
    innovation_ranks(ws::KalmanWorkspace) -> Vector{Int}

Return the rank of `Fₜ` used in each period's measurement update, as the live
buffer the filter writes rather than a copy.

On the Cholesky path this is the number of observed rows in the period, and 0
where nothing is observed. On the rank-revealing path it is the numerical rank of
`Fₜ`, which is the number of directions the period's observation informs.
"""
innovation_ranks(ws::KalmanWorkspace) = _require_output(ws, :innovation_ranks).Ft_rank

"""
    InnovationSupportError <: Exception

The observation of period `t` is incompatible with the model: its innovation has
a component outside the range of the innovation covariance `Fₜ`.

Raised by `kalman_filter!` on the rank-revealing path; see
[`set_rank_revealing!`](@ref).

# Fields
- `t`: The period whose observation is off-support
- `rank`: Numerical rank of `Fₜ`
- `n_rows`: Number of observed rows in the period
- `residual`: Norm of the innovation's component outside the range of `Fₜ`
- `scale`: Scale of the period's observed quantities
- `tol`: Relative tolerance; the residual is accepted when it is at most
  `tol * scale`
"""
struct InnovationSupportError <: Exception
    t::Int
    rank::Int
    n_rows::Int
    residual::Float64
    scale::Float64
    tol::Float64
end

function Base.showerror(io::IO, e::InnovationSupportError)
    print(io, "InnovationSupportError: at period ", e.t,
        " the innovation has a component of norm ", e.residual,
        " outside the range of the innovation covariance (rank ", e.rank,
        " of ", e.n_rows, " observed rows; tolerance ", e.tol, " × ", e.scale,
        " = ", e.tol * e.scale, ").")
    print(io, " The observation is incompatible with the model: no state path",
        " the model admits can produce it.")
    return nothing
end

"""
    observed_rows(ws::KalmanWorkspace, t::Integer) -> SubArray{Int}

Return the indices of the rows observed in period `t`, in increasing order.

The filter stores `vt`, `Ft`, `Ft_L` and `Kt` for a partially observed period in
the leading rows and columns of their slices, so position `k` of those blocks
corresponds to row `observed_rows(ws, t)[k]` of the full observation vector.
"""
function observed_rows(ws::KalmanWorkspace, t::Integer)
    _require_output(ws, :observed_rows)
    1 ≤ t ≤ ws.n_times || throw(BoundsError(ws.n_observed, t))
    return view(ws.obs_rows, 1:ws.n_observed[t], t)
end

"""
    n_observed(ws::KalmanWorkspace, t::Integer) -> Int

Return the number of rows observed in period `t`.
"""
function n_observed(ws::KalmanWorkspace, t::Integer)
    _require_output(ws, :n_observed)
    1 ≤ t ≤ ws.n_times || throw(BoundsError(ws.n_observed, t))
    return ws.n_observed[t]
end

# ============================================
# Parameter setters
# ============================================

"""
    _update_RQR!(ws::KalmanWorkspace)

Recompute R * Q * R' and store in ws.RQR.
Uses scratch space to avoid allocations.
"""
function _update_RQR!(ws::KalmanWorkspace{T}) where {T}
    # RQR = R * Q * R'
    # Step 1: tmp_mr = R * Q
    mul!(ws.tmp_mr, ws.R, ws.Q)
    # Step 2: RQR = tmp_mr * R'
    mul!(ws.RQR, ws.tmp_mr, ws.R')
    return nothing
end

"""
    set_params!(ws, Z, H, T, R, Q)

Set all system matrices. Copies data into workspace.
Automatically recomputes RQR.
"""
function set_params!(
        ws::KalmanWorkspace,
        Z::AbstractMatrix,
        H::AbstractMatrix,
        Tmat::AbstractMatrix,
        R::AbstractMatrix,
        Q::AbstractMatrix
)
    _check_params(ws, "set_params!"; Z, H, Tmat, R, Q)
    copyto!(ws.Z, Z)
    copyto!(ws.H, H)
    copyto!(ws.Tmat, Tmat)
    copyto!(ws.R, R)
    copyto!(ws.Q, Q)
    _update_RQR!(ws)
    return ws
end

# Check each system matrix that is not `nothing` against the workspace dimensions.
function _check_params(ws, caller; Z = nothing, H = nothing, Tmat = nothing, R = nothing,
        Q = nothing)
    p, m, r = ws.obs_dim, ws.state_dim, ws.shock_dim
    Z === nothing || _check_size(caller, "Z", Z, (p, m))
    H === nothing || _check_size(caller, "H", H, (p, p))
    Tmat === nothing || _check_size(caller, "T", Tmat, (m, m))
    R === nothing || _check_size(caller, "R", R, (m, r))
    Q === nothing || _check_size(caller, "Q", Q, (r, r))
    return nothing
end

function _check_initial(ws, caller, a1, P1)
    m = ws.state_dim
    _check_size(caller, "a1", a1, (m,))
    _check_size(caller, "P1", P1, (m, m))
    return nothing
end

# Throw unless `y` is p × n for a workspace with p observables and n periods.
function _check_obs(ws, caller, y::AbstractMatrix)
    _check_size(caller, "y", y, (ws.obs_dim, ws.n_times))
end

"""
    _check_companion(ws::KalmanWorkspace) -> Int

Return the declared companion block width after confirming `ws.Tmat` still has
the shift structure, or 0 when none is declared.

The filter calls this once per pass rather than trusting a declaration made
before an unknown number of parameter updates. A transition that has stopped
being companion raises here, where the wrong recursion would otherwise be
applied silently; the cost is `O(m²)` against the `O(n m³)` of a filter pass.
"""
function _check_companion(ws::KalmanWorkspace{T}) where {T}
    block = ws.companion_block
    block == 0 && return 0
    m = ws.state_dim
    k = m - block
    @inbounds for j in 1:m, i in 1:k

        want = (j == i) ? one(T) : zero(T)
        ws.Tmat[block + i, j] == want || throw(ArgumentError(
            "workspace declares a companion transition of block width $block, " *
            "but Tmat[$(block + i), $j] is $(ws.Tmat[block + i, j]), expected " *
            "$want. Clear the declaration with set_companion_structure!(ws, 0) " *
            "before installing a transition that is not block-companion."))
    end
    return block
end

"""
    set_initial!(ws, a1, P1)

Set initial state mean and covariance.
"""
function set_initial!(ws::KalmanWorkspace, a1::AbstractVector, P1::AbstractMatrix)
    _check_initial(ws, "set_initial!", a1, P1)
    copyto!(ws.a1, a1)
    copyto!(ws.P1, P1)
    return ws
end

"""
    update_params!(ws; Z=nothing, H=nothing, T=nothing, R=nothing, Q=nothing)

Selectively update parameters. Only non-nothing arguments are updated.
Recomputes RQR if R or Q change.
"""
function update_params!(
        ws::KalmanWorkspace;
        Z::Union{Nothing, AbstractMatrix} = nothing,
        H::Union{Nothing, AbstractMatrix} = nothing,
        T::Union{Nothing, AbstractMatrix} = nothing,
        R::Union{Nothing, AbstractMatrix} = nothing,
        Q::Union{Nothing, AbstractMatrix} = nothing
)
    _check_params(ws, "update_params!"; Z, H, Tmat = T, R, Q)
    if Z !== nothing
        copyto!(ws.Z, Z)
    end
    if H !== nothing
        copyto!(ws.H, H)
    end
    if T !== nothing
        copyto!(ws.Tmat, T)
    end

    update_RQR = false
    if R !== nothing
        copyto!(ws.R, R)
        update_RQR = true
    end
    if Q !== nothing
        copyto!(ws.Q, Q)
        update_RQR = true
    end

    if update_RQR
        _update_RQR!(ws)
    end

    return ws
end

# Single-matrix setters. The EM M-step functions are `update_Z!(kf_ws, em_ws, n)` etc.
function set_Z!(ws::KalmanWorkspace, Z::AbstractMatrix)
    _check_params(ws, "set_Z!"; Z)
    copyto!(ws.Z, Z)
    return ws
end

function set_H!(ws::KalmanWorkspace, H::AbstractMatrix)
    _check_params(ws, "set_H!"; H)
    copyto!(ws.H, H)
    return ws
end

function set_T!(ws::KalmanWorkspace, Tmat::AbstractMatrix)
    _check_params(ws, "set_T!"; Tmat)
    copyto!(ws.Tmat, Tmat)
    return ws
end

function set_R!(ws::KalmanWorkspace, R::AbstractMatrix)
    _check_params(ws, "set_R!"; R)
    copyto!(ws.R, R)
    _update_RQR!(ws)
    return ws
end

function set_Q!(ws::KalmanWorkspace, Q::AbstractMatrix)
    _check_params(ws, "set_Q!"; Q)
    copyto!(ws.Q, Q)
    _update_RQR!(ws)
    return ws
end

# ============================================
# Missing data detection
# ============================================

# ============================================
# In-place Kalman filter
# ============================================

"""
    _propagate_cov!(dest, ws, P, block) -> dest

Overwrite `dest` with `T P T' + R Q R'`, the one-step predicted state covariance.

With `block == 0` this is two general `m × m` products. With `block == b` the
transition is block-companion,

    T = [ A₁ … A_lags ;  I 0 … 0 ],

so `T P` has the dense rows `A P` on top and rows `1 : m - b` of `P` beneath,
and the same shift applies on the right. The two `m³` products collapse to
`2 b m²`, a factor of `m / 2b` fewer multiplications — at a 17-variable VAR with
17 lags, `289 / 34`.

`dest` and `P` must not alias.
"""
function _propagate_cov!(
        dest::AbstractMatrix{T}, ws::KalmanWorkspace{T}, P::AbstractMatrix{T},
        block::Int) where {T}
    m = ws.state_dim

    if block == 0
        mul!(ws.tmp_mm2, ws.Tmat, P)
        mul!(dest, ws.tmp_mm2, transpose(ws.Tmat))
    else
        k = m - block
        A = view(ws.Tmat, 1:block, :)
        W = ws.tmp_mm2                      # W = T P
        mul!(view(W, 1:block, :), A, P)
        copyto!(view(W, (block + 1):m, :), view(P, 1:k, :))
        mul!(view(dest, :, 1:block), W, transpose(A))
        copyto!(view(dest, :, (block + 1):m), view(W, :, 1:k))
    end

    @inbounds for idx in eachindex(dest)
        dest[idx] += ws.RQR[idx]
    end
    return dest
end

"""
    kalman_filter!(ws::KalmanWorkspace, y::AbstractMatrix) -> loglik

Run Kalman filter in-place, storing results in workspace.

# Arguments
- `ws`: Pre-allocated workspace
- `y`: Observations (p × n matrix), missing values as NaN

# Returns
- `loglik`: Log-likelihood of non-missing observations. It is `-Inf` when an innovation
  covariance is not positive definite; the filter stops at that period and the
  workspace accessors then throw.

# Stored Results
Access via: `predicted_states(ws)`, `filtered_states(ws)`, etc.

# Notes
Uses BLAS Level 3 operations where possible for performance.
Cholesky factors of Ft are stored for potential reuse (e.g., EM algorithm).

When `set_companion_structure!` has declared a block-companion transition, the
covariance propagation uses the structured form; see `_propagate_cov!`.

When `set_rank_revealing!` has been declared, the measurement update takes a
symmetric eigendecomposition of `Fₜ` in place of the Cholesky factorization and
updates on the range of `Fₜ`, so a singular innovation covariance is supported.
That factorization allocates, unlike the rest of the pass. An observation whose
innovation leaves the range of `Fₜ` raises `InnovationSupportError`.
"""
function kalman_filter!(ws::KalmanWorkspace{T}, y::AbstractMatrix) where {T}
    y = _nan_obs(y)
    p, m, n = ws.obs_dim, ws.state_dim, ws.n_times

    _check_obs(ws, "kalman_filter!", y)

    block = _check_companion(ws)

    # Reset scalars. The status stays _NOT_RUN until the run completes, so an exception
    # thrown part way through leaves no output that the accessors would return.
    ws.status = _NOT_RUN
    ws.loglik = zero(T)
    ws.n_obs_valid = 0
    ws.n_obs_rows = 0
    ws.smoothed = false
    ws.smoothed_cov = false
    ws.crosscov = false

    # Working state/covariance held in scratch space (tmp_m1 / tmp_mm1).
    a_curr = ws.tmp_m1
    P_curr = ws.tmp_mm1
    copyto!(a_curr, ws.a1)
    copyto!(P_curr, ws.P1)

    log2pi = log(T(2π))

    @inbounds for t in 1:n
        # Store predicted state/covariance. `copyto!` is contiguous-safe.
        copyto!(view(ws.at, :, t), a_curr)
        copyto!(view(ws.Pt, :, :, t), P_curr)

        y_t = view(y, :, t)
        obs_rows_t = view(ws.obs_rows, :, t)
        pt = _observed_rows!(obs_rows_t, y_t)
        ws.n_observed[t] = pt
        @inbounds for i in 1:p
            ws.observed_mask[i, t] = !isnan(y_t[i])
        end

        if pt == 0
            ws.missing_mask[t] = true
            ws.Ft_rank[t] = 0

            # Mark innovations / gain as missing without wasted matmuls.
            fill!(view(ws.vt, :, t), T(NaN))
            fill!(view(ws.Ft, :, :, t), T(NaN))
            fill!(view(ws.Ft_L, :, :, t), zero(T))
            fill!(view(ws.Kt, :, :, t), zero(T))

            # Filtered = predicted for missing observations.
            copyto!(view(ws.att, :, t), a_curr)
            copyto!(view(ws.Ptt, :, :, t), P_curr)

            # Propagate: a = T * a, P = T * P * T' + RQR
            mul!(ws.tmp_m2, ws.Tmat, a_curr)
            copyto!(a_curr, ws.tmp_m2)

            _propagate_cov!(ws.tmp_mm3, ws, P_curr, block)
            copyto!(P_curr, ws.tmp_mm3)
        else
            ws.missing_mask[t] = false
            ws.n_obs_valid += 1

            rows = view(obs_rows_t, 1:pt)

            # The measurement update runs on the reduced system (Z_t, H_t, y_t)
            # formed by the observed rows. Every p-shaped buffer is used at its
            # leading pt rows and columns, so a partially observed period costs a
            # pt-sized factorization rather than a p-sized one. When pt == p the
            # gather is the identity and this is the fully observed recursion.
            Zr = view(ws.tmp_pm, 1:pt, :)             # Z_t (pt×m)
            @inbounds for j in 1:m, (k, i) in pairs(rows)

                Zr[k, j] = ws.Z[i, j]
            end

            # Innovation: v = y_t - Z_t * a
            v_full = view(ws.vt, :, t)
            fill!(v_full, T(NaN))
            v_t = view(v_full, 1:pt)
            tmp_p1r = view(ws.tmp_p1, 1:pt)
            mul!(tmp_p1r, Zr, a_curr)
            @inbounds for (k, i) in pairs(rows)
                v_t[k] = y_t[i] - tmp_p1r[k]
            end

            # F = Z_t * P * Z_t' + H_t, assembled into the leading block of
            # tmp_pp1 and stored in the leading block of Ft.
            ZrP = view(ws.tmp_pm2, 1:pt, :)           # Z_t * P (pt×m)
            mul!(ZrP, Zr, P_curr)
            Fr = view(ws.tmp_pp1, 1:pt, 1:pt)
            mul!(Fr, ZrP, transpose(Zr))
            Ft_view = view(ws.Ft, :, :, t)
            fill!(Ft_view, T(NaN))
            @inbounds for (l, j) in pairs(rows), (k, i) in pairs(rows)

                Fr[k, l] += ws.H[i, j]
                Ft_view[k, l] = Fr[k, l]
            end

            FtL_view = view(ws.Ft_L, :, :, t)
            Mr = view(ws.tmp_mp, :, 1:pt)

            if ws.rank_revealing
                # Eigenfactor the symmetric Fr and update on its range. Unlike
                # the Cholesky branch this allocates: `eigen!` builds fresh
                # factors rather than writing into workspace storage.
                eig = eigen!(Symmetric(Fr, :L))
                λ = eig.values
                V = eig.vectors

                λ_max = max(maximum(λ), zero(T))
                keep = λ .> ws.rank_rtol * λ_max
                rk = count(keep)
                ws.n_obs_rows += rk

                # The innovation must lie in the range of F: a component along a
                # discarded eigenvector is an observation the model cannot make.
                res = zero(T)
                for k in eachindex(λ)
                    keep[k] && continue
                    c = zero(T)
                    for i in 1:pt
                        c += V[i, k] * v_t[i]
                    end
                    res += c^2
                end
                res = sqrt(res)

                scale = zero(T)
                for (k, i) in pairs(rows)
                    scale = max(scale, abs(y_t[i]), abs(tmp_p1r[k]), abs(v_t[k]))
                end
                if res > ws.support_rtol * scale
                    throw(InnovationSupportError(
                        t, rk, pt, Float64(res), Float64(scale),
                        Float64(ws.support_rtol)))
                end

                # G = V_r Λ_r^{-1/2}, so F⁺ = G G' and G' v whitens the
                # innovation on the range.
                fill!(FtL_view, zero(T))
                G = view(FtL_view, 1:pt, 1:rk)
                logdetF = zero(T)
                c = 0
                for k in eachindex(λ)
                    keep[k] || continue
                    c += 1
                    logdetF += log(λ[k])
                    invsqrt = inv(sqrt(λ[k]))
                    for i in 1:pt
                        G[i, c] = V[i, k] * invsqrt
                    end
                end
                ws.Ft_rank[t] = rk

                # quad form on the support: v' F⁺ v = || G' v ||^2
                tmp_p2r = view(ws.tmp_p2, 1:rk)
                mul!(tmp_p2r, transpose(G), v_t)
                quad_form = zero(T)
                for i in 1:rk
                    quad_form += tmp_p2r[i]^2
                end
                ws.loglik += -T(0.5) * (logdetF + quad_form)

                # M = P Z_t' F⁺ = (G (G' (Z_t P)))'
                GZrP = view(ws.tmp_pm3, 1:rk, :)
                mul!(GZrP, transpose(G), ZrP)
                mul!(ZrP, G, GZrP)                    # ZrP := F⁺ (Z_t P)
                transpose!(Mr, ZrP)
            else
                ws.n_obs_rows += pt

                # Cholesky on Symmetric(:L) reads only the lower triangle, so we
                # do not need to symmetrize Fr first.
                cholF = cholesky!(Symmetric(Fr, :L); check = false)
                if !issuccess(cholF)
                    ws.status = t
                    ws.loglik = T(-Inf)
                    return ws.loglik
                end
                L_lower = LowerTriangular(cholF.factors)

                # Store lower triangle of L; zero the rest for clean reuse.
                fill!(FtL_view, zero(T))
                @inbounds for j in 1:pt, i in j:pt

                    FtL_view[i, j] = L_lower[i, j]
                end
                ws.Ft_rank[t] = pt

                # log|F| = 2 * Σ log L[i,i]
                logdetF = zero(T)
                @inbounds for i in 1:pt
                    logdetF += log(L_lower[i, i])
                end
                logdetF += logdetF  # multiply by 2

                # quad form: v' * F^{-1} * v = || L^{-1} v ||^2
                tmp_p2r = view(ws.tmp_p2, 1:pt)
                copyto!(tmp_p2r, v_t)
                ldiv!(L_lower, tmp_p2r)                   # tmp_p2r = L^{-1} v
                quad_form = zero(T)
                @inbounds for i in 1:pt
                    quad_form += tmp_p2r[i]^2
                end
                ws.loglik += -T(0.5) * (logdetF + quad_form)

                # Core quantity: M = P * Z_t' * F^{-1} (m×pt). Reused for K, att, Ptt.
                # Compute as (F^{-1} * Z_t * P)' via two ldiv! on Z_t*P, then transpose.
                ldiv!(L_lower, ZrP)                       # L^{-1} (Z_t*P)
                ldiv!(transpose(L_lower), ZrP)            # F^{-1} (Z_t*P)
                transpose!(Mr, ZrP)                       # Mr = M
            end

            # K_t = T * M, stored in the leading pt columns.
            K_view = view(ws.Kt, :, :, t)
            fill!(K_view, zero(T))
            mul!(view(K_view, :, 1:pt), ws.Tmat, Mr)

            # a_filt = a + M * v
            mul!(ws.tmp_m2, Mr, v_t)
            att_view = view(ws.att, :, t)
            @inbounds for i in 1:m
                att_view[i] = a_curr[i] + ws.tmp_m2[i]
            end

            # P_filt = P - M * (Z_t * P). The measurement update overwrote ZrP
            # with F⁻¹ (Z_t P), so recompute it once.
            mul!(ZrP, Zr, P_curr)
            mul!(ws.tmp_mm2, Mr, ZrP)                 # M * (Z_t*P) (m×m)
            Ptt_view = view(ws.Ptt, :, :, t)
            @inbounds for idx in eachindex(P_curr)
                Ptt_view[idx] = P_curr[idx] - ws.tmp_mm2[idx]
            end

            # P_filt must stay symmetric. The subtraction above is symmetric
            # only up to rounding, and the asymmetric part is propagated by
            # P = T P_filt T' + RQR, so a transition with spectral radius above
            # one amplifies it by ρ(T)² each period. Left alone it reaches the
            # scale of F = Z P Z', whose Cholesky reads the lower triangle only,
            # and the factorization then fails on a matrix that is positive
            # definite in exact arithmetic.
            for j in 1:m, i in 1:(j - 1)

                avg = (Ptt_view[i, j] + Ptt_view[j, i]) / 2
                Ptt_view[i, j] = avg
                Ptt_view[j, i] = avg
            end

            # Predict next: a = T * a_filt, P = T * P_filt * T' + RQR
            mul!(a_curr, ws.Tmat, att_view)
            _propagate_cov!(P_curr, ws, Ptt_view, block)
        end
    end

    # The Gaussian constant counts every observed scalar, which is Σₜ pₜ rather
    # than p times the number of periods carrying any observation.
    ws.loglik += -ws.n_obs_rows * log2pi / 2
    ws.status = 0
    return ws.loglik
end

# ============================================
# In-place Kalman smoother
# ============================================

"""
    _apply_Finv!(x, ws, t, pt, scratch) -> x

Overwrite `x` with `Fₜ⁻¹ x`, where `x` is a `pₜ`-vector or a `pₜ × m` matrix.

On the Cholesky path this is the pair of triangular solves against the stored
factor. On the rank-revealing path `Ft_L` holds `G` with `Fₜ⁺ = G G'`, and the
product is formed through `scratch`, whose leading `Ft_rank[t]` rows (or rows and
all `m` columns) take `G' x`. Directions outside the range of `Fₜ` are annihilated,
which is what makes the smoother recursions well defined when `Fₜ` is singular.
"""
function _apply_Finv!(
        x::AbstractVecOrMat, ws::KalmanWorkspace, t::Int, pt::Int,
        scratch::AbstractVecOrMat)
    if ws.rank_revealing
        rk = ws.Ft_rank[t]
        G = view(ws.Ft_L, 1:pt, 1:rk, t)
        if x isa AbstractVector
            w = view(scratch, 1:rk)
            mul!(w, transpose(G), x)
            mul!(x, G, w)
        else
            w = view(scratch, 1:rk, axes(x, 2))
            mul!(w, transpose(G), x)
            mul!(x, G, w)
        end
    else
        L_lower = LowerTriangular(view(ws.Ft_L, 1:pt, 1:pt, t))
        ldiv!(L_lower, x)
        ldiv!(transpose(L_lower), x)
    end
    return x
end

"""
    kalman_smoother!(ws::KalmanWorkspace; crosscov::Bool=false, covariances::Bool=true)

Run RTS smoother in-place using filter results stored in workspace.

# Arguments
- `ws`: Workspace with filter results (must call `kalman_filter!` first; throws an
  `ArgumentError` if the filter has not run or failed)
- `crosscov`: If true, also compute the cross-lag covariances, which EM needs
  (default: false)
- `covariances`: If true, compute smoothed covariances (default: true). When
  false, only the smoothed means are produced, `smoothed_states_cov(ws)` throws
  until a smoother run computes them, and `crosscov` must also be false.

# Stored Results
- `αs`: Smoothed states E[αₜ|y₁:ₙ]
- `Vs`: Smoothed covariances Var[αₜ|y₁:ₙ] (if covariances=true)
- `Pcross`: Cross-lag covariances Cov[αₜ₊₁,αₜ|y₁:ₙ] (if crosscov=true)

# Notes
Implements Durbin & Koopman (2012) backward recursion.
Cross-lag covariances computed during the same backward pass for efficiency.

The mean recursion `r_{t-1} = Zₜ' Fₜ⁻¹ vₜ + Lₜ' rₜ` does not involve `N`, which
enters only the covariance. With `covariances=false` the `N` recursion and the
two `m × m` products forming `Vₜ` are therefore skipped, and `Lₜ' rₜ` is
evaluated as `T' rₜ - Zₜ'(Kₜ' rₜ)` so that the `m × m` matrix `Lₜ` is never
formed. What remains is `O(m²)` per period instead of `O(m³)`.
"""
function kalman_smoother!(
        ws::KalmanWorkspace{T}; crosscov::Bool = false,
        covariances::Bool = true) where {T}
    !covariances && crosscov &&
        throw(ArgumentError(
            "crosscov=true requires covariances=true: the cross-lag covariance " *
            "is built from the smoothed covariance"))
    _require_output(ws, :kalman_smoother!)
    m, n = ws.state_dim, ws.n_times
    p = ws.obs_dim

    if !covariances
        _kalman_smoother_mean!(ws)
        ws.smoothed = true
        ws.smoothed_cov = false
        ws.crosscov = false
        return nothing
    end

    # Initialize smoother recursion: r = 0, N = 0
    fill!(ws.r_smooth, zero(T))
    fill!(ws.N_smooth, zero(T))

    # Get Cholesky lower triangular for solves
    # We'll reconstruct from stored Ft_L

    @inbounds for t in n:-1:1
        # Views to stored filter results
        a_t = view(ws.at, :, t)
        P_t = view(ws.Pt, :, :, t)

        if ws.missing_mask[t]
            # Missing observation: r_{t-1} = T' * r_t, N_{t-1} = T' * N_t * T

            # tmp_m1 = T' * r
            mul!(ws.tmp_m1, ws.Tmat', ws.r_smooth)
            copyto!(ws.r_smooth, ws.tmp_m1)

            # tmp_mm1 = T' * N
            mul!(ws.tmp_mm1, ws.Tmat', ws.N_smooth)
            # tmp_mm2 = tmp_mm1 * T = T' * N * T
            mul!(ws.tmp_mm2, ws.tmp_mm1, ws.Tmat)
            copyto!(ws.N_smooth, ws.tmp_mm2)

            # Smoothed state: α = a + P * r
            mul!(ws.tmp_m1, P_t, ws.r_smooth)
            for i in 1:m
                ws.αs[i, t] = a_t[i] + ws.tmp_m1[i]
            end

            # Smoothed covariance: V = P - P * N * P
            mul!(ws.tmp_mm1, ws.N_smooth, P_t)     # N * P
            mul!(ws.tmp_mm2, P_t, ws.tmp_mm1)      # P * N * P
            for j in 1:m, i in 1:m

                ws.Vs[i, j, t] = P_t[i, j] - ws.tmp_mm2[i, j]
            end
        else
            # Valid observation. The filter stored pt-sized blocks for this
            # period, so the recursion runs against the same reduced system:
            # Z_t is the observed rows of Z, and K_t and L_t are its leading
            # pt columns.
            pt = ws.n_observed[t]
            rows = view(ws.obs_rows, 1:pt, t)

            v_t = view(ws.vt, 1:pt, t)
            K_t = view(ws.Kt, :, 1:pt, t)

            Zr = view(ws.tmp_pm, 1:pt, :)          # Z_t (pt×m)
            @inbounds for j in 1:m, (k, i) in pairs(rows)

                Zr[k, j] = ws.Z[i, j]
            end

            # L_smooth = T - K_t * Z_t
            mul!(ws.L_smooth, K_t, Zr)
            for j in 1:m, i in 1:m

                ws.L_smooth[i, j] = ws.Tmat[i, j] - ws.L_smooth[i, j]
            end

            tmp_p1r = view(ws.tmp_p1, 1:pt)
            copyto!(tmp_p1r, v_t)
            _apply_Finv!(tmp_p1r, ws, t, pt, ws.tmp_p2)  # tmp_p1r = F^{-1} v

            # r_{t-1} = Z_t' * F^{-1} * v + L' * r
            mul!(ws.tmp_m1, transpose(Zr), tmp_p1r)
            mul!(ws.tmp_m2, ws.L_smooth', ws.r_smooth)  # L' * r
            for i in 1:m
                ws.r_smooth[i] = ws.tmp_m1[i] + ws.tmp_m2[i]
            end

            # F^{-1} Z_t, after r_{t-1} because it overwrites Zr in place.
            _apply_Finv!(Zr, ws, t, pt, ws.tmp_pm3)

            # N_{t-1} = Z_t' * F^{-1} * Z_t + L' * N * L. Rebuild Z_t' from the
            # gathered rows, since Zr now holds F^{-1} Z_t.
            ZrT = view(ws.tmp_mp, :, 1:pt)
            @inbounds for (k, i) in pairs(rows), j in 1:m

                ZrT[j, k] = ws.Z[i, j]
            end
            mul!(ws.tmp_mm1, ZrT, Zr)              # Z_t' * F^{-1} * Z_t (m × m)
            mul!(ws.tmp_mm2, ws.L_smooth', ws.N_smooth)  # L' * N
            mul!(ws.tmp_mm3, ws.tmp_mm2, ws.L_smooth)    # L' * N * L
            for j in 1:m, i in 1:m

                ws.N_smooth[i, j] = ws.tmp_mm1[i, j] + ws.tmp_mm3[i, j]
            end

            # Smoothed state: α = a + P * r
            mul!(ws.tmp_m1, P_t, ws.r_smooth)
            for i in 1:m
                ws.αs[i, t] = a_t[i] + ws.tmp_m1[i]
            end

            # Smoothed covariance: V = P - P * N * P
            mul!(ws.tmp_mm1, ws.N_smooth, P_t)     # N * P
            mul!(ws.tmp_mm2, P_t, ws.tmp_mm1)      # P * N * P
            for j in 1:m, i in 1:m

                ws.Vs[i, j, t] = P_t[i, j] - ws.tmp_mm2[i, j]
            end
        end

        # Cross-lag covariance: Cov[α_{t}, α_{t-1} | y_{1:n}]
        # Computed during backward pass for t > 1
        # P_{t,t-1|n} = (I - P_t * N_{t-1}) * L_{t-1} * P_{t-1}
        # where L_{t-1} = T - K_{t-1} * Z
        #
        # Alternative formula (Shumway & Stoffer):
        # P_{t,t-1|n} = V_t * J_{t-1}'
        # where J_{t-1} = P_{t-1|t-1} * T' * inv(P_{t|t-1})
        #
        # We use: P_{t,t-1|n} = P_t * L_{t-1}' * (I - N_{t-1} * P_{t-1}) for t > 1
        # Actually easier: compute after main loop using stored V, P, T, K

        # For efficiency, compute cross-cov here using the formula:
        # Cov[α_{t+1}, α_t | y] for t = 1..n-1
        # We're at time t going backward, so we compute Cov[α_t, α_{t-1}]
        # which corresponds to Pcross[:,:,t-1]

        if crosscov && t > 1
            # Need: Cov[α_t, α_{t-1} | y_{1:n}] = P_{t,t-1|n}
            # Using formula: P_{t,t-1|n} = V_t * J_{t-1}'
            # where J_{t-1} = P_{t-1|t-1} * T' * inv(P_{t|t-1})
            #             = Ptt[:,:,t-1] * T' * inv(Pt[:,:,t])

            V_t = view(ws.Vs, :, :, t)
            Ptt_tm1 = view(ws.Ptt, :, :, (t - 1))      # P_{t-1|t-1}
            Pt_t = view(ws.Pt, :, :, t)            # P_{t|t-1}

            # J_{t-1} = Ptt_{t-1} * T' * inv(Pt_t)
            # tmp_mm1 = Ptt_{t-1} * T'
            mul!(ws.tmp_mm1, Ptt_tm1, ws.Tmat')

            # tmp_mm2 = inv(Pt_t) - use Cholesky with regularization for numerical stability
            copyto!(ws.tmp_mm2, Pt_t)
            # Symmetrize and add small regularization
            eps_reg = T(1e-10) * max(one(T), tr(ws.tmp_mm2) / m)
            for j in 1:m
                for i in 1:(j - 1)
                    avg = (ws.tmp_mm2[i, j] + ws.tmp_mm2[j, i]) / 2
                    ws.tmp_mm2[i, j] = avg
                    ws.tmp_mm2[j, i] = avg
                end
                ws.tmp_mm2[j, j] += eps_reg  # regularization on diagonal
            end
            chol_P = cholesky!(Symmetric(ws.tmp_mm2, :L))

            # J = tmp_mm1 * inv(Pt_t)
            # Solve Pt_t * J' = tmp_mm1' for J'
            # i.e., for each column j of tmp_mm1', solve Pt_t * x = tmp_mm1'[:,j]
            # tmp_mm1' is m × m, so we solve m systems

            # Actually: J = tmp_mm1 * inv(Pt_t)
            # J' = inv(Pt_t)' * tmp_mm1' = inv(Pt_t) * tmp_mm1' (symmetric)
            # Solve Pt_t * J' = tmp_mm1'

            # Copy tmp_mm1' into J_smooth (which is m × m)
            for j in 1:m, i in 1:m

                ws.J_smooth[i, j] = ws.tmp_mm1[j, i]  # transpose
            end

            # Solve L * L' * J' = tmp_mm1' column by column
            L_P = LowerTriangular(chol_P.L)
            for j in 1:m
                col = view(ws.J_smooth, :, j)
                ldiv!(L_P, col)
                ldiv!(L_P', col)
            end
            # Now J_smooth = J' = (Ptt_{t-1} * T' * inv(Pt_t))'

            # P_{t,t-1|n} = V_t * J_{t-1}' = V_t * J_smooth
            Pcross_t = view(ws.Pcross, :, :, (t - 1))
            mul!(Pcross_t, V_t, ws.J_smooth)
        end
    end

    ws.smoothed = true
    ws.smoothed_cov = true
    ws.crosscov = crosscov
    return nothing
end

"""
    _kalman_smoother_mean!(ws::KalmanWorkspace) -> nothing

Backward recursion for the smoothed means alone, writing `ws.αs`.

Runs `r_{t-1} = Zₜ' Fₜ⁻¹ vₜ + Lₜ' rₜ` and `αₜ = aₜ + Pₜ rₜ`. The identity
`Lₜ' rₜ = T' rₜ - Zₜ'(Kₜ' rₜ)` keeps every operation at matrix-vector cost, so a
period costs `O(m²)` against the `O(m³)` of the covariance recursion.
"""
function _kalman_smoother_mean!(ws::KalmanWorkspace{T}) where {T}
    m, n = ws.state_dim, ws.n_times

    fill!(ws.r_smooth, zero(T))

    @inbounds for t in n:-1:1
        a_t = view(ws.at, :, t)
        P_t = view(ws.Pt, :, :, t)

        # r_{t-1} = L_t' r_t (+ the observation term below, when there is one).
        mul!(ws.tmp_m1, transpose(ws.Tmat), ws.r_smooth)

        if !ws.missing_mask[t]
            pt = ws.n_observed[t]
            rows = view(ws.obs_rows, 1:pt, t)
            v_t = view(ws.vt, 1:pt, t)
            K_t = view(ws.Kt, :, 1:pt, t)

            # F_t^{-1} v_t, then subtract K_t' r_t to complete L_t' r_t.
            tmp_p1r = view(ws.tmp_p1, 1:pt)
            copyto!(tmp_p1r, v_t)
            _apply_Finv!(tmp_p1r, ws, t, pt, ws.tmp_p2)

            tmp_p2r = view(ws.tmp_p2, 1:pt)
            mul!(tmp_p2r, transpose(K_t), ws.r_smooth)
            for k in 1:pt
                tmp_p2r[k] = tmp_p1r[k] - tmp_p2r[k]
            end

            # Scatter Z_t' (F_t^{-1} v_t - K_t' r_t) back over the m state rows.
            for (k, i) in pairs(rows), j in 1:m

                ws.tmp_m1[j] += ws.Z[i, j] * tmp_p2r[k]
            end
        end

        copyto!(ws.r_smooth, ws.tmp_m1)

        mul!(ws.tmp_m2, P_t, ws.r_smooth)
        for i in 1:m
            ws.αs[i, t] = a_t[i] + ws.tmp_m2[i]
        end
    end

    return nothing
end

# ============================================
# Combined filter and smooth
# ============================================

"""
    kalman_filter_and_smooth!(ws, y::AbstractMatrix; crosscov=false, covariances=true) -> loglik

Run filter and smoother in one call.

# Arguments
- `ws`: Pre-allocated workspace
- `y`: Observations (p × n)

Keywords are passed to `kalman_smoother!`:
- `crosscov`: Also compute the cross-lag covariances (default: false)
- `covariances`: Compute smoothed covariances (default: true; `KalmanWorkspace`
  only). See `kalman_smoother!` for what is skipped when this is false.

# Returns
- `loglik`: Log-likelihood. When the filter fails it is `-Inf` and the smoother is
  skipped.
"""
function kalman_filter_and_smooth!(ws::AbstractFilterWorkspace, y::AbstractMatrix;
        kwargs...)
    y = _nan_obs(y)
    loglik = kalman_filter!(ws, y)
    _run_status(ws) == 0 && kalman_smoother!(ws; kwargs...)
    return loglik
end

# ============================================
# Additional Accessors (aliases and new ones)
# ============================================

"""
Return the stored factors of Ft as a view.

Each slice holds the lower-triangular Cholesky factor of the period's innovation
covariance in its leading `n_observed[t]` rows and columns. On the rank-revealing
path it instead holds `G`, with `Fₜ⁺ = G G'`, in its leading
`n_observed[t] × innovation_ranks(ws)[t]` block; see [`set_rank_revealing!`](@ref).
"""
cholesky_factors(ws::KalmanWorkspace) = _require_output(ws, :cholesky_factors).Ft_L

_has_smoothed_cov(ws::KalmanWorkspace) = ws.smoothed_cov

"""Return count of non-missing observations."""
n_valid_obs(ws::KalmanWorkspace) = _require_output(ws, :n_valid_obs).n_obs_valid

# ============================================
# Dimension accessors
# ============================================

obs_dim(ws::KalmanWorkspace) = ws.obs_dim
state_dim(ws::KalmanWorkspace) = ws.state_dim
shock_dim(ws::KalmanWorkspace) = ws.shock_dim
n_times(ws::KalmanWorkspace) = ws.n_times

# ============================================
# DiffuseKalmanWorkspace - Exact diffuse initialization
# ============================================

"""
    DiffuseKalmanWorkspace{T<:Real}

Pre-allocated workspace for in-place Kalman filter with exact diffuse initialization.

Extends `KalmanWorkspace` with additional storage for the diffuse period:
- `P1_star`: Finite part of initial covariance
- `P1_inf`: Diffuse (infinite) part of initial covariance
- `Pinf`: Current diffuse covariance (working storage)
- `Pstar`: Current finite covariance (working storage)
- `Finf`: Diffuse innovation covariance Z*Pinf*Z'
- `diffuse_flags`: Track which step type was used (1=Finf invertible, 2=Finf singular but nonzero, 0=Finf zero, -1=missing)
- `d`: Length of diffuse period (updated during filtering)
- `diffuse_ended`: Flag indicating if diffuse period has ended

# Usage
```julia
ws = DiffuseKalmanWorkspace{Float64}(p, m, r, n)
set_params!(ws, Z, H, T, R, Q)
set_initial_diffuse!(ws, a1, P1_star, P1_inf)
kalman_filter!(ws, y)
```

# See Also
- `KalmanWorkspace`: Standard filter workspace
- `kalman_filter(p, y, a1, P1_star, P1_inf)`: Pure functional version
"""
mutable struct DiffuseKalmanWorkspace{T <: Real} <: AbstractFilterWorkspace
    # Embed standard workspace (for shared functionality)
    base::KalmanWorkspace{T}

    # Diffuse initialization
    P1_star::Matrix{T}     # m × m: finite part of initial covariance
    P1_inf::Matrix{T}      # m × m: diffuse part of initial covariance

    # Diffuse period working storage
    Pinf::Matrix{T}        # m × m: current diffuse covariance
    Pstar::Matrix{T}       # m × m: current finite covariance
    Finf::Matrix{T}        # p × p: diffuse innovation covariance
    Finf_inv::Matrix{T}    # p × p: inverse of Finf (when invertible)

    # Diffuse period tracking
    diffuse_flags::Vector{Int}  # max length n: step type flags
    d::Int                      # current diffuse period length
    diffuse_ended::Bool         # flag for phase switching
    tol::T                      # tolerance for diffuse convergence

    # Additional scratch for diffuse computations
    tmp_mm_diff1::Matrix{T}  # m × m: for Linf, etc.
    tmp_mm_diff2::Matrix{T}  # m × m
    tmp_mp_diff::Matrix{T}   # m × p: for Kinf, Kstar
    tmp_pp_diff::Matrix{T}   # p × p: for Fstar
end

"""
    DiffuseKalmanWorkspace{T}(p::Int, m::Int, r::Int, n::Int; tol=1e-8) where T

Create diffuse workspace with specified dimensions.
"""
function DiffuseKalmanWorkspace{T}(
        p::Int,
        m::Int,
        r::Int,
        n::Int;
        tol::Real = 1e-8
) where {T <: Real}
    base = KalmanWorkspace{T}(p, m, r, n)

    DiffuseKalmanWorkspace{T}(
        base,
        # Diffuse initialization
        zeros(T, m, m),  # P1_star
        zeros(T, m, m),  # P1_inf
        # Diffuse working storage
        zeros(T, m, m),  # Pinf
        zeros(T, m, m),  # Pstar
        zeros(T, p, p),  # Finf
        zeros(T, p, p),  # Finf_inv
        # Diffuse tracking
        zeros(Int, n),   # diffuse_flags (preallocate max)
        0,                       # d
        false,                   # diffuse_ended
        T(tol),                  # tol
        # Additional scratch
        zeros(T, m, m),  # tmp_mm_diff1
        zeros(T, m, m),  # tmp_mm_diff2
        zeros(T, m, p),  # tmp_mp_diff
        zeros(T, p, p)   # tmp_pp_diff
    )
end

function DiffuseKalmanWorkspace(p::Int, m::Int, r::Int, n::Int; tol::Real = 1e-8)
    DiffuseKalmanWorkspace{Float64}(p, m, r, n; tol = tol)
end

"""
    DiffuseKalmanWorkspace(Z, H, T, R, Q, a1, P1_star, P1_inf, n; tol=1e-8)

Create workspace from parameter matrices and set parameters.
"""
function DiffuseKalmanWorkspace(
        Z::AbstractMatrix{T},
        H::AbstractMatrix,
        Tmat::AbstractMatrix,
        R::AbstractMatrix,
        Q::AbstractMatrix,
        a1::AbstractVector,
        P1_star::AbstractMatrix,
        P1_inf::AbstractMatrix,
        n::Int;
        tol::Real = 1e-8
) where {T}
    p, m = size(Z)
    r = size(Q, 1)

    ws = DiffuseKalmanWorkspace{T}(p, m, r, n; tol = tol)
    set_params!(ws, Z, H, Tmat, R, Q)
    set_initial_diffuse!(ws, a1, P1_star, P1_inf)

    return ws
end

# ============================================
# DiffuseKalmanWorkspace - Parameter setters
# ============================================

"""
    set_params!(ws::DiffuseKalmanWorkspace, Z, H, T, R, Q)

Set system matrices (delegates to base workspace).
"""
function set_params!(
        ws::DiffuseKalmanWorkspace,
        Z::AbstractMatrix,
        H::AbstractMatrix,
        Tmat::AbstractMatrix,
        R::AbstractMatrix,
        Q::AbstractMatrix
)
    set_params!(ws.base, Z, H, Tmat, R, Q)
    return ws
end

"""
    set_initial_diffuse!(ws::DiffuseKalmanWorkspace, a1, P1_star, P1_inf)

Set initial state mean and diffuse covariance decomposition.
"""
function set_initial_diffuse!(
        ws::DiffuseKalmanWorkspace,
        a1::AbstractVector,
        P1_star::AbstractMatrix,
        P1_inf::AbstractMatrix
)
    _check_initial(ws.base, "set_initial_diffuse!", a1, P1_star)
    _check_size("set_initial_diffuse!", "P1_inf", P1_inf, size(ws.P1_inf))
    copyto!(ws.base.a1, a1)
    copyto!(ws.P1_star, P1_star)
    copyto!(ws.P1_inf, P1_inf)
    # Also set P1 in base to P1_star for compatibility
    copyto!(ws.base.P1, P1_star)
    return ws
end

# The filter output lives in the base workspace; the shared accessors read it from there.
_filter_store(ws::DiffuseKalmanWorkspace) = ws.base
system_matrices(ws::DiffuseKalmanWorkspace) = system_matrices(ws.base)

obs_dim(ws::DiffuseKalmanWorkspace) = ws.base.obs_dim
state_dim(ws::DiffuseKalmanWorkspace) = ws.base.state_dim
shock_dim(ws::DiffuseKalmanWorkspace) = ws.base.shock_dim
n_times(ws::DiffuseKalmanWorkspace) = ws.base.n_times

observed_mask(ws::DiffuseKalmanWorkspace) = observed_mask(ws.base)
observed_rows(ws::DiffuseKalmanWorkspace, t::Integer) = observed_rows(ws.base, t)

"""
    diffuse_period(ws::DiffuseKalmanWorkspace) -> Int

Return the number of observations in the diffuse period.
"""
diffuse_period(ws::DiffuseKalmanWorkspace) = (_require_output(ws, :diffuse_period); ws.d)

"""
    diffuse_flags(ws::DiffuseKalmanWorkspace) -> Vector{Int}

Return flags for diffuse period steps (view of first d elements).
"""
function diffuse_flags(ws::DiffuseKalmanWorkspace)
    _require_output(ws, :diffuse_flags)
    return view(ws.diffuse_flags, 1:ws.d)
end

# ============================================
# In-place diffuse Kalman filter
# ============================================

"""
    _safe_inverse_inplace!(Finv, F, tmp, tol) -> flag

Attempt to invert F in-place into Finv. Returns 1 if successful, 0 if singular.
Uses tmp as scratch space.
"""
@inline function _safe_inverse_inplace!(
        Finv::AbstractMatrix{T},
        F::AbstractMatrix{T},
        tmp::AbstractMatrix{T},
        tol::Real
) where {T}
    d = det(F)
    if abs(d) > tol
        copyto!(tmp, F)
        # Symmetrize
        p = size(F, 1)
        for j in 1:p, i in 1:(j - 1)

            avg = (tmp[i, j] + tmp[j, i]) / 2
            tmp[i, j] = avg
            tmp[j, i] = avg
        end
        # Compute inverse via Cholesky
        chol = cholesky!(Symmetric(tmp, :L))
        copyto!(Finv, I)
        ldiv!(chol, Finv)
        return 1
    else
        return 0
    end
end

"""
    kalman_filter!(ws::DiffuseKalmanWorkspace, y::AbstractMatrix) -> loglik

Run Kalman filter with exact diffuse initialization in-place. The workspace type
selects it: a `DiffuseKalmanWorkspace` holds the diffuse part `P1_inf` of the initial
covariance, while a `KalmanWorkspace` runs the ordinary filter.

# Arguments
- `ws`: Pre-allocated diffuse workspace (must call `set_initial_diffuse!` first)
- `y`: Observations (p × n matrix), missing values as NaN

# Returns
- `loglik`: Log-likelihood (non-diffuse observations only)

# Notes
The filter runs in two phases:
1. Diffuse period: While norm(Pinf) > tol, uses exact diffuse recursion
2. Non-diffuse period: Standard Kalman filter recursion

Only observations after the diffuse period contribute to the log-likelihood.

NaN marks a missing value, one element at a time. In a partially observed period
both phases run on the reduced system formed by the observed rows, and `vt`, `Ft`,
`Ft_L` and `Kt` hold pₜ-sized blocks in their leading rows and columns, as in
[`kalman_filter!`](@ref) on a `KalmanWorkspace`.

A diffuse-period step where `Finf = Z_t Pinf Z_t'` is singular but nonzero (more
observed rows than diffuse directions they load on) is processed one row at a time
(Koopman & Durbin 2000). A non-diagonal `H` is decorrelated first with its unit LDL
factor.
"""
function kalman_filter!(ws::DiffuseKalmanWorkspace{T}, y::AbstractMatrix) where {T}
    y = _nan_obs(y)
    base = ws.base
    p, m, n = base.obs_dim, base.state_dim, base.n_times

    _check_obs(base, "kalman_filter!", y)
    base.rank_revealing && throw(ArgumentError(
        "the rank-revealing measurement update is implemented for " *
        "KalmanWorkspace only; clear it with set_rank_revealing!(ws.base, false) " *
        "before filtering with a DiffuseKalmanWorkspace"))

    # Reset state
    base.loglik = zero(T)
    base.n_obs_valid = 0
    base.n_obs_rows = 0
    base.smoothed = false
    base.smoothed_cov = false
    base.crosscov = false
    ws.d = 0
    ws.diffuse_ended = false

    # Initialize: a = a1, Pstar = P1_star, Pinf = P1_inf
    a_curr = base.tmp_m1
    copyto!(a_curr, base.a1)
    copyto!(ws.Pstar, ws.P1_star)
    copyto!(ws.Pinf, ws.P1_inf)

    log2pi = log(T(2π))
    tol = ws.tol
    Tt = transpose(base.Tmat)

    @inbounds for t in 1:n
        # Store predicted state and covariance (Pstar is the finite part)
        copyto!(view(base.at, :, t), a_curr)
        copyto!(view(base.Pt, :, :, t), ws.Pstar)

        y_t = view(y, :, t)
        obs_rows_t = view(base.obs_rows, :, t)
        pt = _observed_rows!(obs_rows_t, y_t)
        base.n_observed[t] = pt
        for i in 1:p
            base.observed_mask[i, t] = !isnan(y_t[i])
        end

        if pt == 0
            base.missing_mask[t] = true

            # Store NaN for innovation
            for i in 1:p
                base.vt[i, t] = T(NaN)
            end

            # Compute F = Z * Pstar * Z' + H
            mul!(base.tmp_pm, base.Z, ws.Pstar)
            mul!(base.tmp_pp1, base.tmp_pm, base.Z')
            for j in 1:p, i in 1:p

                base.Ft[i, j, t] = base.tmp_pp1[i, j] + base.H[i, j]
                base.Ft_L[i, j, t] = zero(T)
            end

            fill!(view(base.Kt, :, :, t), zero(T))
            copyto!(view(base.att, :, t), a_curr)
            copyto!(view(base.Ptt, :, :, t), ws.Pstar)

            mul!(base.tmp_m2, base.Tmat, a_curr)
            copyto!(a_curr, base.tmp_m2)

            # Pstar = T * Pstar * T' + RQR
            mul!(base.tmp_mm2, base.Tmat, ws.Pstar)
            mul!(base.tmp_mm3, base.tmp_mm2, Tt)
            for j in 1:m, i in 1:m

                ws.Pstar[i, j] = base.tmp_mm3[i, j] + base.RQR[i, j]
            end

            if !ws.diffuse_ended
                ws.d += 1
                ws.diffuse_flags[ws.d] = -1  # missing

                # Pinf = T * Pinf * T'
                mul!(base.tmp_mm2, base.Tmat, ws.Pinf)
                mul!(ws.tmp_mm_diff1, base.tmp_mm2, Tt)
                copyto!(ws.Pinf, ws.tmp_mm_diff1)

                if norm(ws.Pinf) <= tol
                    ws.diffuse_ended = true
                end
            end
            continue
        end

        base.missing_mask[t] = false
        rows = view(obs_rows_t, 1:pt)

        # Both phases run on the reduced system (Z_t, H_t, y_t) formed by the
        # observed rows, using the leading pt rows and columns of every p-shaped
        # buffer. When pt == p this is the fully observed recursion.
        Zr = view(base.tmp_pm, 1:pt, :)          # Z_t (pt×m)
        for j in 1:m, (k, i) in pairs(rows)

            Zr[k, j] = base.Z[i, j]
        end
        Hr = view(base.tmp_pp2, 1:pt, 1:pt)      # H_t (pt×pt)
        for (l, j) in pairs(rows), (k, i) in pairs(rows)

            Hr[k, l] = base.H[i, j]
        end

        # Innovation: v = y_t - Z_t * a
        v_full = view(base.vt, :, t)
        fill!(v_full, T(NaN))
        v_t = view(v_full, 1:pt)
        tmp_p1r = view(base.tmp_p1, 1:pt)
        mul!(tmp_p1r, Zr, a_curr)
        for (k, i) in pairs(rows)
            v_t[k] = y_t[i] - tmp_p1r[k]
        end

        Ft_view = view(base.Ft, :, :, t)
        fill!(Ft_view, T(NaN))
        fill!(view(base.Ft_L, :, :, t), zero(T))
        K_full = view(base.Kt, :, :, t)
        fill!(K_full, zero(T))
        K_t = view(K_full, :, 1:pt)

        ZrP = view(base.tmp_pm2, 1:pt, :)        # Z_t * P (pt×m)
        tmp_p2r = view(base.tmp_p2, 1:pt)
        att_view = view(base.att, :, t)
        Ptt_view = view(base.Ptt, :, :, t)

        # During the diffuse period, check whether Finf = Z_t * Pinf * Z_t' is
        # invertible.
        flag = 0
        if !ws.diffuse_ended
            ws.d += 1
            Finf = view(ws.Finf, 1:pt, 1:pt)
            Finf_inv = view(ws.Finf_inv, 1:pt, 1:pt)
            mul!(ZrP, Zr, ws.Pinf)
            mul!(Finf, ZrP, transpose(Zr))
            flag = _diffuse_step_flag(Finf, tol)
            flag == 1 &&
                _safe_inverse_inplace!(Finf_inv, Finf, view(ws.tmp_pp_diff, 1:pt, 1:pt), tol)
            ws.diffuse_flags[ws.d] = flag
        end

        if flag == 2
            # === Finf singular but nonzero: process the rows one at a time ===
            # The filtered finite covariance is built in Ptt and the diffuse one in
            # Pinf; G is the gain with a_filt = a + G v.
            Zs = ZrP
            copyto!(Zs, Zr)
            copyto!(tmp_p2r, v_t)
            copyto!(Ptt_view, ws.Pstar)
            G = view(ws.tmp_mp_diff, :, 1:pt)
            (ll, nrows, ok) = _univariate_diffuse_update!(
                base.tmp_m2, G, Ptt_view, ws.Pinf, Zs, tmp_p2r, view(base.tmp_p1, 1:pt),
                Hr, view(ws.tmp_pp_diff, 1:pt, 1:pt),
                view(base.tmp_mp, :, 1), view(base.tmp_mp, :, 2), tol)
            if !ok
                base.status = t
                base.loglik = T(-Inf)
                return base.loglik
            end
            base.loglik += ll
            base.n_obs_rows += nrows
            nrows > 0 && (base.n_obs_valid += 1)

            for i in 1:m
                att_view[i] = a_curr[i] + base.tmp_m2[i]
            end
            mul!(K_t, base.Tmat, G)
            copyto!(view(Ft_view, 1:pt, 1:pt), view(ws.Finf, 1:pt, 1:pt))

            # Predict next: a = T a_filt, Pinf = T Pinf_filt T',
            # Pstar = T Pstar_filt T' + RQR
            mul!(a_curr, base.Tmat, att_view)
            mul!(base.tmp_mm2, base.Tmat, ws.Pinf)
            mul!(ws.Pinf, base.tmp_mm2, Tt)
            mul!(base.tmp_mm2, base.Tmat, Ptt_view)
            mul!(ws.Pstar, base.tmp_mm2, Tt)
            for j in 1:m, i in 1:m

                ws.Pstar[i, j] += base.RQR[i, j]
            end
        elseif flag == 1
            # === Finf invertible: exact diffuse update, no likelihood term ===
            # Durbin & Koopman (2012, §5.2). With A = Pinf * Z_t' * Finf^{-1},
            # the filtered state and the finite part of its covariance are the
            # κ → ∞ limits of the update with P = κ Pinf + Pstar:
            #   a_filt     = a + A v
            #   Pstar_filt = Pstar - A Mstar' - Mstar A' + A Fstar A'
            #   Pinf_filt  = Pinf - A Minf'
            # where Minf = Pinf Z_t' and Mstar = Pstar Z_t'. Predicting each
            # through T gives the K⁽⁰⁾/K⁽¹⁾ recursion.
            Minf = view(base.tmp_mp, :, 1:pt)
            mul!(Minf, ws.Pinf, transpose(Zr))
            A = view(ws.tmp_mp_diff, :, 1:pt)
            mul!(A, Minf, Finf_inv)

            # a_filt = a + A v
            mul!(base.tmp_m2, A, v_t)
            for i in 1:m
                att_view[i] = a_curr[i] + base.tmp_m2[i]
            end

            # Kinf = K⁽⁰⁾ = T A, stored as the gain during the diffuse period,
            # with Finf as the innovation covariance.
            mul!(K_t, base.Tmat, A)
            copyto!(view(Ft_view, 1:pt, 1:pt), Finf)

            # Pinf = T (Pinf - A Minf') T'
            copyto!(ws.tmp_mm_diff1, ws.Pinf)
            mul!(ws.tmp_mm_diff1, A, transpose(Minf), -one(T), one(T))
            mul!(base.tmp_mm2, base.Tmat, ws.tmp_mm_diff1)
            mul!(ws.Pinf, base.tmp_mm2, Tt)

            # Fstar = Z_t * Pstar * Z_t' + H_t
            Fstar = view(ws.tmp_pp_diff, 1:pt, 1:pt)
            mul!(ZrP, Zr, ws.Pstar)
            mul!(Fstar, ZrP, transpose(Zr))
            Fstar .+= Hr

            # Pstar_filt = Pstar - A Mstar' - Mstar A' + A Fstar A'
            Mstar = Minf                              # Minf is no longer needed
            mul!(Mstar, ws.Pstar, transpose(Zr))
            copyto!(Ptt_view, ws.Pstar)
            mul!(Ptt_view, A, transpose(Mstar), -one(T), one(T))
            mul!(Ptt_view, Mstar, transpose(A), -one(T), one(T))
            AF = Mstar                                # Mstar is no longer needed
            mul!(AF, A, Fstar)
            mul!(Ptt_view, AF, transpose(A), one(T), one(T))

            # Predict next: a = T * a_filt, Pstar = T * Pstar_filt * T' + RQR
            mul!(a_curr, base.Tmat, att_view)
            mul!(base.tmp_mm2, base.Tmat, Ptt_view)
            mul!(ws.Pstar, base.tmp_mm2, Tt)
            for j in 1:m, i in 1:m

                ws.Pstar[i, j] += base.RQR[i, j]
            end
        else
            # === Standard update on Pstar ===
            # Used after the diffuse period and, during it, whenever Finf is
            # zero; the period then contributes to the likelihood.
            base.n_obs_valid += 1
            base.n_obs_rows += pt

            # F = Z_t * Pstar * Z_t' + H_t
            mul!(ZrP, Zr, ws.Pstar)
            Fr = view(base.tmp_pp1, 1:pt, 1:pt)
            mul!(Fr, ZrP, transpose(Zr))
            for l in 1:pt, k in 1:pt

                Fr[k, l] += Hr[k, l]
                Ft_view[k, l] = Fr[k, l]
            end

            # Cholesky on Symmetric(:L) reads only the lower triangle.
            cholF = cholesky!(Symmetric(Fr, :L); check = false)
            if !issuccess(cholF)
                base.status = t
                base.loglik = T(-Inf)
                return base.loglik
            end
            L_lower = LowerTriangular(cholF.factors)
            for j in 1:pt, i in j:pt

                base.Ft_L[i, j, t] = L_lower[i, j]
            end

            # Log-likelihood
            logdetF = zero(T)
            for i in 1:pt
                logdetF += 2 * log(L_lower[i, i])
            end
            copyto!(tmp_p2r, v_t)
            ldiv!(L_lower, tmp_p2r)                  # L^{-1} v
            quad_form = zero(T)
            for i in 1:pt
                quad_form += tmp_p2r[i]^2
            end
            base.loglik += -T(0.5) * (logdetF + quad_form)

            # M = Pstar * Z_t' * F^{-1} (m×pt)
            Mr = view(base.tmp_mp, :, 1:pt)
            mul!(Mr, ws.Pstar, transpose(Zr))
            rdiv!(Mr, L_lower')
            rdiv!(Mr, L_lower)

            # K_t = T * M
            mul!(K_t, base.Tmat, Mr)

            # a_filt = a + M * v
            mul!(base.tmp_m2, Mr, v_t)
            for i in 1:m
                att_view[i] = a_curr[i] + base.tmp_m2[i]
            end

            # P_filt = Pstar - M * (Z_t * Pstar)
            mul!(base.tmp_mm2, Mr, ZrP)
            for j in 1:m, i in 1:m

                Ptt_view[i, j] = ws.Pstar[i, j] - base.tmp_mm2[i, j]
            end

            # Predict next: a = T * a_filt, Pstar = T * P_filt * T' + RQR
            mul!(a_curr, base.Tmat, att_view)
            mul!(base.tmp_mm2, base.Tmat, Ptt_view)
            mul!(ws.Pstar, base.tmp_mm2, Tt)
            for j in 1:m, i in 1:m

                ws.Pstar[i, j] += base.RQR[i, j]
            end

            if !ws.diffuse_ended
                # Pinf = T * Pinf * T'
                mul!(base.tmp_mm2, base.Tmat, ws.Pinf)
                mul!(ws.tmp_mm_diff2, base.tmp_mm2, Tt)
                copyto!(ws.Pinf, ws.tmp_mm_diff2)
            end
        end

        # Check if diffuse period ends
        if !ws.diffuse_ended && norm(ws.Pinf) <= tol
            ws.diffuse_ended = true
        end
    end

    # The Gaussian constant counts every observed scalar that contributed.
    base.loglik += -base.n_obs_rows * log2pi / 2
    base.status = 0

    return base.loglik
end

# ============================================
# EM Algorithm Workspace
# ============================================

"""
    EMWorkspace{T<:Real}

Pre-allocated workspace for EM algorithm sufficient statistics and updates.

For a state-space model:
    y_t = Z * α_t + ε_t,  ε_t ~ N(0, H)
    α_{t+1} = T * α_t + R * η_t,  η_t ~ N(0, Q)

The EM algorithm computes sufficient statistics from smoothed states:
    S_00 = Σ_{t=1}^{n-1} E[α_t α_t' | Y]
    S_11 = Σ_{t=2}^{n} E[α_t α_t' | Y]
    S_10 = Σ_{t=2}^{n} E[α_t α_{t-1}' | Y]

For the observation equation, summed over the observed periods only:
    S_yy = Σ_t y_t y_t'
    S_yα = Σ_t y_t E[α_t' | Y]
    S_αα = Σ_t E[α_t α_t' | Y]

# Free-cell masks
The masks select which cells the M-step updates; every other cell keeps its
value. A default-constructed workspace frees every cell of `Z`, `H`, `T` and
`Q` and keeps the initial state fixed. `set_em_masks!` sets the masks,
and the tie groups below, from an `SSMSpec`; `fit!(EM(), ...)` calls it.

- `Z_free`, `H_free`, `T_free`, `Q_free`: cells to estimate.
- `H_diag_only`: `H` is diagonal; only its diagonal is updated.
- `a1_free`, `P1_free`: initial-state cells to estimate.
- `Z_ties`, `H_ties`, `T_ties`, `Q_ties`: groups of free cells that share one
  value. Cells in a group stay equal after every M-step. `H` and `Q` groups must
  lie on the diagonal of a diagonal block.
"""
mutable struct EMWorkspace{T <: Real}
    # Dimensions
    obs_dim::Int      # p
    state_dim::Int    # m
    shock_dim::Int    # r
    n_times::Int      # n

    # Sufficient statistics for state equation (m × m)
    S_00::Matrix{T}   # Σ E[α_t α_t' | Y] for t = 1:n-1
    S_11::Matrix{T}   # Σ E[α_t α_t' | Y] for t = 2:n
    S_10::Matrix{T}   # Σ E[α_t α_{t-1}' | Y] for t = 2:n

    # Sufficient statistics for observation equation
    S_yy::Matrix{T}   # Σ y_t y_t' over observed t (p × p)
    S_yα::Matrix{T}   # Σ y_t α_t' over observed t (p × m)
    S_αα::Matrix{T}   # Σ E[α_t α_t' | Y] over observed t (m × m)

    # Scratch space for M-step computations
    tmp_mm1::Matrix{T}  # m × m
    tmp_mm2::Matrix{T}  # m × m
    tmp_pp1::Matrix{T}  # p × p
    tmp_pm::Matrix{T}   # p × m
    tmp_rr::Matrix{T}   # r × r
    tmp_mr::Matrix{T}   # m × r
    tmp_rm::Matrix{T}   # r × m

    # For dynamic factor models: masks for constrained estimation
    # These indicate which parameters are free vs fixed
    Z_free::BitMatrix      # p × m: which Z elements to estimate
    H_free::BitMatrix      # p × p: which H elements to estimate
    H_diag_only::Bool      # If true, H is diagonal
    T_free::BitMatrix      # m × m: which T elements to estimate
    Q_free::BitMatrix      # r × r: which Q elements to estimate

    # Initial-state masks. The M-step sets a₁ ← E[α₁|y] and P₁ ← Var[α₁|y]
    # on the free cells.
    a1_free::BitVector     # m: which a1 entries to estimate
    P1_free::BitMatrix     # m × m: which P1 entries to estimate

    # Degeneracy freeze masks (used by `allow_degen` path). When `H_zero[i]` is
    # true the EM treats `H[i, i]` as structurally zero: the M-step skips it
    # and the Kalman filter sees a zero variance for observation `i`. Same for
    # `Q_zero[i]`. These flips are committed in `_try_degenerate!`, which
    # tentatively zeros each below-threshold cell, runs a Kalman pass, and
    # commits only if loglik does not decrease. Mirrors MARSS `degen.test`.
    H_zero::BitVector      # p: which H diagonal entries are frozen at 0
    Q_zero::BitVector      # r: which Q diagonal entries are frozen at 0

    # Groups of free cells that share one parameter.
    Z_ties::Vector{Vector{CartesianIndex{2}}}
    H_ties::Vector{Vector{CartesianIndex{2}}}
    T_ties::Vector{Vector{CartesianIndex{2}}}
    Q_ties::Vector{Vector{CartesianIndex{2}}}
end

"""
    EMWorkspace(p::Int, m::Int, r::Int, n::Int, ::Type{T}=Float64) where T

Create EM workspace with given dimensions.
"""
function EMWorkspace(p::Int, m::Int, r::Int, n::Int, ::Type{T} = Float64) where {T}
    EMWorkspace{T}(
        p,
        m,
        r,
        n,
        # Sufficient statistics
        zeros(T, m, m),  # S_00
        zeros(T, m, m),  # S_11
        zeros(T, m, m),  # S_10
        zeros(T, p, p),  # S_yy
        zeros(T, p, m),  # S_yα
        zeros(T, m, m),  # S_αα
        # Scratch space
        zeros(T, m, m),
        zeros(T, m, m),
        zeros(T, p, p),
        zeros(T, p, m),
        zeros(T, r, r),
        zeros(T, m, r),
        zeros(T, r, m),
        # Default: all parameters free
        trues(p, m),     # Z_free
        trues(p, p),     # H_free
        false,           # H_diag_only
        trues(m, m),     # T_free
        trues(r, r),     # Q_free
        # Default: initial state fixed.
        falses(m),       # a1_free
        falses(m, m),    # P1_free
        # Default: no entries frozen at zero — this is the existing behaviour.
        # The opt-in `allow_degen` path in `fit!(EM(), …)` flips these per cell.
        falses(p),       # H_zero
        falses(r),       # Q_zero
        Vector{CartesianIndex{2}}[],  # Z_ties
        Vector{CartesianIndex{2}}[],  # H_ties
        Vector{CartesianIndex{2}}[],  # T_ties
        Vector{CartesianIndex{2}}[]   # Q_ties
    )
end

"""
    EMWorkspace(kf_ws::KalmanWorkspace{T}) where T

Create EM workspace matching dimensions of Kalman workspace.
"""
function EMWorkspace(kf_ws::KalmanWorkspace{T}) where {T}
    EMWorkspace(kf_ws.obs_dim, kf_ws.state_dim, kf_ws.shock_dim, kf_ws.n_times, T)
end

# ============================================
# E-step: Compute sufficient statistics
# ============================================

"""
    compute_sufficient_stats!(em_ws::EMWorkspace, kf_ws::KalmanWorkspace, y::AbstractMatrix) -> n_valid

Compute the sufficient statistics of the M-step from the smoother output and
return the number of observed periods. The observation-equation sums `S_yy`,
`S_yα` and `S_αα` run over the observed periods only, so the `Z` and `H` updates
stay unbiased when whole periods are missing. Requires
`kalman_smoother!(kf_ws; crosscov=true)` to have run on the current filter
output, and throws an `ArgumentError` otherwise.
"""
function compute_sufficient_stats!(
        em_ws::EMWorkspace{T},
        kf_ws::KalmanWorkspace{T},
        y::AbstractMatrix
) where {T}
    # Checks that the smoother has run with the cross-lag covariances.
    crosslag_covs(kf_ws)
    Base.require_one_based_indexing(y)
    p, m, n = em_ws.obs_dim, em_ws.state_dim, em_ws.n_times
    size(y) == (p, n) ||
        throw(DimensionMismatch("y must be $p × $n, got $(size(y))"))

    fill!(em_ws.S_00, zero(T))
    fill!(em_ws.S_11, zero(T))
    fill!(em_ws.S_10, zero(T))
    fill!(em_ws.S_yy, zero(T))
    fill!(em_ws.S_yα, zero(T))
    fill!(em_ws.S_αα, zero(T))

    n_valid = 0

    for t in 1:n
        α_t = view(kf_ws.αs, :, t)       # E[α_t | Y]
        V_t = view(kf_ws.Vs, :, :, t)    # Var[α_t | Y]
        observed = !kf_ws.missing_mask[t]
        in_00 = t < n                    # S_00 sums t = 1:n-1
        in_11 = t > 1                    # S_11 sums t = 2:n

        # E[α_t α_t'|Y] = V_t + α_t α_t', read once per t.
        for j in 1:m, i in 1:m

            moment = V_t[i, j] + α_t[i] * α_t[j]
            observed && (em_ws.S_αα[i, j] += moment)
            in_00 && (em_ws.S_00[i, j] += moment)
            in_11 && (em_ws.S_11[i, j] += moment)
        end

        # S_10: Σ_{t=2}^{n} E[α_t α_{t-1}'|Y] = Σ (Pcross_{t-1} + α_t α_{t-1}').
        if t > 1
            α_tm1 = view(kf_ws.αs, :, t - 1)
            Pcross_tm1 = view(kf_ws.Pcross, :, :, t - 1)
            for j in 1:m, i in 1:m

                em_ws.S_10[i, j] += Pcross_tm1[i, j] + α_t[i] * α_tm1[j]
            end
        end

        if observed
            n_valid += 1
            y_t = view(y, :, t)
            for j in 1:p, i in 1:p

                em_ws.S_yy[i, j] += y_t[i] * y_t[j]
            end
            for j in 1:m, i in 1:p

                em_ws.S_yα[i, j] += y_t[i] * α_t[j]
            end
        end
    end

    return n_valid
end

# ============================================
# M-step: Update parameters
# ============================================

"""
    update_Z!(kf_ws::KalmanWorkspace, em_ws::EMWorkspace, n_valid::Int)

M-step update for observation matrix Z.
Z_new = S_yα * S_αα^{-1}

Only updates elements where em_ws.Z_free is true. When some cells are fixed,
the update is the restricted MLE, row by row: with `J` the free and `F` the
fixed columns of row `i`,

    Z[i, J] = (S_yα[i, J] - Z[i, F] S_αα[F, J]) * inv(S_αα[J, J]).

This is exact when `H` is diagonal or all rows share the same free columns.
When cells are tied (`em_ws.Z_ties`), or `H` is not diagonal and the rows have
different free columns, the update is the generalized least-squares solution
with weight `H⁻¹` (see `_update_gls!`).
"""
function update_Z!(kf_ws::KalmanWorkspace{T}, em_ws::EMWorkspace{T}, n_valid::Int) where {T}
    p, m = em_ws.obs_dim, em_ws.state_dim

    if _needs_gls(em_ws.Z_free, em_ws.Z_ties, kf_ws.H)
        _update_gls!(kf_ws.Z, em_ws.Z_free, em_ws.Z_ties, kf_ws.H,
            em_ws.S_αα, em_ws.S_yα, "Z", "H") && return nothing
    end
    all(em_ws.Z_free) || return _update_Z_restricted!(kf_ws, em_ws)

    # Compute S_αα^{-1} (use only valid observations portion)
    # tmp_mm1 = S_αα
    copyto!(em_ws.tmp_mm1, em_ws.S_αα)

    # Regularize for numerical stability
    for i in 1:m
        em_ws.tmp_mm1[i, i] += T(1e-10)
    end

    # Cholesky factorization
    chol = cholesky!(Symmetric(em_ws.tmp_mm1, :L))

    # tmp_pm = S_yα * S_αα^{-1}
    # Solve S_αα * X' = S_yα' for X', then X = (X')'
    # Or: (S_yα * S_αα^{-1})' = S_αα^{-1} * S_yα'
    # Solve S_αα * Y = S_yα' for Y, then Z_new = Y'

    # tmp_pm = S_yα (will become Z_new after solving)
    copyto!(em_ws.tmp_pm, em_ws.S_yα)

    # Solve: need S_yα * S_αα^{-1}
    # S_yα is p × m, S_αα is m × m
    # Result is p × m
    # Use rdiv!: tmp_pm := tmp_pm * S_αα^{-1}
    L_lower = LowerTriangular(chol.L)
    rdiv!(em_ws.tmp_pm, L_lower')
    rdiv!(em_ws.tmp_pm, L_lower)

    # Update only free elements
    @inbounds for j in 1:m, i in 1:p

        if em_ws.Z_free[i, j]
            kf_ws.Z[i, j] = em_ws.tmp_pm[i, j]
        end
    end

    return nothing
end

# Restricted row-wise Z update for a partially free Z (see update_Z!). Rows that
# share the previous row's free pattern reuse its Cholesky factor.
function _update_Z_restricted!(kf_ws::KalmanWorkspace{T}, em_ws::EMWorkspace{T}) where {T}
    p, m = em_ws.obs_dim, em_ws.state_dim
    S_αα, S_yα, Z = em_ws.S_αα, em_ws.S_yα, kf_ws.Z

    J = Int[]
    chol = nothing
    rhs = T[]
    @inbounds for i in 1:p
        free_i = view(em_ws.Z_free, i, :)
        any(free_i) || continue
        if chol === nothing || !(length(J) == count(free_i) && all(j -> free_i[j], J))
            J = findall(free_i)
            S_JJ = S_αα[J, J]
            for d in eachindex(J)
                S_JJ[d, d] += T(1e-10)
            end
            chol = cholesky!(Symmetric(S_JJ, :L))
            rhs = Vector{T}(undef, length(J))
        end
        # rhs = S_yα[i, J] - Z[i, F] S_αα[F, J]
        for (jj, j) in enumerate(J)
            s = S_yα[i, j]
            for f in 1:m
                if !free_i[f] && !iszero(Z[i, f])
                    s -= Z[i, f] * S_αα[f, j]
                end
            end
            rhs[jj] = s
        end
        ldiv!(chol, rhs)
        for (jj, j) in enumerate(J)
            Z[i, j] = rhs[jj]
        end
    end

    return nothing
end

# True when the row-wise update of a partially free coefficient matrix is not
# the restricted maximizer: cells are tied, or the rows have different free
# columns and the weight matrix `W` couples the rows.
function _needs_gls(free::AbstractMatrix{Bool}, ties, W::AbstractMatrix)
    isempty(ties) || return true
    all(free) && return false
    common = all(i -> view(free, i, :) == view(free, 1, :), axes(free, 1))
    return !common && !isdiag(W)
end

# Restricted M-step for a coefficient matrix `X` (`Z` or `T`). It minimizes
#
#     tr(W⁻¹ (X S Xᵀ - C Xᵀ - X Cᵀ))
#
# over the free cells of `X`, with the cells in each tie group sharing one value
# and every other cell held at its current value. For `Z`: W = H, S = S_αα,
# C = S_yα. For `T`: W = R Q Rᵀ, S = S_00, C = S_10. Setting the derivative with
# respect to each group value βₖ (cells Gₖ) to zero gives the linear system
#
#     Σₗ A[k, l] βₗ = Σ_{(i,j)∈Gₖ} [W⁻¹ (C - F S)][i, j],
#     A[k, l] = Σ_{(i,j)∈Gₖ} Σ_{(a,b)∈Gₗ} W⁻¹[i, a] S[b, j],
#
# where F is X with its free cells set to zero. Returns false, leaving X
# unchanged, when W is singular and no cells are tied; a singular W with tied
# cells throws, because no exact update exists.
function _update_gls!(X::AbstractMatrix{T}, free::AbstractMatrix{Bool}, ties,
        W::AbstractMatrix, S::AbstractMatrix, C::AbstractMatrix,
        xname::AbstractString, wname::AbstractString) where {T}
    Wchol = cholesky(Symmetric(W), check = false)
    if !issuccess(Wchol)
        isempty(ties) && return false
        throw(ArgumentError(
            "EM cannot update the tied cells of $xname: $wname is singular. " *
            "Use fit!(MLE(), ...)."))
    end
    Winv = inv(Wchol)

    tied = Set{CartesianIndex{2}}()
    groups = Vector{Vector{CartesianIndex{2}}}()
    for g in ties
        push!(groups, g)
        union!(tied, g)
    end
    for I in CartesianIndices(free)
        free[I] && !(I in tied) && push!(groups, [I])
    end
    K = length(groups)
    K == 0 && return true

    F = copy(X)
    for g in groups, I in g

        F[I] = zero(T)
    end
    G = Winv * (C - F * S)

    A = zeros(T, K, K)
    b = zeros(T, K)
    for k in 1:K
        for I in groups[k]
            b[k] += G[I]
        end
        for l in k:K
            a = zero(T)
            for I in groups[k], J in groups[l]

                a += Winv[I[1], J[1]] * S[J[2], I[2]]
            end
            A[k, l] = a
            A[l, k] = a
        end
        A[k, k] += T(1e-10)
    end
    β = cholesky!(Symmetric(A)) \ b

    for k in 1:K, I in groups[k]

        X[I] = β[k]
    end
    return true
end

# Set every cell of each tie group to the group mean. For a diagonal covariance
# block whose tied cells lie on the diagonal, the mean of the per-cell M-step
# values is the restricted maximizer.
function _average_ties!(X::AbstractMatrix, ties)
    for g in ties
        v = sum(I -> X[I], g) / length(g)
        for I in g
            X[I] = v
        end
    end
    return X
end

"""
    update_H!(kf_ws::KalmanWorkspace, em_ws::EMWorkspace, n_valid::Int)

M-step update for observation covariance H. The Shumway–Stoffer formula is

    H_new = (1/n) Σ_t [(y_t - Z α̂_t)(y_t - Z α̂_t)' + Z V̂_t Z']
          = (1/n) [S_yy - Z S_yα' - S_yα Z' + Z S_αα Z'],

where `S_αα = Σ_t E[α_t α_t' | Y]` includes the smoother variance term
`Σ_t V̂_t`. The four-term form is required: the shorter `S_yy - Z S_yα'`
holds only when `Z` sits at its unrestricted optimum `S_yα S_αα⁻¹`, and gives
negative variances when `Z` is fixed or partially free.

Honours `em_ws.H_free` (cells to update), `em_ws.H_diag_only` (skips
off-diagonal updates),
`em_ws.H_zero` (per-cell freezes from the `allow_degen` path) and
`em_ws.H_ties` (diagonal cells that share one variance).
"""
function update_H!(kf_ws::KalmanWorkspace{T}, em_ws::EMWorkspace{T}, n_valid::Int) where {T}
    p, m = em_ws.obs_dim, em_ws.state_dim

    # tmp_pp1 = Z * S_yα'                              (p × p)
    mul!(em_ws.tmp_pp1, kf_ws.Z, em_ws.S_yα')

    # tmp_pm = Z * S_αα                                (p × m)
    mul!(em_ws.tmp_pm, kf_ws.Z, em_ws.S_αα)

    scale = one(T) / n_valid

    if em_ws.H_diag_only
        # Diagonal entries only: avoid the full p × p matmul for Z S_αα Z'.
        @inbounds for i in 1:p
            em_ws.H_free[i, i] || continue
            if em_ws.H_zero[i]
                kf_ws.H[i, i] = zero(T)   # frozen at 0 by allow_degen path
                continue
            end
            # (Z S_αα Z')[i, i] = Σ_k tmp_pm[i, k] · Z[i, k]
            zsz_ii = zero(T)
            @simd for k in 1:m
                zsz_ii += em_ws.tmp_pm[i, k] * kf_ws.Z[i, k]
            end
            # H[i,i] = (S_yy[i,i] - 2 (Z S_yα')[i,i] + (Z S_αα Z')[i,i]) / n
            h_ii = (em_ws.S_yy[i, i] - T(2) * em_ws.tmp_pp1[i, i] + zsz_ii) * scale
            kf_ws.H[i, i] = max(h_ii, T(1e-10))
        end
    else
        # Full p × p update. tmp_pp1 currently holds Z * S_yα'; we need
        # H_new = (S_yy - Z S_yα' - (Z S_yα')' + Z S_αα Z') / n.
        # Compute Z S_αα Z' on top of tmp_pp1 - the Z S_yα' contributions:
        # Use a fresh accumulation to avoid clobbering tmp_pp1 mid-computation.
        # H_new = S_yy - Z S_yα' - (Z S_yα')' + (Z S_αα) Z'
        @inbounds for j in 1:p, i in 1:p

            em_ws.H_free[i, j] || continue
            zsz_ij = zero(T)
            @simd for k in 1:m
                zsz_ij += em_ws.tmp_pm[i, k] * kf_ws.Z[j, k]
            end
            kf_ws.H[i, j] = (em_ws.S_yy[i, j] -
                             em_ws.tmp_pp1[i, j] - em_ws.tmp_pp1[j, i] +
                             zsz_ij) * scale
        end
        # Symmetrise (avg upper/lower triangle to clean up FP noise).
        for j in 1:p, i in 1:(j - 1)

            avg = (kf_ws.H[i, j] + kf_ws.H[j, i]) / 2
            kf_ws.H[i, j] = avg
            kf_ws.H[j, i] = avg
        end
        # Floor diagonals (skip frozen ones).
        @inbounds for i in 1:p
            em_ws.H_free[i, i] || continue
            if em_ws.H_zero[i]
                kf_ws.H[i, i] = zero(T)
                continue
            end
            kf_ws.H[i, i] = max(kf_ws.H[i, i], T(1e-10))
        end
    end
    _average_ties!(kf_ws.H, em_ws.H_ties)

    return nothing
end

"""
    update_T!(kf_ws::KalmanWorkspace, em_ws::EMWorkspace)

M-step update for state transition matrix T (restricted MLE).

For an unconstrained `T` the M-step closed form is `T = S_10 / S_00`. When some
cells of `T` are structurally fixed (e.g. diagonal-only `T`, or any pattern
encoded by `em_ws.T_free`), the restricted-MLE update is **row-wise**:
for each row `i`, with `J_i = {j : T_free[i, j]}`,

    T[i, J_i] = S_10[i, J_i] * inv(S_00[J_i, J_i]).

If row `i` has no free cells it is left untouched. The row-wise update is exact
when `R Q Rᵀ` is diagonal or every row has the same free columns. When cells are
tied (`em_ws.T_ties`), or `R Q Rᵀ` couples rows with different free columns,
the update is the generalized least-squares solution with weight `(R Q Rᵀ)⁻¹`
(see `_update_gls!`); a singular `R Q Rᵀ` keeps the row-wise update.
"""
function update_T!(kf_ws::KalmanWorkspace{T}, em_ws::EMWorkspace{T}) where {T}
    m = em_ws.state_dim

    if _needs_gls(em_ws.T_free, em_ws.T_ties, kf_ws.RQR)
        _update_gls!(kf_ws.Tmat, em_ws.T_free, em_ws.T_ties, kf_ws.RQR,
            em_ws.S_00, em_ws.S_10, "T", "R Q Rᵀ") && return nothing
    end

    # Detect whether all rows share the same free-column pattern. Common cases:
    # all-true (unrestricted), diagonal-only, lower-triangular, fixed sub-block.
    common_pattern = true
    @inbounds for i in 2:m, j in 1:m

        if em_ws.T_free[i, j] != em_ws.T_free[1, j]
            common_pattern = false
            break
        end
    end

    if common_pattern
        # Same J for every row → solve once for all rows.
        J = Int[]
        @inbounds for j in 1:m
            em_ws.T_free[1, j] && push!(J, j)
        end
        isempty(J) && return nothing
        if length(J) == m
            # Fast unrestricted path — bit-for-bit identical to the original code.
            copyto!(em_ws.tmp_mm1, em_ws.S_00)
            @inbounds for i in 1:m
                em_ws.tmp_mm1[i, i] += T(1e-10)
            end
            chol = cholesky!(Symmetric(em_ws.tmp_mm1, :L))
            copyto!(em_ws.tmp_mm2, em_ws.S_10)
            L_lower = LowerTriangular(chol.L)
            rdiv!(em_ws.tmp_mm2, L_lower')
            rdiv!(em_ws.tmp_mm2, L_lower)
            @inbounds for j in 1:m, i in 1:m

                if em_ws.T_free[i, j]
                    kf_ws.Tmat[i, j] = em_ws.tmp_mm2[i, j]
                end
            end
            return nothing
        end
        # Sub-block solve: build S_00[J, J] and S_10[:, J], one Cholesky for all rows.
        k = length(J)
        S00_sub = Matrix{T}(undef, k, k)
        S10_sub = Matrix{T}(undef, m, k)
        @inbounds for jj in 1:k, ii in 1:k

            S00_sub[ii, jj] = em_ws.S_00[J[ii], J[jj]]
        end
        @inbounds for i in 1:k
            S00_sub[i, i] += T(1e-10)
        end
        @inbounds for jj in 1:k, ii in 1:m

            S10_sub[ii, jj] = em_ws.S_10[ii, J[jj]]
        end
        chol = cholesky!(Symmetric(S00_sub, :L))
        L_lower = LowerTriangular(chol.L)
        rdiv!(S10_sub, L_lower')
        rdiv!(S10_sub, L_lower)
        @inbounds for jj in 1:k, i in 1:m

            if em_ws.T_free[i, J[jj]]
                kf_ws.Tmat[i, J[jj]] = S10_sub[i, jj]
            end
        end
        return nothing
    end

    # General path: rows have different free patterns → restricted OLS per row.
    Jbuf = Vector{Int}(undef, m)
    @inbounds for i in 1:m
        nj = 0
        for j in 1:m
            if em_ws.T_free[i, j]
                nj += 1
                Jbuf[nj] = j
            end
        end
        nj == 0 && continue
        S00_i = Matrix{T}(undef, nj, nj)
        rhs_i = Vector{T}(undef, nj)
        for jj in 1:nj
            for ii in 1:nj
                S00_i[ii, jj] = em_ws.S_00[Jbuf[ii], Jbuf[jj]]
            end
            S00_i[jj, jj] += T(1e-10)
            rhs_i[jj] = em_ws.S_10[i, Jbuf[jj]]
        end
        sol = Symmetric(S00_i, :L) \ rhs_i
        for jj in 1:nj
            kf_ws.Tmat[i, Jbuf[jj]] = sol[jj]
        end
    end

    return nothing
end

"""
    update_Q!(kf_ws::KalmanWorkspace, em_ws::EMWorkspace)

M-step update for state covariance Q.
Q_new = (1/(n-1)) * R' * (S_11 - T * S_10' - S_10 * T' + T * S_00 * T') * R

For the case R = [I; 0] (shocks affect first r states only):
Q_new = (1/(n-1)) * (S_11[1:r,1:r] - T[1:r,:] * S_10'[:,1:r] - ...)

Simplified when R = I (r = m):
Q_new = (1/(n-1)) * (S_11 - T * S_10' - S_10 * T' + T * S_00 * T')
"""
function update_Q!(kf_ws::KalmanWorkspace{T}, em_ws::EMWorkspace{T}) where {T}
    m, r = em_ws.state_dim, em_ws.shock_dim
    n = em_ws.n_times

    # Compute: S_11 - T * S_10' - S_10 * T' + T * S_00 * T'
    # = S_11 - S_10 * T' - T * S_10' + T * S_00 * T'

    # tmp_mm1 = T * S_00
    mul!(em_ws.tmp_mm1, kf_ws.Tmat, em_ws.S_00)

    # tmp_mm2 = T * S_00 * T'
    mul!(em_ws.tmp_mm2, em_ws.tmp_mm1, kf_ws.Tmat')

    # tmp_mm2 += S_11
    for j in 1:m, i in 1:m

        em_ws.tmp_mm2[i, j] += em_ws.S_11[i, j]
    end

    # tmp_mm1 = S_10 * T'
    mul!(em_ws.tmp_mm1, em_ws.S_10, kf_ws.Tmat')

    # tmp_mm2 -= S_10 * T' + T * S_10' = S_10 * T' + (S_10 * T')'
    for j in 1:m, i in 1:m

        em_ws.tmp_mm2[i, j] -= em_ws.tmp_mm1[i, j] + em_ws.tmp_mm1[j, i]
    end

    # Scale by 1/(n-1)
    scale = one(T) / (n - 1)

    # Now compute R' * tmp_mm2 * R to get Q
    # For general R (m × r): Q = R' * tmp_mm2 * R
    if r == m
        # R = I case: Q = tmp_mm2
        @inbounds for j in 1:r, i in 1:r

            if em_ws.Q_free[i, j]
                kf_ws.Q[i, j] = em_ws.tmp_mm2[i, j] * scale
            end
        end
    else
        # General case: Q = R' * tmp_mm2 * R
        # tmp_mr = tmp_mm2 * R
        mul!(em_ws.tmp_mr, em_ws.tmp_mm2, kf_ws.R)
        # tmp_rr = R' * tmp_mr
        mul!(em_ws.tmp_rr, kf_ws.R', em_ws.tmp_mr)

        @inbounds for j in 1:r, i in 1:r

            if em_ws.Q_free[i, j]
                kf_ws.Q[i, j] = em_ws.tmp_rr[i, j] * scale
            end
        end
    end

    # Ensure symmetry and positive definiteness; honour Q_zero freezes from
    # the allow_degen path. A frozen diagonal entry forces the entire
    # corresponding row and column of Q to zero (the shock is structurally
    # absent, not just smaller than the floor).
    for j in 1:r, i in 1:(j - 1)

        avg = (kf_ws.Q[i, j] + kf_ws.Q[j, i]) / 2
        kf_ws.Q[i, j] = avg
        kf_ws.Q[j, i] = avg
    end
    @inbounds for i in 1:r
        if em_ws.Q_zero[i]
            kf_ws.Q[i, i] = zero(T)
            for j in 1:r
                if j != i
                    kf_ws.Q[i, j] = zero(T)
                    kf_ws.Q[j, i] = zero(T)
                end
            end
        else
            kf_ws.Q[i, i] = max(kf_ws.Q[i, i], T(1e-10))
        end
    end
    _average_ties!(kf_ws.Q, em_ws.Q_ties)

    # Update RQR
    _update_RQR!(kf_ws)

    return nothing
end

"""
    update_initial_state!(kf_ws::KalmanWorkspace, em_ws::EMWorkspace)

M-step update for the initial-state distribution `(a₁, P₁)`.

For a linear-Gaussian state-space model the closed-form M-step gives
    a₁_new = E[α₁ | y₁:n]      (= `kf_ws.αs[:, 1]`)
    P₁_new = Var[α₁ | y₁:n]    (= `kf_ws.Vs[:, :, 1]`).

Only entries marked as free in `em_ws.a1_free` / `em_ws.P1_free` are
written; entries flagged false retain their value, mirroring the way
`update_Z!` / `update_T!` / `update_Q!` honour their per-cell masks.

Assumes `kalman_smoother!(kf_ws)` has populated the workspace's smoothed
state mean and covariance.
"""
function update_initial_state!(kf_ws::KalmanWorkspace{T}, em_ws::EMWorkspace{T}) where {T}
    m = em_ws.state_dim

    @inbounds for i in 1:m
        if em_ws.a1_free[i]
            kf_ws.a1[i] = kf_ws.αs[i, 1]
        end
    end

    # Updating individual P₁ cells from a generally-non-PSD subset would risk
    # producing a non-PSD P₁ and breaking the next iteration's Cholesky. When
    # any cell is free, snapshot the smoothed Var[α₁|y] (which is PSD by
    # construction) into the free cells; symmetry is preserved because Vs is
    # symmetric.
    if any(em_ws.P1_free)
        @inbounds for j in 1:m, i in 1:m

            if em_ws.P1_free[i, j]
                kf_ws.P1[i, j] = kf_ws.Vs[i, j, 1]
            end
        end
        # Defensive symmetrisation — if the user supplies an asymmetric mask
        # (free upper triangle but fixed lower triangle, etc.) without ties,
        # the lower/upper halves can drift apart by 1 ULP. Average the off-
        # diagonals so subsequent Cholesky factorizations stay numerical.
        @inbounds for j in 1:m
            for i in 1:(j - 1)
                avg = (kf_ws.P1[i, j] + kf_ws.P1[j, i]) / 2
                kf_ws.P1[i, j] = avg
                kf_ws.P1[j, i] = avg
            end
        end
    end

    return nothing
end

# ============================================
# Main EM Algorithm
# ============================================

"""
    EMResult{T}

Result of an EM run: the system matrices at the last iterate, with convergence
information.

# Fields
- `converged::Bool`: the relative change in the log-likelihood fell below `tol`.
- `iterations::Int`: number of E-steps run.
- `loglik::T`: log-likelihood at the returned parameters.
- `loglik_history::Vector{T}`: log-likelihood at each iteration.
- `Z`, `H`, `T`, `R`, `Q`: system matrices.
- `a1`, `P1`: initial state mean and covariance.
"""
struct EMResult{T <: Real}
    converged::Bool
    iterations::Int
    loglik::T
    loglik_history::Vector{T}
    Z::Matrix{T}
    H::Matrix{T}
    T::Matrix{T}
    R::Matrix{T}
    Q::Matrix{T}
    a1::Vector{T}
    P1::Matrix{T}
end

function Base.show(io::IO, r::EMResult)
    p, m = size(r.Z)
    status = r.converged ? "converged" : "not converged"
    print(io, "EMResult(p = $p, m = $m, $status after $(r.iterations) iterations, ",
        "loglik = $(round(r.loglik; sigdigits = 8)))")
end

"""
    em_estimate!(kf_ws::KalmanWorkspace, em_ws::EMWorkspace, y::AbstractMatrix;
                 maxiter::Int=500, tol::Real=1e-6, verbose::Bool=false, warn::Bool=true,
                 estimate_Z::Bool=true, estimate_H::Bool=true,
                 estimate_T::Bool=true, estimate_Q::Bool=true,
                 estimate_initial::Bool=true) -> EMResult

Run the EM algorithm on the parameters held in `kf_ws`, updating them in place.
The masks and tie groups in `em_ws` select the cells to estimate; see
`EMWorkspace` and `set_em_masks!`. `fit!(EM(), model, y)` is the
model-level entry point.

Periods with some but not all observations missing are not supported and throw
an `ArgumentError`; whole missing periods are fine.

# Arguments
- `kf_ws`: Kalman filter/smoother workspace (contains initial parameters)
- `em_ws`: EM algorithm workspace
- `y`: Observations (p × n matrix), missing values as NaN

# Keyword Arguments
- `maxiter`: Maximum number of EM iterations (default: 500)
- `tol`: Convergence tolerance for log-likelihood change (default: 1e-6)
- `verbose`: Log progress with `@info` (default: false)
- `warn`: Emit a warning when EM stops without converging (default: true)
- `estimate_Z`: Update observation matrix Z (default: true)
- `estimate_H`: Update observation covariance H (default: true)
- `estimate_T`: Update transition matrix T (default: true)
- `estimate_Q`: Update state covariance Q (default: true)
- `estimate_initial`: Update `(a₁, P₁)` from the smoother whenever
  `em_ws.a1_free` / `em_ws.P1_free` mark cells as free (default: true).
  Cells with `a1_free[i] == false` (or `P1_free[i,j] == false`) stay
  fixed at their initial value. Both masks are all-false in a
  default-constructed `EMWorkspace`.

# Returns
- `EMResult` containing convergence info and estimated parameters
"""
function em_estimate!(
        kf_ws::KalmanWorkspace{T},
        em_ws::EMWorkspace{T},
        y::AbstractMatrix;
        maxiter::Int = 500,
        tol::Real = 1e-6,
        verbose::Bool = false,
        estimate_Z::Bool = true,
        estimate_H::Bool = true,
        estimate_T::Bool = true,
        estimate_Q::Bool = true,
        estimate_initial::Bool = true,
        allow_degen::Bool = false,
        degen_lim::Real = 1e-4,
        min_degen_iter::Int = 50,
        warn::Bool = true
) where {T}
    y = _nan_obs(y)
    _require_no_partial_periods(y, "em_estimate!", "Use fit!(MLE(), ...), which handles them.")
    loglik_history = Vector{T}(undef, maxiter)
    ll_prev = T(-Inf)
    converged = false
    iter = 0

    for i in 1:maxiter
        iter = i

        # E-step: Filter and smooth
        kalman_filter_and_smooth!(kf_ws, y; crosscov = true)
        ll = kf_ws.loglik
        loglik_history[i] = ll

        # A failed filter leaves nothing to compute the sufficient statistics from.
        kf_ws.status == 0 || break

        # Check convergence
        if i > 1
            ll_change = abs(ll - ll_prev)
            rel_change = ll_change / (abs(ll_prev) + T(1e-10))

            if verbose && (i % 10 == 0 || i <= 5)
                @info "EM iteration $i" loglik=ll change=ll_change rel_change
            end

            if rel_change < tol
                converged = true
                verbose && @info "EM converged at iteration $i" loglik = ll
                break
            end
        elseif verbose
            @info "EM iteration $i" loglik = ll
        end

        ll_prev = ll

        # E-step: Compute sufficient statistics
        n_valid = compute_sufficient_stats!(em_ws, kf_ws, y)

        # M-step: Update parameters
        if estimate_Z
            update_Z!(kf_ws, em_ws, n_valid)
        end
        if estimate_H
            update_H!(kf_ws, em_ws, n_valid)
        end
        if estimate_T
            update_T!(kf_ws, em_ws)
        end
        if estimate_Q
            update_Q!(kf_ws, em_ws)
        end
        if estimate_initial
            update_initial_state!(kf_ws, em_ws)
        end

        # Try freezing degenerate H/Q diagonal entries (mirror MARSS
        # `degen.test`). Only after the M-step has had time to converge
        # roughly, and only when the user has opted in via `allow_degen`.
        if allow_degen && i > min_degen_iter
            if estimate_H
                _try_degenerate!(kf_ws, em_ws, y; degen_lim = degen_lim, what = :H)
            end
            if estimate_Q
                _try_degenerate!(kf_ws, em_ws, y; degen_lim = degen_lim, what = :Q)
            end
        end
    end

    # When converged we broke out *before* the M-step ran, so kf_ws already
    # holds filter+smoother results consistent with the current parameters.
    # Only re-run when we exhausted maxiter without converging (params were
    # just updated and the workspace is stale).
    failed_at = kf_ws.status
    if !converged && failed_at == 0
        kalman_filter_and_smooth!(kf_ws, y; crosscov = true)
        failed_at = kf_ws.status
    end
    warn && _warn_em(converged, iter, failed_at, tol)

    return EMResult{T}(
        converged,
        iter,
        kf_ws.loglik,
        loglik_history[1:iter],
        copy(kf_ws.Z),
        copy(kf_ws.H),
        copy(kf_ws.Tmat),
        copy(kf_ws.R),
        copy(kf_ws.Q),
        copy(kf_ws.a1),
        copy(kf_ws.P1)
    )
end

# Warn that an EM run stopped without converging. `failed_at` is the period in
# which the last filter run failed, or 0.
function _warn_em(converged::Bool, iter::Integer, failed_at::Integer, tol::Real)
    converged && return nothing
    if failed_at > 0
        @warn "EM stopped at iteration $iter: $(_failure_message(failed_at)). " *
              "Try other starting values. Pass `warn = false` to silence this warning."
    else
        @warn "EM did not converge in $iter iterations (relative change in the " *
              "log-likelihood still above tol = $tol). Increase `maxiter` or pass " *
              "`warn = false` to silence this warning."
    end
    return nothing
end

# `_try_degenerate!(kf_ws, em_ws, y; degen_lim, what)` — internal helper for
# the `allow_degen` opt-in path of `em_estimate!`. Mirrors MARSS's
# `degen.test`: for each diagonal cell of `H` (when `what === :H`) or `Q`
# (when `what === :Q`) whose current value is below `degen_lim`, tentatively
# zero the cell, run `kalman_filter_and_smooth!`, and commit the freeze iff loglik
# is finite and does not decrease beyond `√eps`. On a failed freeze the prior
# value is restored. When any trial ran, the filter and smoother are re-run at
# the end so the workspace matches the committed parameters. Returns the number
# of cells frozen by this call.
function _try_degenerate!(
        kf_ws::KalmanWorkspace{T},
        em_ws::EMWorkspace{T},
        y::AbstractMatrix;
        degen_lim::Real,
        what::Symbol
) where {T}
    em_ws.H_diag_only || return 0   # only meaningful for diagonal H/Q
    n_frozen = 0
    n_trials = 0
    ll_old = kf_ws.loglik
    isfinite(ll_old) || return 0
    threshold = T(degen_lim)
    eps_tol = sqrt(eps(T))

    if what === :H
        p = em_ws.obs_dim
        for i in 1:p
            em_ws.H_zero[i] && continue
            h_save = kf_ws.H[i, i]
            h_save < threshold || continue
            kf_ws.H[i, i] = zero(T)
            n_trials += 1
            ll_new = kalman_filter_and_smooth!(kf_ws, y; crosscov = true)
            if isfinite(ll_new) && ll_new >= ll_old - eps_tol
                em_ws.H_zero[i] = true
                ll_old = ll_new
                n_frozen += 1
            else
                kf_ws.H[i, i] = h_save
            end
        end
    elseif what === :Q
        r = em_ws.shock_dim
        for i in 1:r
            em_ws.Q_zero[i] && continue
            q_save = kf_ws.Q[i, i]
            q_save < threshold || continue
            row_save = copy(kf_ws.Q[i, :])
            col_save = copy(kf_ws.Q[:, i])
            kf_ws.Q[i, i] = zero(T)
            for j in 1:r
                if j != i
                    kf_ws.Q[i, j] = zero(T)
                    kf_ws.Q[j, i] = zero(T)
                end
            end
            _update_RQR!(kf_ws)
            n_trials += 1
            ll_new = kalman_filter_and_smooth!(kf_ws, y; crosscov = true)
            if isfinite(ll_new) && ll_new >= ll_old - eps_tol
                em_ws.Q_zero[i] = true
                ll_old = ll_new
                n_frozen += 1
            else
                kf_ws.Q[i, :] .= row_save
                kf_ws.Q[:, i] .= col_save
                _update_RQR!(kf_ws)
            end
        end
    else
        throw(ArgumentError("`what` must be :H or :Q, got $what"))
    end

    # Re-sync the workspace with the committed parameters so subsequent M-step
    # calls see the up-to-date sufficient statistics.
    if n_trials > 0
        kalman_filter_and_smooth!(kf_ws, y; crosscov = true)
    end

    return n_frozen
end

# ============================================
# Estimation Method Types (StatsAPI-style)
# ============================================

"""
    EM

Estimation method type for EM (Expectation-Maximization) algorithm.
Used with `fit!(EM(), model, data; kwargs...)`.
"""
struct EM end

"""
    MLE

Estimation method type for direct Maximum Likelihood Estimation.
Used with `fit!(MLE(), model, data; kwargs...)`.
"""
struct MLE end

# ============================================
# Abstract State-Space Model
# ============================================

"""
    AbstractStateSpaceModel

Abstract supertype for all state-space models in Siphon.jl.
Provides a common interface for fitting, querying, and forecasting.

Subtypes:
- `StateSpaceModel` - General state-space model from SSMSpec
- `DynamicFactorModel` - Dynamic factor model with EM estimation
"""
abstract type AbstractStateSpaceModel end

# Every model stores `fitted`, `converged`, `iterations` and `loglik`.
_fit_hint(::AbstractStateSpaceModel) = "fit!(MLE(), model, y) or fit!(EM(), model, y)"

function _require_fitted(model::AbstractStateSpaceModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call $(_fit_hint(model)) first."))
    return nothing
end

"""
    isfitted(model::AbstractStateSpaceModel) -> Bool

Return whether the model has been fitted (or built with known parameters).
"""
isfitted(model::AbstractStateSpaceModel) = model.fitted

"""
    isconverged(model::AbstractStateSpaceModel) -> Bool

Return whether the estimation converged. Throws if the model is not fitted.
"""
isconverged(model::AbstractStateSpaceModel) = (_require_fitted(model); model.converged)

"""
    niterations(model::AbstractStateSpaceModel) -> Int

Return the number of estimation iterations. Throws if the model is not fitted.
"""
niterations(model::AbstractStateSpaceModel) = (_require_fitted(model); model.iterations)

loglikelihood(model::AbstractStateSpaceModel) = (_require_fitted(model); model.loglik)

# ============================================
# StateSpaceModel
# ============================================

"""
    StateSpaceModelWorkspaces{T}

Pre-allocated workspaces for in-place EM computation.
"""
struct StateSpaceModelWorkspaces{T <: Real}
    kf_ws::KalmanWorkspace{T}
    em_ws::EMWorkspace{T}
end

"""
    StateSpaceModel{T}

Mutable state-space model container wrapping an SSMSpec.

Supports both MLE and EM estimation.

# Usage
```julia
# Create from SSMSpec
spec = local_level(var_obs=:free, var_level=:free)
model = StateSpaceModel(spec, 200)

# Fit with MLE (pure Kalman filter, AD-compatible)
fit!(MLE(), model, y)

# Or fit with EM
fit!(EM(), model, y; maxiter=500, verbose=true)

# Access results
params = parameters(model)         # NamedTuple
f = filtered_states(model)         # m × n
s = smoothed_states(model)         # computed on-demand

# Forecast
fc = forecast(model, 10)
```

# Backend Selection for EM
- If max(n_states, n_obs) > 13: uses in-place KalmanWorkspace (zero allocations)
- Otherwise: uses pure StaticArrays implementation

# Fields (internal)
See source for field documentation.
"""
mutable struct StateSpaceModel{T <: Real} <: AbstractStateSpaceModel
    # Specification
    spec::SSMSpec
    n_times::Int

    # Fitted parameters
    theta_values::Vector{T}
    theta_fitted::Bool

    # Fit statistics
    loglik::T
    fitted::Bool
    converged::Bool
    iterations::Int

    # Filter results (pre-allocated)
    at::Matrix{T}           # Predicted states (m × n)
    Pt::Array{T, 3}          # Predicted covariances (m × m × n)
    att::Matrix{T}          # Filtered states (m × n)
    Ptt::Array{T, 3}         # Filtered covariances (m × m × n)
    vt::Matrix{T}           # Innovations (p × n)
    Ft::Array{T, 3}          # Innovation covariances (p × p × n)
    Kt::Array{T, 3}          # Kalman gains (m × p × n)
    missing_mask::BitVector
    observed_mask::BitMatrix  # p × n: true where y[i,t] is observed
    # _NOT_RUN, 0 after a successful filter run, or the period where it failed
    filter_status::Int

    # Smoother cache (lazily computed)
    smoother_computed::Bool
    smoothed_alpha::Matrix{T}
    smoothed_V::Array{T, 3}

    # In-place workspaces (for large EM)
    kf_workspace_allocated::Bool
    workspaces_ref::Base.RefValue{Any}
end

"""
    StateSpaceModel(spec::SSMSpec, n_times::Int; T::Type{<:Real}=Float64)

Construct a StateSpaceModel from an SSMSpec.

# Arguments
- `spec::SSMSpec`: Model specification from DSL (local_level, ar1, custom_ssm, etc.)
- `n_times::Int`: Number of time periods

# Keyword Arguments
- `T::Type{<:Real}`: Element type (default: Float64)

# Behavior
- Pre-allocates filter storage
- Pre-allocates the KalmanWorkspace and EMWorkspace used by `fit!(EM(), ...)`

# Example
```julia
spec = local_level(var_obs=:free, var_level=:free)
model = StateSpaceModel(spec, 200)
```
"""
function StateSpaceModel(spec::SSMSpec, n_times::Int; T::Type{<:Real} = Float64)
    m = spec.n_states
    p = spec.n_obs
    r = spec.n_shocks
    n = n_times

    # Pre-allocate filter storage
    at = zeros(T, m, n)
    Pt = zeros(T, m, m, n)
    att = zeros(T, m, n)
    Ptt = zeros(T, m, m, n)
    vt = zeros(T, p, n)
    Ft = zeros(T, p, p, n)
    Kt = zeros(T, m, p, n)
    missing_mask = falses(n)
    observed_mask = falses(p, n)

    # Smoother storage (allocated but not valid until computed)
    smoothed_alpha = zeros(T, m, n)
    smoothed_V = zeros(T, m, m, n)

    # Parameter storage
    n_params = length(spec.params)
    theta_values = zeros(T, n_params)

    # Always allocate the in-place Kalman/EM workspace. Benchmarks (see
    # benchmark/em_paths.jl) show in-place EM beats the pure-static path on
    # every model size, including m = p = 1, because the per-iteration
    # allocation cost of the result-returning kalman_filter swamps the
    # state-side static-arithmetic savings. The workspace is a one-time
    # allocation of order n × max(m², p²) words, negligible vs. the EM
    # iteration count.
    theta_init = [prm.init for prm in spec.params]
    names = Tuple(prm.name for prm in spec.params)
    theta_nt = NamedTuple{names}(Tuple(theta_init))

    kfparms = Siphon.build_kfparms(spec, theta_nt)
    a1_init, P1_init = Siphon.build_initial_state(spec, theta_nt)

    kf_ws = KalmanWorkspace(
        kfparms.Z,
        kfparms.H,
        kfparms.T,
        kfparms.R,
        kfparms.Q,
        a1_init,
        P1_init,
        n
    )
    em_ws = EMWorkspace(kf_ws)
    workspaces_ref = Ref{Any}(StateSpaceModelWorkspaces{T}(kf_ws, em_ws))

    StateSpaceModel{T}(
        spec,
        n,
        theta_values,
        false,  # theta not fitted
        T(-Inf),
        false,
        false,
        0,
        at,
        Pt,
        att,
        Ptt,
        vt,
        Ft,
        Kt,
        missing_mask,
        observed_mask,
        _NOT_RUN,  # filter_status
        false,
        smoothed_alpha,
        smoothed_V,
        true,  # kf_workspace_allocated: always true now
        workspaces_ref
    )
end

"""
    StateSpaceModel(spec::SSMSpec, θ::NamedTuple, n_times::Int; T::Type{<:Real}=Float64)

Construct a StateSpaceModel with known (externally provided) parameters.

# Arguments
- `spec::SSMSpec`: Model specification from DSL
- `θ::NamedTuple`: Parameter values (e.g., `(var_obs=100.0, var_level=50.0)`)
- `n_times::Int`: Number of time periods

# Behavior
- Creates model with parameters pre-set
- Sets `fitted=true`, `theta_fitted=true`, `converged=true`
- No estimation is performed; parameters are taken as given

# Example
```julia
spec = local_level()
θ = (var_obs = 100.0, var_level = 50.0)
model = StateSpaceModel(spec, θ, 200)
```
"""
function StateSpaceModel(spec::SSMSpec, θ::NamedTuple, n_times::Int; T::Type{<:Real} = Float64)
    # Create base model
    model = StateSpaceModel(spec, n_times; T = T)

    # Store parameters
    _ssm_store_theta!(model, θ)

    # Mark as fitted with external parameters
    model.fitted = true
    model.converged = true

    return model
end

"""
    StateSpaceModel(spec::SSMSpec, θ_vec::Vector, n_times::Int; T::Type{<:Real}=Float64)

Construct a StateSpaceModel with known parameters provided as a Vector.

Parameters are matched to spec.params in order.

# Example
```julia
spec = local_level()
θ_vec = [100.0, 50.0]  # [var_obs, var_level]
model = StateSpaceModel(spec, θ_vec, 200)
```
"""
function StateSpaceModel(spec::SSMSpec, θ_vec::AbstractVector{<:Real}, n_times::Int; T::Type{<:Real} = Float64)
    length(θ_vec) == length(spec.params) ||
        throw(ArgumentError("θ_vec length ($(length(θ_vec))) must match number of parameters ($(length(spec.params)))"))

    # Convert to NamedTuple
    names = Tuple(prm.name for prm in spec.params)
    θ = NamedTuple{names}(Tuple(θ_vec))

    return StateSpaceModel(spec, θ, n_times; T = T)
end

# ============================================
# StateSpaceModel Helper Functions
# ============================================

"""Store fitted parameters from NamedTuple to internal storage."""
function _ssm_store_theta!(model::StateSpaceModel{T}, theta::NamedTuple) where {T}
    for (i, prm) in enumerate(model.spec.params)
        model.theta_values[i] = T(getproperty(theta, prm.name))
    end
    model.theta_fitted = true
    return nothing
end

"""Reconstruct NamedTuple from internal parameter storage."""
function _ssm_get_theta_namedtuple(model::StateSpaceModel)
    model.theta_fitted || throw(ArgumentError("Parameters not fitted"))
    names = Tuple(prm.name for prm in model.spec.params)
    return NamedTuple{names}(Tuple(model.theta_values))
end

"""Copy KalmanFilterResult to model storage."""
function _ssm_store_filter_results!(model::StateSpaceModel, filt)
    copyto!(model.at, filt.at)
    copyto!(model.Pt, filt.Pt)
    copyto!(model.att, filt.att)
    copyto!(model.Ptt, filt.Ptt)
    copyto!(model.vt, filt.vt)
    copyto!(model.Ft, filt.Ft)
    copyto!(model.Kt, filt.Kt)
    copyto!(model.missing_mask, filt.missing_mask)
    _store_observed_mask!(model.observed_mask, filt)
    model.filter_status = filt.failed_at
    return nothing
end

function _store_observed_mask!(mask, filt::Union{KalmanFilterResult, DiffuseFilterResult})
    copyto!(mask, filt.observed_mask)
end

"""Copy filter results from KalmanWorkspace to model."""
function _ssm_copy_from_workspace!(model::StateSpaceModel, kf_ws::KalmanWorkspace)
    copyto!(model.at, kf_ws.at)
    copyto!(model.Pt, kf_ws.Pt)
    copyto!(model.att, kf_ws.att)
    copyto!(model.Ptt, kf_ws.Ptt)
    copyto!(model.vt, kf_ws.vt)
    copyto!(model.Ft, kf_ws.Ft)
    copyto!(model.Kt, kf_ws.Kt)
    # Missing mask from workspace
    for t in 1:model.n_times
        model.missing_mask[t] = kf_ws.missing_mask[t]
    end
    copyto!(model.observed_mask, kf_ws.observed_mask)
    model.filter_status = kf_ws.status
    return nothing
end

# ============================================
# StateSpaceModel Status Accessors
# ============================================

"""
    parameters(model::StateSpaceModel) -> NamedTuple

Return the fitted parameters θ as a NamedTuple keyed by parameter name. Throws if the
model is not fitted.
"""
function parameters(model::StateSpaceModel)
    _require_fitted(model)
    return _ssm_get_theta_namedtuple(model)
end

# ============================================
# StateSpaceModel Filter Accessors
# ============================================

function _require_filter(model::StateSpaceModel)
    status = model.filter_status
    status == 0 && return nothing
    status == _NOT_RUN && throw(ArgumentError(
        "Filter not run. Call fit!(MLE(), model, y), fit!(EM(), model, y) or " *
        "kalman_filter!(model, y) first."))
    throw(ArgumentError("At the model's parameters $(_failure_message(status))."))
end

# Throw unless `y` is n_obs × n_times for `model`.
function _check_model_obs(model::StateSpaceModel, caller, y::AbstractMatrix)
    _check_size(caller, "y", y, (model.spec.n_obs, model.n_times))
end

# A model built with known parameters has no log-likelihood until its filter has run.
function loglikelihood(model::StateSpaceModel)
    _require_fitted(model)
    model.filter_status == _NOT_RUN && _require_filter(model)
    return model.loglik
end

filtered_states(model::StateSpaceModel) = (_require_filter(model); model.att)
filtered_states_cov(model::StateSpaceModel) = (_require_filter(model); model.Ptt)
predicted_states(model::StateSpaceModel) = (_require_filter(model); model.at)
predicted_states_cov(model::StateSpaceModel) = (_require_filter(model); model.Pt)
function prediction_errors(model::StateSpaceModel)
    _require_filter(model)
    return _unpack_innovations(model.vt, model.observed_mask)
end
function prediction_errors_cov(model::StateSpaceModel)
    _require_filter(model)
    return _unpack_innovation_covs(model.Ft, model.observed_mask)
end
function kalman_gains(model::StateSpaceModel)
    _require_filter(model)
    return _unpack_gains(model.Kt, model.observed_mask)
end
missing_mask(model::StateSpaceModel) = (_require_filter(model); model.missing_mask)
observed_mask(model::StateSpaceModel) = (_require_filter(model); model.observed_mask)

# ============================================
# StateSpaceModel Smoother (Lazy Computation)
# ============================================

"""Compute smoother and cache results."""
function _ssm_compute_smoother!(model::StateSpaceModel)
    _require_filter(model)
    Siphon.uses_exact_diffuse(model.spec) && throw(ArgumentError(
        "Smoothing is not available for exact diffuse initialisation (diffuse = :exact). " *
        "Use approximate diffuse initialisation (diffuse = true) to smooth."))

    # Get system matrices at fitted parameters
    theta_nt = _ssm_get_theta_namedtuple(model)
    kfparms = Siphon.build_kfparms(model.spec, theta_nt)

    # Run smoother using stored filter results
    sm = _kalman_smoother(kfparms.Z, kfparms.T, model.at, model.Pt, model.vt, model.Ft;
        missing_mask = model.missing_mask, observed_mask = model.observed_mask,
        Ptt = model.Ptt)

    # Cache results
    copyto!(model.smoothed_alpha, sm.αs)
    copyto!(model.smoothed_V, sm.Vs)
    model.smoother_computed = true
    return nothing
end

"""
    smoothed_states(model::StateSpaceModel)

Return smoothed states E[αₜ|y₁:ₙ] (m × n).

Computes smoother on first call and caches the result.
"""
function smoothed_states(model::StateSpaceModel)
    _require_filter(model)
    if !model.smoother_computed
        _ssm_compute_smoother!(model)
    end
    return model.smoothed_alpha
end

"""
    smoothed_states_cov(model::StateSpaceModel)

Return smoothed state covariances Var[αₜ|y₁:ₙ] (m × m × n).

Computes smoother on first call and caches the result.
"""
function smoothed_states_cov(model::StateSpaceModel)
    _require_filter(model)
    if !model.smoother_computed
        _ssm_compute_smoother!(model)
    end
    return model.smoothed_V
end

# ============================================
# StateSpaceModel Matrix Accessors
# ============================================

"""
    system_matrices(model::StateSpaceModel) -> KFParms

Return the system matrices at the fitted parameters as a `KFParms` (fields `Z`, `H`,
`T`, `R`, `Q`). Throws if the model is not fitted.

# Example
```julia
p = system_matrices(model)
p.Z  # observation matrix
filt = kalman_filter(p, y, initial_state(model), initial_state_cov(model))
```
"""
function system_matrices(model::StateSpaceModel)
    _require_fitted(model)
    return Siphon.build_kfparms(model.spec, _ssm_get_theta_namedtuple(model))
end

"""
    initial_state(model::StateSpaceModel) -> Vector

Return the initial state mean a₁ at the fitted parameters. Throws if the model is not
fitted.
"""
function initial_state(model::StateSpaceModel)
    _require_fitted(model)
    return first(Siphon.build_initial_state(model.spec, _ssm_get_theta_namedtuple(model)))
end

"""
    initial_state_cov(model::StateSpaceModel) -> Matrix

Return the initial state covariance P₁ at the fitted parameters; under exact diffuse
initialisation, its finite part P₁*. Throws if the model is not fitted.
"""
function initial_state_cov(model::StateSpaceModel)
    _require_fitted(model)
    return last(Siphon.build_initial_state(model.spec, _ssm_get_theta_namedtuple(model)))
end

# ============================================
# Unified StateSpaceModel API (filter/smoother)
# ============================================

"""
    kalman_loglik(model::StateSpaceModel, y::AbstractMatrix; static=:auto)

Compute log-likelihood for observations without storing filter results.

# Arguments
- `model`: StateSpaceModel with fitted parameters (via constructor with θ or fit!)
- `y`: Observations (p × n matrix), missing values as NaN
- `static`: `:auto`, `:on` or `:off`, passed to [`build_linear_state_space`](@ref)

# Returns
Log-likelihood value (Float64)

# Example
```julia
spec = local_level()
model = StateSpaceModel(spec, (var_obs=100.0, var_level=50.0), 100)
y = randn(1, 100)
ll = kalman_loglik(model, y)
```
"""
function kalman_loglik(
        model::StateSpaceModel{T}, y::AbstractMatrix; static::Symbol = :auto) where {T}
    y = _nan_obs(y)
    model.theta_fitted ||
        throw(ArgumentError("Parameters not set. Provide θ in constructor or call fit!."))

    _check_model_obs(model, "kalman_loglik", y)

    theta_nt = _ssm_get_theta_namedtuple(model)
    ss = Siphon.build_linear_state_space(model.spec, theta_nt, y; static)

    return DSL._ss_loglik(ss, y)
end

"""
    kalman_filter!(model::StateSpaceModel, y::AbstractMatrix; static=:auto)

Run Kalman filter and store results in model.

# Arguments
- `model`: StateSpaceModel with fitted parameters
- `y`: Observations (p × n matrix), missing values as NaN
- `static`: `:auto`, `:on` or `:off`, passed to [`build_linear_state_space`](@ref)

# Returns
Log-likelihood value (Float64)

# Side Effects
- Records the filter status; if an innovation covariance is not positive definite
  the log-likelihood is `-Inf` and the filter accessors throw
- Stores filtered/predicted states and covariances
- Updates `model.loglik`

# Example
```julia
spec = local_level()
model = StateSpaceModel(spec, (var_obs=100.0, var_level=50.0), 100)
y = randn(1, 100)
ll = kalman_filter!(model, y)
att = filtered_states(model)
```
"""
function kalman_filter!(
        model::StateSpaceModel{T}, y::AbstractMatrix; static::Symbol = :auto) where {T}
    y = _nan_obs(y)
    model.theta_fitted ||
        throw(ArgumentError("Parameters not set. Provide θ in constructor or call fit!."))

    _check_model_obs(model, "kalman_filter!", y)

    theta_nt = _ssm_get_theta_namedtuple(model)
    ss = Siphon.build_linear_state_space(model.spec, theta_nt, y; static)

    filt = DSL._ss_filter(ss, y)

    # Store results
    _ssm_store_filter_results!(model, filt)
    model.loglik = filt.loglik
    model.smoother_computed = false

    return filt.loglik
end

"""
    kalman_smoother!(model::StateSpaceModel)

Run Kalman smoother using stored filter results.

Requires that `kalman_filter!` has been called first.

# Arguments
- `model`: StateSpaceModel with valid filter results

# Side Effects
- Computes and caches smoothed states and covariances
- Sets `model.smoother_computed = true`

# Example
```julia
spec = local_level()
model = StateSpaceModel(spec, (var_obs=100.0, var_level=50.0), 100)
y = randn(1, 100)
kalman_filter!(model, y)
kalman_smoother!(model)
αs = smoothed_states(model)
```
"""
function kalman_smoother!(model::StateSpaceModel)
    _require_filter(model)

    # Use existing internal function
    _ssm_compute_smoother!(model)
    return nothing
end

# ============================================
# StateSpaceModel fit!(MLE(), ...)
# ============================================

"""
    fit!(::MLE, model::StateSpaceModel, y::AbstractMatrix; static=:auto, kwargs...)

Fit state-space model using Maximum Likelihood Estimation.

Uses the pure AD-compatible Kalman filter via Optimization.jl. For small
models the system matrices are promoted to `SMatrix`/`SVector` so the inner
filter loop runs without heap allocations and the compiler can unroll.

# Arguments
- `MLE()`: Estimation method selector
- `model`: StateSpaceModel to fit (mutated in-place)
- `y`: Observations (p × n matrix), missing values as NaN

# Keyword Arguments
- `static::Symbol = :auto`: Controls whether the optimization uses the
  StaticArrays specialization of the Kalman filter.
  - `:auto` — use it when `n_states`, `n_obs` and `n_shocks` are all at most
    [`STATIC_THRESHOLD`](@ref).
  - `:on`   — force the static path. Warns if any dimension exceeds
    `STATIC_THRESHOLD`; StaticArrays above that size typically lose to BLAS.
  - `:off`  — force the dynamic-`Matrix` path. Useful when you do nested
    `ForwardDiff` or want to avoid the per-`Dual{...}` SMatrix recompile.
  The choice affects speed; the estimates agree up to rounding error, which an
  ill-conditioned innovation covariance amplifies.
- `method`: Optimization algorithm (default: LBFGS from Optim.jl)
- `verbose`: Print optimization progress (default: false)
- Additional kwargs passed to `optimize_ssm`.

# Returns
The fitted `model` (same object, mutated).

# Static-path tradeoff
The `SMatrix` specialization gives ~6-10× speedup for one-shot likelihoods on
small models, but each unique element type and dimension combination
requires a fresh specialization (~100s of ms). When you take gradients via
`ForwardDiff`, every chunk size triggers another compile; nested AD
compounds it. Pass `static = :off` to opt out for those workflows.

# Example
```julia
spec = local_level(var_obs=:free, var_level=:free)
model = StateSpaceModel(spec, 100)
fit!(MLE(), model, randn(1, 100))
parameters(model)  # (var_obs=..., var_level=...)
```
"""
function fit!(
        ::MLE,
        model::StateSpaceModel{T},
        y::AbstractMatrix;
        static::Symbol = :auto,
        verbose::Bool = false,
        kwargs...
) where {T}
    y = _nan_obs(y)
    _check_model_obs(model, "fit!(MLE(), ...)", y)

    result = Siphon.optimize_ssm(model.spec, y; static, kwargs...)

    # Store fitted parameters
    _ssm_store_theta!(model, result.θ)

    # Run full filter to get filter results
    theta_nt = _ssm_get_theta_namedtuple(model)
    ss = Siphon.build_linear_state_space(model.spec, theta_nt, y; static)
    filt = DSL._ss_filter(ss, y)

    # Copy filter results to model storage
    _ssm_store_filter_results!(model, filt)

    # Update model state
    model.loglik = result.loglik
    model.fitted = true
    model.converged = result.converged
    model.iterations = 0  # MLE doesn't have iteration count in same sense
    model.smoother_computed = false

    if verbose
        println("MLE fitting complete:")
        println("  Log-likelihood: ", round(model.loglik, digits = 4))
        println("  Converged: ", model.converged)
    end

    return model
end

# ============================================
# StateSpaceModel Forecasting
# ============================================

"""
    forecast(model::StateSpaceModel, h::Integer) -> ForecastResult

Forecast the `h` periods after the sample from a fitted model, starting from the
filtered state of the last period. See [`ForecastResult`](@ref).

# Example
```julia
model = StateSpaceModel(local_level(), 100)
fit!(MLE(), model, y)
fc = forecast(model, 10)
forecast_observations(fc)  # 1 × 10 forecasts of periods 101:110
```
"""
function forecast(model::StateSpaceModel, h::Integer)
    _require_fitted(model)
    _require_filter(model)

    kfparms = Siphon.build_kfparms(model.spec, _ssm_get_theta_namedtuple(model))
    n = model.n_times
    return _forecast_from_filtered(kfparms, model.att[:, n], model.Ptt[:, :, n], n, h)
end

# ============================================
# Dynamic Factor Model
# ============================================

"""
    DynamicFactorModelSpec

Specification for a dynamic factor model with optional dynamic loadings and AR errors.

Model structure (following Stock & Watson notation):
    Xₜ = λ(L) fₜ + eₜ
    fₜ = Ψ(L) fₜ₋₁ + ηₜ
    eᵢₜ = δ(L) eᵢ,ₜ₋₁ + vᵢₜ

where:
    λ(L) = λ₀ + λ₁L + ... + λₚLᵖ  (dynamic factor loadings)
    Ψ(L) = Ψ₁L + ... + ΨᵧLᵧ       (factor VAR dynamics)
    δ(L) = δ₁L + ... + δᵣLʳ       (AR idiosyncratic errors, common across i)

State-space representation:
    State vector: αₜ = [fₜ; fₜ₋₁; ...; fₜ₋ₛ₊₁], with s = max(q, p + r + 1)

    With white-noise errors (r = 0) the observation equation is
    Xₜ = λ(L) fₜ + vₜ. With AR errors (r > 0) it is quasi-differenced:

        X̃ₜ = Xₜ - δ₁Xₜ₋₁ - ... - δᵣXₜ₋ᵣ = λ(L)(1 - δ(L)L) fₜ + vₜ,

    so the observation noise is vₜ with H = diag(σ²_v) in both cases and the
    errors eₜ are not states.

Dimensions:
    N = number of observables
    k = number of factors
    p = lags in λ(L) (dynamic loadings)
    q = lags in Ψ(L) (factor VAR)
    r = lags in δ(L) (AR errors)

    m = k * s (state dimension)
"""
struct DynamicFactorModelSpec
    n_obs::Int           # N: number of observables
    n_factors::Int       # k: number of factors
    loading_lags::Int    # p: lags in λ(L), p=0 means static loadings
    factor_lags::Int     # q: lags in factor VAR Ψ(L)
    error_lags::Int      # r: lags in AR errors δ(L), r=0 means white noise
end

"""
    DynamicFactorModelWorkspace{T}

Internal workspace for DFM estimation: the model parameters in their natural
form, plus the quasi-differenced data and sufficient statistics used when the
idiosyncratic errors are autoregressive.
"""
mutable struct DynamicFactorModelWorkspace{T <: Real}
    # Model specification
    spec::DynamicFactorModelSpec

    # Dimensions
    n_obs::Int           # N
    n_factors::Int       # k
    loading_lags::Int    # p
    factor_lags::Int     # q
    error_lags::Int      # r
    n_times::Int         # n
    state_dim::Int       # m = k * s, s = max(q, p + r + 1)

    # Parameter storage (separate from KalmanWorkspace for clarity)
    # Factor loadings: Λ[j] is N × k for lag j, j = 0, 1, ..., p
    Λ::Vector{Matrix{T}}

    # Factor VAR: Φ[j] is k × k for lag j, j = 1, ..., q
    Φ::Vector{Matrix{T}}

    # Factor innovation covariance
    Σ_η::Matrix{T}  # k × k

    # AR coefficients for errors: δ[j] is scalar for lag j, j = 1, ..., r
    δ::Vector{T}

    # Idiosyncratic innovation variances (diagonal)
    σ²_v::Vector{T}  # N × 1

    # AR errors only (r > 0)
    y_qd::Matrix{T}        # N × n: quasi-differenced data X̃ₜ, NaN where undefined
    y_tail::Matrix{T}      # N × r: last r observations, for forecasting
    S_cc::Array{T, 3}      # (r+1) × (r+1) × N: moments of (eᵢₜ, …, eᵢ,ₜ₋ᵣ)
end

"""
    _setup_dfm(spec::DynamicFactorModelSpec, n_times::Int, ::Type{T}=Float64)

Create workspaces for full dynamic factor model estimation.

Returns (kf_ws, em_ws, dfm_ws) where:
- kf_ws: KalmanWorkspace for filter/smoother
- em_ws: EMWorkspace for basic EM sufficient statistics
- dfm_ws: DynamicFactorModelWorkspace for DFM-specific parameters and updates
"""
function _setup_dfm(
        spec::DynamicFactorModelSpec,
        n_times::Int,
        ::Type{T} = Float64
) where {T}
    N = spec.n_obs
    k = spec.n_factors
    p = spec.loading_lags
    q = spec.factor_lags
    r = spec.error_lags

    # The state holds fₜ, …, fₜ₋ₛ₊₁: q lags for the VAR, and p + r + 1 for the
    # quasi-differenced loadings λ(L)(1 - δ(L)L), which reach fₜ₋ₚ₋ᵣ.
    s = max(q, p + r + 1)
    m = k * s

    # ========================================
    # Build state-space matrices
    # ========================================

    # Observation equation: Z = Λ W(δ) is filled by sync_params_to_ssm! below;
    # H = diag(σ²_v).
    Z = zeros(T, N, m)
    H = Matrix{T}(I, N, N)

    # --- State equation: αₜ₊₁ = T αₜ + R ηₜ ---
    # Factor block: companion form for VAR(q)
    # [fₜ₊₁  ]   [Φ₁ Φ₂ ... Φᵧ 0 ... 0] [fₜ    ]   [I]
    # [fₜ    ] = [I  0  ... 0  0 ... 0] [fₜ₋₁  ] + [0] ηₜ
    # [fₜ₋₁  ]   [0  I  ... 0  0 ... 0] [fₜ₋₂  ]   [0]
    # ...
    Tmat = zeros(T, m, m)
    for lag in 1:min(q, s)
        col_start = (lag - 1) * k + 1
        for i in 1:k
            # Initialize with small diagonal AR coefficients
            Tmat[i, col_start + i - 1] = T(0.5) / lag
        end
    end
    for lag in 1:(s - 1)
        row_start = lag * k + 1
        col_start = (lag - 1) * k + 1
        for i in 1:k
            Tmat[row_start + i - 1, col_start + i - 1] = one(T)
        end
    end

    # Factor shocks enter the first k states.
    R = zeros(T, m, k)
    for i in 1:k
        R[i, i] = one(T)
    end
    Q = Matrix{T}(I, k, k) * T(0.1)

    # Initial state
    a1 = zeros(T, m)
    P1 = Matrix{T}(I, m, m) * T(10.0)

    kf_ws = KalmanWorkspace(Z, H, Tmat, R, Q, a1, P1, n_times)
    em_ws = EMWorkspace(kf_ws)

    # EM constraints. With white-noise errors the loadings are the free cells of
    # Z (columns of fₜ, …, fₜ₋ₚ); with AR errors Z = Λ W(δ) is not free cell by
    # cell and the loadings have their own M-step (_update_dfm_ar!).
    fill!(em_ws.Z_free, false)
    for lag in 0:p
        col_start = lag * k + 1
        for j in 1:k, i in 1:N

            em_ws.Z_free[i, col_start + j - 1] = true
        end
    end
    em_ws.H_diag_only = true

    # T: factor VAR coefficients (first k rows, first k*q columns) are free
    fill!(em_ws.T_free, false)
    for i in 1:k
        for lag in 1:q
            col_start = (lag - 1) * k + 1
            for j in 1:k
                em_ws.T_free[i, col_start + j - 1] = true
            end
        end
    end

    # Q: the factor innovation covariance is free
    fill!(em_ws.Q_free, true)

    # ========================================
    # Create DFM-specific workspace
    # ========================================

    # Random initial contemporaneous loadings; lagged loadings start at zero.
    Λ = [zeros(T, N, k) for _ in 0:p]
    for j in 1:k, i in 1:N

        Λ[1][i, j] = randn(T) * T(0.1)
    end

    Φ = [zeros(T, k, k) for _ in 1:q]
    for lag in 1:q
        col_start = (lag - 1) * k + 1
        Φ[lag] .= Tmat[1:k, col_start:(col_start + k - 1)]
    end

    Σ_η = Matrix{T}(I, k, k) * T(0.1)

    δ_vec = zeros(T, max(r, 1))
    for lag in 1:r
        δ_vec[lag] = T(0.3) / lag
    end

    σ²_v = ones(T, N)

    dfm_ws = DynamicFactorModelWorkspace{T}(
        spec,
        N,
        k,
        p,
        q,
        r,
        n_times,
        m,
        Λ,
        Φ,
        Σ_η,
        δ_vec,
        σ²_v,
        r > 0 ? fill(T(NaN), N, n_times) : Matrix{T}(undef, N, 0),  # y_qd
        Matrix{T}(undef, N, r),                                     # y_tail
        zeros(T, r + 1, r + 1, r > 0 ? N : 0)                       # S_cc
    )
    sync_params_to_ssm!(kf_ws, dfm_ws)

    return kf_ws, em_ws, dfm_ws
end

"""
    _dfm_composite_loadings!(Z, dfm_ws::DynamicFactorModelWorkspace)

Fill `Z` (N × m) with the loadings of the observation equation on the factor
state, `Λ W(δ)`: block column `c` (the loading on fₜ₋꜀) is `Σ_{a+b=c} Λₐ dᵦ`
with `d₀ = 1` and `dᵦ = -δᵦ`. With white-noise errors this is
`[Λ₀ … Λₚ 0 … 0]`.
"""
function _dfm_composite_loadings!(Z, dfm_ws::DynamicFactorModelWorkspace{T}) where {T}
    k, p, r = dfm_ws.n_factors, dfm_ws.loading_lags, dfm_ws.error_lags
    fill!(Z, zero(T))
    for a in 0:p, b in 0:r

        d = b == 0 ? one(T) : -dfm_ws.δ[b]
        cols = ((a + b) * k + 1):((a + b + 1) * k)
        view(Z, :, cols) .+= d .* dfm_ws.Λ[a + 1]
    end
    return Z
end

"""
    sync_params_to_ssm!(kf_ws::KalmanWorkspace, dfm_ws::DynamicFactorModelWorkspace)

Copy parameters from DynamicFactorModelWorkspace to KalmanWorkspace state-space matrices.
"""
function sync_params_to_ssm!(
        kf_ws::KalmanWorkspace{T},
        dfm_ws::DynamicFactorModelWorkspace{T}
) where {T}
    N, k, q = dfm_ws.n_obs, dfm_ws.n_factors, dfm_ws.factor_lags

    _dfm_composite_loadings!(kf_ws.Z, dfm_ws)

    # Factor VAR coefficients
    for lag in 1:q
        col_start = (lag - 1) * k + 1
        for j in 1:k, i in 1:k

            kf_ws.Tmat[i, col_start + j - 1] = dfm_ws.Φ[lag][i, j]
        end
    end

    kf_ws.Q .= dfm_ws.Σ_η

    for i in 1:N
        kf_ws.H[i, i] = dfm_ws.σ²_v[i]
    end

    _update_RQR!(kf_ws)

    return nothing
end

"""
    sync_params_from_ssm!(dfm_ws::DynamicFactorModelWorkspace, kf_ws::KalmanWorkspace)

Copy the factor VAR coefficients and factor innovation covariance, and with
white-noise errors the loadings, from KalmanWorkspace back to
DynamicFactorModelWorkspace after the EM updates.

The idiosyncratic variances `σ²_v`, the AR coefficients `δ` and, with AR errors,
the loadings are written directly into `dfm_ws` by the DFM M-step.
"""
function sync_params_from_ssm!(
        dfm_ws::DynamicFactorModelWorkspace{T},
        kf_ws::KalmanWorkspace{T}
) where {T}
    N, k, p, q, r = dfm_ws.n_obs, dfm_ws.n_factors, dfm_ws.loading_lags,
    dfm_ws.factor_lags, dfm_ws.error_lags

    if r == 0
        for lag in 0:p
            col_start = lag * k + 1
            for j in 1:k, i in 1:N

                dfm_ws.Λ[lag + 1][i, j] = kf_ws.Z[i, col_start + j - 1]
            end
        end
    end

    for lag in 1:q
        col_start = (lag - 1) * k + 1
        for j in 1:k, i in 1:k

            dfm_ws.Φ[lag][i, j] = kf_ws.Tmat[i, col_start + j - 1]
        end
    end

    dfm_ws.Σ_η .= kf_ws.Q

    return nothing
end

"""
    _dfm_quasi_difference!(dfm_ws::DynamicFactorModelWorkspace, y::AbstractMatrix)

Store `X̃ₜ = Xₜ - δ₁Xₜ₋₁ - … - δᵣXₜ₋ᵣ` in `dfm_ws.y_qd`. It is `NaN` for
`t ≤ r` and wherever one of `Xₜ, …, Xₜ₋ᵣ` is missing, so the likelihood is
conditional on the first `r` observations.
"""
function _dfm_quasi_difference!(dfm_ws::DynamicFactorModelWorkspace{T}, y::AbstractMatrix) where {T}
    N, r, n = dfm_ws.n_obs, dfm_ws.error_lags, dfm_ws.n_times
    y_qd, δ = dfm_ws.y_qd, dfm_ws.δ
    @inbounds for t in 1:n
        usable = t > r && all(l -> !any(isnan, view(y, :, t - l)), 0:r)
        for i in 1:N
            if usable
                s = T(y[i, t])
                for l in 1:r
                    s -= δ[l] * y[i, t - l]
                end
                y_qd[i, t] = s
            else
                y_qd[i, t] = T(NaN)
            end
        end
    end
    return y_qd
end

"""
    _update_dfm_ar!(dfm_ws::DynamicFactorModelWorkspace, kf_ws::KalmanWorkspace, y)

M-step for the loadings, AR coefficients and idiosyncratic variances of a DFM
with AR errors, run after the smoother on the quasi-differenced data. Three
conditional maximisations (an ECM step), each increasing the expected
complete-data log-likelihood, so the EM log-likelihood still increases:

1. Loadings given `δ`. With `Λ = [Λ₀ … Λₚ]` and `Z = Λ W(δ)`, the observation
   equation is `X̃ₜ = Λ (W αₜ) + vₜ`, a regression on `W αₜ`:
   `Λ = S_yα W' (W S_αα W')⁻¹`, summed over the periods where `X̃ₜ` is defined.
2. `δ` given `Λ` and `σ²`. The residual `X̃ₜ - Z αₜ` equals
   `cₜ,₀ - Σₗ δₗ cₜ,ₗ` with `cₜ,ₗ = Xₜ₋ₗ - Λ Bₗ αₜ = eₜ₋ₗ`, where `Bₗ αₜ` stacks
   `fₜ₋ₗ, …, fₜ₋ₗ₋ₚ`. With `S_cc[:, :, i] = Σₜ E[cᵢ cᵢ' | Y]`, `δ` is the weighted
   least-squares solution `(Σᵢ S_cc[1:, 1:, i]/σ²ᵢ) \\ (Σᵢ S_cc[1:, 0, i]/σ²ᵢ)`
   (indices from 0). If it is not stationary it is shrunk so the spectral
   radius of the AR companion matrix is 0.99.
3. `σ²ᵢ = (S_cc[0, 0, i] - 2δ'S_cc[1:, 0, i] + δ'S_cc[1:, 1:, i]δ) / n_valid`.
"""
function _update_dfm_ar!(
        dfm_ws::DynamicFactorModelWorkspace{T},
        kf_ws::KalmanWorkspace{T},
        y::AbstractMatrix
) where {T}
    N, k, p, r = dfm_ws.n_obs, dfm_ws.n_factors, dfm_ws.loading_lags, dfm_ws.error_lags
    m, n = dfm_ws.state_dim, dfm_ws.n_times
    αs, Vs, y_qd = kf_ws.αs, kf_ws.Vs, dfm_ws.y_qd
    valid = [!any(isnan, view(y_qd, :, t)) for t in 1:n]
    n_valid = count(valid)

    # --- 1. Loadings ---
    P = k * (p + 1)
    W = zeros(T, P, m)
    for a in 0:p, b in 0:r

        d = b == 0 ? one(T) : -dfm_ws.δ[b]
        for j in 1:k
            W[a * k + j, (a + b) * k + j] += d
        end
    end
    S_yα = zeros(T, N, m)
    S_αα = zeros(T, m, m)
    for t in 1:n
        valid[t] || continue
        a_t = view(αs, :, t)
        mul!(S_yα, view(y_qd, :, t), a_t', one(T), one(T))
        S_αα .+= view(Vs, :, :, t)
        mul!(S_αα, a_t, a_t', one(T), one(T))
    end
    WSW = W * S_αα * W'
    for i in 1:P
        WSW[i, i] += T(1e-10)
    end
    Λall = (S_yα * W') / cholesky(Symmetric(WSW))
    for a in 0:p
        dfm_ws.Λ[a + 1] .= view(Λall, :, (a * k + 1):((a + 1) * k))
    end

    # --- 2. AR coefficients ---
    # G[l + 1] = Λ Bₗ: the loadings of eₜ₋ₗ = Xₜ₋ₗ - G[l + 1] αₜ on the state.
    G = [zeros(T, N, m) for _ in 0:r]
    for l in 0:r, a in 0:p

        view(G[l + 1], :, ((l + a) * k + 1):((l + a + 1) * k)) .= dfm_ws.Λ[a + 1]
    end
    S_cc = dfm_ws.S_cc
    fill!(S_cc, zero(T))
    μ = Matrix{T}(undef, N, r + 1)
    U = [Matrix{T}(undef, m, N) for _ in 0:r]
    for t in 1:n
        valid[t] || continue
        a_t = view(αs, :, t)
        V_t = view(Vs, :, :, t)
        for l in 0:r
            mul!(view(μ, :, l + 1), G[l + 1], a_t)
            for i in 1:N
                μ[i, l + 1] = y[i, t - l] - μ[i, l + 1]
            end
            mul!(U[l + 1], V_t, G[l + 1]')
        end
        @inbounds for i in 1:N, l2 in 0:r, l1 in 0:r
            S_cc[l1 + 1, l2 + 1, i] += μ[i, l1 + 1] * μ[i, l2 + 1] +
                                       dot(view(G[l1 + 1], i, :), view(U[l2 + 1], :, i))
        end
    end

    A = zeros(T, r, r)
    b = zeros(T, r)
    for i in 1:N
        w = one(T) / dfm_ws.σ²_v[i]
        for j in 1:r
            b[j] += w * S_cc[j + 1, 1, i]
            for l in 1:r
                A[j, l] += w * S_cc[j + 1, l + 1, i]
            end
        end
    end
    δ = Symmetric(A) \ b

    # Spectral radius of the AR companion matrix; shrink δⱼ by c^j to rescale it.
    C = zeros(T, r, r)
    C[1, :] .= δ
    for j in 2:r
        C[j, j - 1] = one(T)
    end
    ρ = maximum(abs, eigvals(C))
    if ρ >= T(0.99)
        c = T(0.99) / ρ
        for j in 1:r
            δ[j] *= c^j
        end
    end
    copyto!(dfm_ws.δ, δ)

    # --- 3. Idiosyncratic variances ---
    for i in 1:N
        Sxe = view(S_cc, 2:(r + 1), 1, i)
        Sxx = view(S_cc, 2:(r + 1), 2:(r + 1), i)
        s = S_cc[1, 1, i] - 2 * dot(δ, Sxe) + dot(δ, Sxx, δ)
        dfm_ws.σ²_v[i] = max(s / n_valid, T(1e-10))
    end

    return nothing
end

"""
    _em_dfm!(kf_ws::KalmanWorkspace, em_ws::EMWorkspace, dfm_ws::DynamicFactorModelWorkspace,
                 y::AbstractMatrix; maxiter=500, tol=1e-6, verbose=false)

EM algorithm for full dynamic factor model.

With AR errors each iteration filters the quasi-differenced data for the current
`δ`, so the log-likelihood is that of `X_{r+1:n}` given `X_{1:r}`.
"""
function _em_dfm!(
        kf_ws::KalmanWorkspace{T},
        em_ws::EMWorkspace{T},
        dfm_ws::DynamicFactorModelWorkspace{T},
        y::AbstractMatrix;
        maxiter::Int = 500,
        tol::Real = 1e-6,
        verbose::Bool = false,
        warn::Bool = true
) where {T}
    N, r = dfm_ws.n_obs, dfm_ws.error_lags

    loglik_history = Vector{T}(undef, maxiter)
    ll_prev = T(-Inf)
    converged = false
    iter = 0

    # The data the filter sees: y itself, or its quasi-difference for the current δ.
    filter_data!() = r > 0 ? _dfm_quasi_difference!(dfm_ws, y) : y

    for i in 1:maxiter
        iter = i

        # Sync parameters to state-space form
        sync_params_to_ssm!(kf_ws, dfm_ws)

        # E-step: Filter and smooth
        y_f = filter_data!()
        kalman_filter_and_smooth!(kf_ws, y_f; crosscov = true)
        ll = kf_ws.loglik
        loglik_history[i] = ll
        kf_ws.status == 0 || break

        # Check convergence
        if i > 1
            ll_change = abs(ll - ll_prev)
            rel_change = ll_change / (abs(ll_prev) + T(1e-10))

            if verbose && (i % 10 == 0 || i <= 5)
                @info "EM iteration $i" loglik = ll rel_change
            end

            if rel_change < tol
                converged = true
                verbose && @info "EM converged at iteration $i" loglik = ll
                break
            end
        elseif verbose
            @info "EM iteration $i" loglik = ll
        end

        ll_prev = ll

        # Compute sufficient statistics
        n_valid = compute_sufficient_stats!(em_ws, kf_ws, y_f)

        if r > 0
            # Factor dynamics, then loadings, δ and σ²_v
            update_T!(kf_ws, em_ws)
            update_Q!(kf_ws, em_ws)
            _update_dfm_ar!(dfm_ws, kf_ws, y)
        else
            update_Z!(kf_ws, em_ws, n_valid)  # Updates Λ
            update_T!(kf_ws, em_ws)           # Updates Φ
            update_Q!(kf_ws, em_ws)           # Updates Σ_η
            update_H!(kf_ws, em_ws, n_valid)  # Updates σ²_v
            for i in 1:N
                dfm_ws.σ²_v[i] = kf_ws.H[i, i]
            end
        end

        # Sync back to dfm_ws
        sync_params_from_ssm!(dfm_ws, kf_ws)
    end

    # Final sync and filter. A failed E-step stopped the loop before the M-step,
    # so the workspace already holds the run at the current parameters.
    if kf_ws.status == 0
        sync_params_to_ssm!(kf_ws, dfm_ws)
        kalman_filter_and_smooth!(kf_ws, filter_data!(); crosscov = true)
    end
    warn && _warn_em(converged, iter, kf_ws.status, tol)
    if r > 0
        n = dfm_ws.n_times
        dfm_ws.y_tail .= view(y, :, (n - r + 1):n)
    end

    return (
        converged = converged,
        iterations = iter,
        loglik = kf_ws.loglik,
        loglik_history = loglik_history[1:iter]
    )
end

"""
    DynamicFactorModel{T}

Mutable struct holding specification, workspaces, and fitting state for a dynamic factor model.

# Usage
```julia
# Step 1: Create model (allocates workspaces)
model = DynamicFactorModel(n_obs, n_factors, n_times;
                           loading_lags=0, factor_lags=1, error_lags=0)

# Step 2: Fit with EM algorithm
fit!(EM(), model, y; maxiter=500, tol=1e-6, verbose=false)

# Step 3: Check results
isfitted(model)       # true
isconverged(model)    # true/false
loglikelihood(model)  # final log-likelihood

# Step 4: Access factors and forecast
f = factors(model)    # k × n smoothed factors
fc = forecast(model, h)  # h-step ahead forecast
```

# Model Structure (following Stock & Watson notation)
    Xₜ = λ(L) fₜ + eₜ
    fₜ = Ψ(L) fₜ₋₁ + ηₜ
    eᵢₜ = δ(L) eᵢ,ₜ₋₁ + vᵢₜ

# Fields
## Specification
- `spec::DynamicFactorModelSpec` - Model dimensions and lag structure

## Workspaces (allocated at construction)
- `kf_ws::KalmanWorkspace{T}` - Kalman filter/smoother workspace
- `em_ws::EMWorkspace{T}` - EM algorithm workspace
- `dfm_ws::DynamicFactorModelWorkspace{T}` - DFM-specific workspace

## Fitting state (updated by fit!)
- `fitted::Bool` - Whether model has been fitted
- `converged::Bool` - Whether EM converged
- `iterations::Int` - Number of EM iterations
- `loglik::T` - Final log-likelihood
- `loglik_history::Vector{T}` - Log-likelihood at each iteration
"""
mutable struct DynamicFactorModel{T <: Real} <: AbstractStateSpaceModel
    # Specification
    spec::DynamicFactorModelSpec

    # Workspaces (allocated at construction)
    kf_ws::KalmanWorkspace{T}
    em_ws::EMWorkspace{T}
    dfm_ws::DynamicFactorModelWorkspace{T}

    # Fitting state
    fitted::Bool
    converged::Bool
    iterations::Int
    loglik::T
    loglik_history::Vector{T}
end

"""
    DynamicFactorModel(n_obs::Int, n_factors::Int, n_times::Int;
                       loading_lags::Int=0, factor_lags::Int=1, error_lags::Int=0,
                       T::Type{<:Real}=Float64)

Construct a dynamic factor model with pre-allocated workspaces.

# Arguments
- `n_obs`: Number of observables (N)
- `n_factors`: Number of latent factors (k)
- `n_times`: Number of time periods (n)

# Keyword Arguments
- `loading_lags`: Lags in λ(L), p=0 means static loadings (default: 0)
- `factor_lags`: Lags in factor VAR Ψ(L) (default: 1)
- `error_lags`: Lags in AR errors δ(L), r=0 means white noise (default: 0)
- `T`: Element type for matrices (default: Float64)

# Returns
- `DynamicFactorModel{T}` ready for fitting with `fit!(EM(), model, data)`

# Example
```julia
# 100 observables, 6 factors, 200 time periods, VAR(3) dynamics
model = DynamicFactorModel(100, 6, 200; factor_lags=3)
```
"""
function DynamicFactorModel(
        n_obs::Int,
        n_factors::Int,
        n_times::Int;
        loading_lags::Int = 0,
        factor_lags::Int = 1,
        error_lags::Int = 0,
        T::Type{<:Real} = Float64
)
    n_obs >= 1 || throw(ArgumentError("n_obs must be at least 1, got $n_obs"))
    n_factors >= 1 || throw(ArgumentError("n_factors must be at least 1, got $n_factors"))
    n_times >= 2 || throw(ArgumentError("n_times must be at least 2, got $n_times"))
    for (name, val) in ((:loading_lags, loading_lags), (:factor_lags, factor_lags),
        (:error_lags, error_lags))
        val >= 0 || throw(ArgumentError("$name must be non-negative, got $val"))
    end
    spec = DynamicFactorModelSpec(n_obs, n_factors, loading_lags, factor_lags, error_lags)
    kf_ws, em_ws, dfm_ws = _setup_dfm(spec, n_times, T)

    return DynamicFactorModel{T}(
        spec,
        kf_ws,
        em_ws,
        dfm_ws,
        false,           # fitted
        false,           # converged
        0,               # iterations
        T(-Inf),         # loglik
        T[]              # loglik_history
    )
end

# ============================================
# Status Accessors
# ============================================

_fit_hint(::DynamicFactorModel) = "fit!(EM(), model, y)"

# ============================================
# Factor Accessors
# ============================================

"""
    factors(model::DynamicFactorModel)

Extract smoothed factors (k × n) from fitted model.
"""
function factors(model::DynamicFactorModel)
    _require_fitted(model)
    k = model.spec.n_factors
    return smoothed_states(model.kf_ws)[1:k, :]
end

"""
    factors_cov(model::DynamicFactorModel)

Extract smoothed factor covariances (k × k × n) from fitted model.
"""
function factors_cov(model::DynamicFactorModel)
    _require_fitted(model)
    k = model.spec.n_factors
    return smoothed_states_cov(model.kf_ws)[1:k, 1:k, :]
end

"""
    filtered_factors(model::DynamicFactorModel)

Extract filtered factors (k × n) from fitted model.
"""
function filtered_factors(model::DynamicFactorModel)
    _require_fitted(model)
    k = model.spec.n_factors
    return filtered_states(model.kf_ws)[1:k, :]
end

"""
    filtered_factors_cov(model::DynamicFactorModel)

Extract filtered factor covariances (k × k × n) from fitted model.
"""
function filtered_factors_cov(model::DynamicFactorModel)
    _require_fitted(model)
    k = model.spec.n_factors
    return filtered_states_cov(model.kf_ws)[1:k, 1:k, :]
end

# ============================================
# Parameter Accessors
# ============================================

"""
    loadings(model::DynamicFactorModel)

Return factor loadings [Λ₀, Λ₁, ..., Λₚ] from fitted model.
"""
function loadings(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    return model.dfm_ws.Λ
end

"""
    var_coefficients(model::DynamicFactorModel)

Return factor VAR coefficients [Φ₁, ..., Φq] from fitted model.
"""
function var_coefficients(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    return model.dfm_ws.Φ
end

"""
    innovation_cov(model::DynamicFactorModel)

Return factor innovation covariance Σ_η (k × k) from fitted model.
"""
function innovation_cov(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    return model.dfm_ws.Σ_η
end

"""
    ar_coefficients(model::DynamicFactorModel)

Return AR error coefficients [δ₁, ..., δᵣ] from fitted model.
"""
function ar_coefficients(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    return model.dfm_ws.δ
end

"""
    idiosyncratic_variances(model::DynamicFactorModel)

Return idiosyncratic innovation variances σ²_v (length N) from fitted model.
"""
function idiosyncratic_variances(model::DynamicFactorModel)
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))
    return model.dfm_ws.σ²_v
end

# ============================================
# fit! (StatsAPI-style interface)
# ============================================

"""
    fit!(::EM, model::DynamicFactorModel, y::AbstractMatrix;
         maxiter::Int=500, tol::Real=1e-6, verbose::Bool=false, warn::Bool=true)

Fit a dynamic factor model using the EM algorithm.

# Arguments
- `EM()`: Estimation method selector
- `model`: DynamicFactorModel to fit (mutated in-place)
- `y`: Observations (N × n matrix), missing values as NaN

# Keyword Arguments
- `maxiter`: Maximum EM iterations (default: 500)
- `tol`: Convergence tolerance for log-likelihood (default: 1e-6)
- `verbose`: Print progress (default: false)
- `warn`: Warn when EM stops without converging (default: true)

# Returns
- The fitted `model` (same object, mutated)

With AR errors (`error_lags = r > 0`) the model is estimated on the
quasi-differenced data `Xₜ - δ₁Xₜ₋₁ - … - δᵣXₜ₋ᵣ`, so the log-likelihood is that
of `X_{r+1:n}` given `X_{1:r}`, and a period contributes only when it and its `r`
predecessors are observed. The smoothed factors of the first `r` periods are
informed by the factor dynamics only.

# Example
```julia
model = DynamicFactorModel(100, 6, 200; factor_lags=3)
fit!(EM(), model, y; verbose=true)
f = factors(model)  # k × n smoothed factors
```
"""
function fit!(
        ::EM,
        model::DynamicFactorModel{T},
        y::AbstractMatrix;
        maxiter::Int = 500,
        tol::Real = 1e-6,
        verbose::Bool = false,
        warn::Bool = true
) where {T}
    y = _nan_obs(y)
    N, n = size(y)
    spec = model.spec

    # Validate dimensions
    N == spec.n_obs ||
        throw(DimensionMismatch("Data has $N observables but model expects $(spec.n_obs)"))
    n == model.kf_ws.n_times || throw(
        DimensionMismatch(
        "Data has $n time periods but model workspace allocated for $(model.kf_ws.n_times)",
    ),
    )
    _require_no_partial_periods(y, "fit!(EM(), ::DynamicFactorModel, ...)")

    if verbose
        println("Fitting DynamicFactorModel via EM:")
        println("  Observables (N): ", spec.n_obs)
        println("  Factors (k): ", spec.n_factors)
        println("  Loading lags (p): ", spec.loading_lags)
        println("  Factor VAR lags (q): ", spec.factor_lags)
        println("  Error AR lags (r): ", spec.error_lags)
        println("  State dimension: ", model.dfm_ws.state_dim)
    end

    # Run EM algorithm
    result = _em_dfm!(
        model.kf_ws,
        model.em_ws,
        model.dfm_ws,
        y;
        maxiter = maxiter,
        tol = tol,
        verbose = verbose,
        warn = warn
    )

    # Update model state
    model.fitted = true
    model.converged = result.converged
    model.iterations = result.iterations
    model.loglik = result.loglik
    model.loglik_history = result.loglik_history

    return model
end

# ============================================
# Forecasting Functions
# ============================================

"""
    forecast(model::DynamicFactorModel, h::Integer) -> ForecastResult

Compute h-step ahead forecasts from a fitted DFM.

Uses the filtered state at the last observation (αₙ|ₙ, Pₙ|ₙ) and iterates
the state equation forward:
    E[αₙ₊ₕ|Y₁:ₙ] = Tʰ αₙ|ₙ
    Var[αₙ₊ₕ|Y₁:ₙ] = Tʰ Pₙ|ₙ (T')ʰ + Σⱼ₌₀ʰ⁻¹ Tʲ RQR' (T')ʲ

With AR errors the observation forecast also iterates
`Xₙ₊ⱼ = δ₁Xₙ₊ⱼ₋₁ + … + δᵣXₙ₊ⱼ₋ᵣ + Z αₙ₊ⱼ + vₙ₊ⱼ` from the last `r` observations,
which must not be missing.

The factor forecasts are the first `k` states: `forecast_states(fc)[1:k, :]`, with
intervals from `forecast_interval(fc; what = :states)`.

# Example
```julia
model = DynamicFactorModel(100, 6, 200; factor_lags=3)
fit!(EM(), model, y)
fc = forecast(model, 4)
lower, upper = forecast_interval(fc, 0.95)
```
"""
function forecast(model::DynamicFactorModel{T}, h::Integer) where {T}
    model.fitted ||
        throw(ArgumentError("Model not fitted. Call fit!(EM(), model, y) first."))

    model.spec.error_lags > 0 && return _forecast_dfm_ar(model, h)

    kf_ws = model.kf_ws
    m, n = kf_ws.state_dim, kf_ws.n_times
    # The workspace keeps RQR' only, so it stands in for Q with R = I.
    p = KFParms(kf_ws.Z, kf_ws.H, kf_ws.Tmat, Matrix{T}(I, m, m), kf_ws.RQR)
    return _forecast_from_filtered(p, kf_ws.att[:, n], kf_ws.Ptt[:, :, n], n, h)
end

# Forecast of a DFM with AR errors. The observations follow
#   Xₜ₊₁ = δ₁Xₜ + … + δᵣXₜ₋ᵣ₊₁ + Z αₜ₊₁ + vₜ₊₁,
# so the forecast iterates the augmented state βₜ = [αₜ; Xₜ; …; Xₜ₋ᵣ₊₁] from the
# filtered αₙ and the last r observations, which are known.
function _forecast_dfm_ar(model::DynamicFactorModel{T}, h::Integer) where {T}
    h >= 1 || throw(ArgumentError("forecast horizon must be at least 1, got $h"))
    N, r = model.spec.n_obs, model.spec.error_lags
    kf_ws, dfm_ws = model.kf_ws, model.dfm_ws
    m, n = kf_ws.state_dim, kf_ws.n_times
    any(isnan, dfm_ws.y_tail) && throw(ArgumentError(
        "forecasting a DFM with AR errors needs the last $r observations; some are missing"))

    Z, Tm, H, RQR = kf_ws.Z, kf_ws.Tmat, kf_ws.H, kf_ws.RQR
    M = m + N * r
    xs = (m + 1):(m + N)                       # rows of Xₜ in βₜ

    Ta = zeros(T, M, M)
    Ta[1:m, 1:m] .= Tm
    Ta[xs, 1:m] .= Z * Tm
    for l in 1:r
        for i in 1:N
            Ta[m + i, m + (l - 1) * N + i] = dfm_ws.δ[l]
        end
    end
    for l in 1:(r - 1), i in 1:N

        Ta[m + l * N + i, m + (l - 1) * N + i] = one(T)
    end

    # Shocks: Rηₜ₊₁ to αₜ₊₁, Z Rηₜ₊₁ + vₜ₊₁ to Xₜ₊₁.
    Qa = zeros(T, M, M)
    Qa[1:m, 1:m] .= RQR
    Qa[1:m, xs] .= RQR * Z'
    Qa[xs, 1:m] .= Z * RQR
    Qa[xs, xs] .= Z * RQR * Z' .+ H

    a = zeros(T, M)
    a[1:m] .= view(kf_ws.att, :, n)
    for l in 1:r
        a[(m + (l - 1) * N + 1):(m + l * N)] .= view(dfm_ws.y_tail, :, r - l + 1)
    end
    P = zeros(T, M, M)
    P[1:m, 1:m] .= view(kf_ws.Ptt, :, :, n)

    state_mean = Matrix{T}(undef, m, h)
    state_cov = Array{T, 3}(undef, m, m, h)
    obs_mean = Matrix{T}(undef, N, h)
    obs_cov = Array{T, 3}(undef, N, N, h)
    for j in 1:h
        a = Ta * a
        P = Ta * P * Ta' + Qa
        state_mean[:, j] .= view(a, 1:m)
        state_cov[:, :, j] .= view(P, 1:m, 1:m)
        obs_mean[:, j] .= view(a, xs)
        obs_cov[:, :, j] .= view(P, xs, xs)
    end

    return ForecastResult(obs_mean, obs_cov, state_mean, state_cov, n)
end
