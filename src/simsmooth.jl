"""
    simsmooth.jl

Durbin & Koopman (2002) simulation smoother: draws from the conditional
distribution of the state path given the observations.
"""

# ============================================
# Positive semi-definite square roots
# ============================================

"""
    PSDFactor{T}

A factor `L` of a symmetric positive semi-definite matrix `S`, satisfying
`L * L' == S`. `L` is `k × k` where `k = size(S, 1)`, with zero columns where
`S` is rank deficient, so `L * randn(k)` is `N(0, S)` whatever the rank of `S`.

Built by eigendecomposition rather than Cholesky: the covariances that arise in
a companion-form VAR (`Q` of rank `n` out of `n·p`) and a noiseless measurement
equation (`H = 0`) are singular, and `cholesky` rejects them.
"""
struct PSDFactor{T <: Real}
    L::Matrix{T}
end

Base.size(f::PSDFactor) = size(f.L)
Base.size(f::PSDFactor, d::Integer) = size(f.L, d)

"""
    psd_factor(S::AbstractMatrix; tol=nothing) -> PSDFactor

Factor a symmetric positive semi-definite `S` as `L * L'`.

Eigenvalues below `tol` are treated as zero; the default scales with the largest
eigenvalue and the size of `S`. An eigenvalue below `-tol` means `S` is not
positive semi-definite and raises an `ArgumentError` rather than being clamped,
since a negative variance is a modeling error and not a rounding artifact.
"""
function psd_factor(S::AbstractMatrix{<:Real}; tol::Union{Nothing, Real} = nothing)
    k = LinearAlgebra.checksquare(S)
    T = float(eltype(S))
    k == 0 && return PSDFactor(Matrix{T}(undef, 0, 0))

    E = eigen(Symmetric(Matrix{T}(S)))
    λ = E.values
    τ = tol === nothing ? k * eps(T) * max(one(T), maximum(abs, λ)) : T(tol)

    λmin = minimum(λ)
    if λmin < -τ
        throw(ArgumentError(
            "matrix is not positive semi-definite: smallest eigenvalue $λmin " *
            "is below the tolerance -$τ"))
    end

    L = E.vectors * Diagonal(sqrt.(max.(λ, zero(T))))
    return PSDFactor(L)
end

"""
    psd_rand!(out, f::PSDFactor, rng, scratch) -> out

Overwrite `out` with a draw from `N(0, S)`, where `f` factors `S`. `scratch` is a
`size(f, 2)`-vector of working space; it is overwritten with standard normals.
"""
function psd_rand!(
        out::AbstractVector, f::PSDFactor, rng::Random.AbstractRNG,
        scratch::AbstractVector)
    randn!(rng, scratch)
    mul!(out, f.L, scratch)
    return out
end

# ============================================
# Simulation smoother workspace
# ============================================

"""
    SimulationSmootherWorkspace{T}

Working storage for `simulation_smoother!`. Holds the factors of `P1`, `Q` and
`H`, the simulated path and its synthetic observations, and a copy of the
smoothed states of the real data.

The factors depend only on `(P1, Q, H)`, so they are computed when the workspace
is built and refreshed by `refresh_factors!` when those matrices change. A draw
therefore costs two filter/smoother passes over storage that is already allocated,
or one when the smoothed states of the data are supplied.
"""
mutable struct SimulationSmootherWorkspace{T <: Real}
    state_dim::Int
    obs_dim::Int
    shock_dim::Int
    n_times::Int

    P1_factor::PSDFactor{T}
    Q_factor::PSDFactor{T}
    H_factor::PSDFactor{T}

    α_plus::Matrix{T}     # m × n: simulated state path
    y_plus::Matrix{T}     # p × n: synthetic observations, censored like y
    α_hat::Matrix{T}      # m × n: smoothed states of the real data
    obs_pattern::BitMatrix # p × n: true where y is observed

    scratch_m::Vector{T}  # m
    scratch_r::Vector{T}  # r
    scratch_p::Vector{T}  # p
    noise_m::Vector{T}    # m
    noise_r::Vector{T}    # r
    noise_p::Vector{T}    # p
end

"""
    SimulationSmootherWorkspace(ws::KalmanWorkspace)

Build simulation-smoother storage sized to `ws` and factor its `(P1, Q, H)`.
"""
function SimulationSmootherWorkspace(ws::KalmanWorkspace{T}) where {T}
    m, p, r, n = ws.state_dim, ws.obs_dim, ws.shock_dim, ws.n_times
    return SimulationSmootherWorkspace{T}(
        m, p, r, n,
        psd_factor(ws.P1), psd_factor(ws.Q), psd_factor(ws.H),
        Matrix{T}(undef, m, n), Matrix{T}(undef, p, n), Matrix{T}(undef, m, n),
        BitMatrix(undef, p, n),
        Vector{T}(undef, m), Vector{T}(undef, r), Vector{T}(undef, p),
        Vector{T}(undef, m), Vector{T}(undef, r), Vector{T}(undef, p)
    )
end

"""
    refresh_factors!(sws::SimulationSmootherWorkspace, ws::KalmanWorkspace)

Recompute the factors of `P1`, `Q` and `H` from `ws`. Call after changing any of
those three matrices; the draw uses the cached factors and will otherwise keep
simulating from the previous model.
"""
function refresh_factors!(
        sws::SimulationSmootherWorkspace, ws::KalmanWorkspace)
    sws.P1_factor = psd_factor(ws.P1)
    sws.Q_factor = psd_factor(ws.Q)
    sws.H_factor = psd_factor(ws.H)
    return sws
end

# ============================================
# Unconditional simulation
# ============================================

"""
    _simulate_path!(sws, ws, rng, observed_mask)

Fill `sws.α_plus` with a draw from the model's unconditional state distribution
and `sws.y_plus` with the matching observations, set to `NaN` wherever
`observed_mask` says the real data is missing.

Censoring `y⁺` to the real pattern is what makes the draw correct: both smoothing
passes must condition on the same set of positions, or they run on different
information and the difference is not a draw from the posterior.
"""
function _simulate_path!(
        sws::SimulationSmootherWorkspace{T}, ws::KalmanWorkspace{T},
        rng::Random.AbstractRNG, observed_mask::AbstractMatrix{Bool}) where {T}
    m, p, n = sws.state_dim, sws.obs_dim, sws.n_times

    # α⁺₁ ~ N(a₁, P₁)
    psd_rand!(sws.scratch_m, sws.P1_factor, rng, sws.noise_m)
    for i in 1:m
        sws.α_plus[i, 1] = ws.a1[i] + sws.scratch_m[i]
    end

    for t in 1:n
        α_t = view(sws.α_plus, :, t)

        # y⁺ₜ = Z α⁺ₜ + ε⁺ₜ, ε⁺ₜ ~ N(0, H)
        y_t = view(sws.y_plus, :, t)
        mul!(y_t, ws.Z, α_t)
        psd_rand!(sws.scratch_p, sws.H_factor, rng, sws.noise_p)
        for i in 1:p
            y_t[i] = observed_mask[i, t] ? y_t[i] + sws.scratch_p[i] : T(NaN)
        end

        # α⁺ₜ₊₁ = T α⁺ₜ + R η⁺ₜ, η⁺ₜ ~ N(0, Q)
        if t < n
            α_next = view(sws.α_plus, :, t + 1)
            mul!(α_next, ws.Tmat, α_t)
            psd_rand!(sws.scratch_r, sws.Q_factor, rng, sws.noise_r)
            mul!(sws.scratch_m, ws.R, sws.scratch_r)
            for i in 1:m
                α_next[i] += sws.scratch_m[i]
            end
        end
    end

    return sws
end

# ============================================
# Simulation smoother
# ============================================

"""
    simulation_smoother!(out, sws, ws, y; rng=Random.default_rng(), smoothed=nothing)

Overwrite `out` (`m × n`) with one draw from `p(α₁:ₙ | y₁:ₙ)` and return it.

Implements the mean-correction form of Durbin & Koopman (2002): simulate a state
path `α⁺` and matching observations `y⁺` from the model, smooth both `y` and
`y⁺`, and form

    α̃ = α̂ − α̂⁺ + α⁺

which has the mean of `α̂` and the covariance of the conditional distribution.

`ws` must already carry the model parameters and initial conditions. It is left
holding the filter and smoother results for `y⁺`, not for `y`, so a caller that
needs the smoothed mean of the real data should keep the `smoothed` copy this
function makes rather than reading `ws` afterwards. Only the means are computed:
`ws.Vs` is left holding whatever an earlier pass wrote and must not be read.

# Arguments
- `out`: destination, `m × n`
- `sws`: simulation-smoother storage, factors matching `ws`
- `ws`: Kalman workspace with parameters set
- `y`: observations, `p × n`, missing entries `NaN`
- `rng`: random source; pass an explicit seeded generator for reproducibility
- `smoothed`: `m × n` smoothed states of `y` from an earlier call. Supplying this
  skips the first of the two smoothing passes, which halves the cost of every
  draw after the first for a fixed model and dataset.

# Notes
Requires the in-place filter, which is the only path that handles periods where
some rows are observed and others are not. `Q` and `H` may be singular.
"""
function simulation_smoother!(
        out::AbstractMatrix, sws::SimulationSmootherWorkspace{T},
        ws::KalmanWorkspace{T}, y::AbstractMatrix;
        rng::Random.AbstractRNG = Random.default_rng(),
        smoothed::Union{Nothing, AbstractMatrix} = nothing) where {T}
    m, p, n = sws.state_dim, sws.obs_dim, sws.n_times
    size(y) == (p, n) || throw(DimensionMismatch(
        "observations are $(size(y)) but the workspace expects ($p, $n)"))
    size(out) == (m, n) || throw(DimensionMismatch(
        "output is $(size(out)) but the state path is ($m, $n)"))

    # α̂: smoothed states of the real data.
    if smoothed === nothing
        filter_and_smooth!(ws, y; crosscov = false, covariances = false)
        copyto!(sws.α_hat, ws.αs)
    else
        size(smoothed) == (m, n) || throw(DimensionMismatch(
            "smoothed states are $(size(smoothed)) but the state path is ($m, $n)"))
        copyto!(sws.α_hat, smoothed)
    end

    # α⁺, y⁺ from the model, censored to the pattern of y. The pattern is read
    # from `y` itself rather than from the workspace, whose mask describes
    # whichever dataset it last filtered.
    for t in 1:n, i in 1:p

        sws.obs_pattern[i, t] = !isnan(y[i, t])
    end
    _simulate_path!(sws, ws, rng, sws.obs_pattern)

    # α̂⁺: smoothed states of the synthetic data.
    filter_and_smooth!(ws, sws.y_plus; crosscov = false, covariances = false)

    for t in 1:n, i in 1:m

        out[i, t] = sws.α_hat[i, t] - ws.αs[i, t] + sws.α_plus[i, t]
    end

    return out
end

"""
    simulation_smoother(ws::KalmanWorkspace, y; rng=Random.default_rng(), n_draws=1)

Draw from `p(α₁:ₙ | y₁:ₙ)` and return the draws.

With `n_draws == 1` returns an `m × n` matrix; otherwise an `m × n × n_draws`
array. The smoothing pass over `y` is shared across draws.

`ws` must already carry the model parameters and initial conditions, and is left
holding filter and smoother results for the last synthetic dataset.

# Example
```julia
ws = KalmanWorkspace(Z, H, T, R, Q, a1, P1, n)
draws = simulation_smoother(ws, y; rng = MersenneTwister(1), n_draws = 100)
```
"""
function simulation_smoother(
        ws::KalmanWorkspace{T}, y::AbstractMatrix;
        rng::Random.AbstractRNG = Random.default_rng(),
        n_draws::Integer = 1) where {T}
    n_draws >= 1 || throw(ArgumentError("n_draws must be at least 1, got $n_draws"))
    m, n = ws.state_dim, ws.n_times
    sws = SimulationSmootherWorkspace(ws)

    filter_and_smooth!(ws, y; crosscov = false, covariances = false)
    α_hat = copy(ws.αs)

    if n_draws == 1
        out = Matrix{T}(undef, m, n)
        return simulation_smoother!(out, sws, ws, y; rng = rng, smoothed = α_hat)
    end

    draws = Array{T, 3}(undef, m, n, n_draws)
    out = Matrix{T}(undef, m, n)
    for d in 1:n_draws
        simulation_smoother!(out, sws, ws, y; rng = rng, smoothed = α_hat)
        copyto!(view(draws, :, :, d), out)
    end
    return draws
end
