# ============================================
# EM estimation of SSMSpec models
# ============================================
#
# `em_estimate!` works on the system matrices held in a KalmanWorkspace. This
# file maps an SSMSpec onto that representation: which cells EM updates, which
# cells share a parameter, and how the updated matrices map back to parameter
# values.

# Where each spec parameter lives, as seen by the EM M-step.
#
# - `cells[name] = (mat, cells)`: the parameter fills plain `ParameterRef`
#   cells of one matrix (`:Z`, `:H`, `:T`, `:Q`, `:a1` or `:P1`). For `:a1` the
#   cell is `CartesianIndex(i, 1)`.
# - `cov[mat] = expr`: `H` or `Q` is a full covariance built by a
#   `CovMatrixExpr`; EM updates the whole block.
# - `blocked[name] = reason`: EM has no closed-form update for the parameter.
struct _EMLayout
    cells::Dict{Symbol, Tuple{Symbol, Vector{CartesianIndex{2}}}}
    cov::Dict{Symbol, CovMatrixExpr}
    blocked::Dict{Symbol, String}
end

const _EM_MATRICES = (:Z, :H, :T, :Q, :P1)

function _em_layout(spec::SSMSpec)
    names = Set(prm.name for prm in spec.params)
    cells = Dict{Symbol, Tuple{Symbol, Vector{CartesianIndex{2}}}}()
    cov = Dict{Symbol, CovMatrixExpr}()
    blocked = Dict{Symbol, String}()

    block!(name, reason) = (name in names && !haskey(blocked, name)) &&
                           (blocked[name] = reason)
    function add_cell!(name, mat, I)
        name in names || return
        if haskey(cells, name) && cells[name][1] != mat
            block!(name, "it appears in both $(cells[name][1]) and $mat")
        elseif haskey(cells, name)
            push!(cells[name][2], I)
        else
            cells[name] = (mat, [I])
        end
    end
    function scan_spec!(mat, mspec::SSMMatrixSpec)
        for ((i, j), elem) in mspec.elements
            elem isa ParameterRef && add_cell!(elem.name, mat, CartesianIndex(i, j))
        end
    end

    for mat in _EM_MATRICES
        if haskey(spec.matrix_exprs, mat)
            expr = spec.matrix_exprs[mat]
            if expr isa CovMatrixExpr && mat in (:H, :Q)
                cov[mat] = expr
            elseif expr isa CovMatrixExpr
                for s in vcat(expr.σ_param_names, expr.corr_param_names)
                    block!(s, "it enters $mat through a covariance expression")
                end
            else
                for prm in expr.params
                    block!(prm.name, "it enters $mat through a matrix expression")
                end
            end
        else
            scan_spec!(mat, getfield(spec, mat))
        end
    end
    for (i, elem) in enumerate(spec.a1)
        elem isa ParameterRef && add_cell!(elem.name, :a1, CartesianIndex(i, 1))
    end
    if haskey(spec.matrix_exprs, :R)
        for prm in spec.matrix_exprs[:R].params
            block!(prm.name, "it enters R, which EM does not update")
        end
    else
        for (_, elem) in spec.R.elements
            elem isa ParameterRef &&
                block!(elem.name, "it enters R, which EM does not update")
        end
    end

    # A parameter shared by several cells needs an M-step that keeps the cells
    # equal. Symmetric pairs (i, j)/(j, i) of a covariance are one cell.
    for (name, (mat, cs)) in cells
        haskey(blocked, name) && continue
        if mat in (:H, :Q, :P1)
            cs_upper = unique(CartesianIndex(minmax(I[1], I[2])) for I in cs)
            length(cs_upper) > 1 || continue
            if mat == :P1
                block!(name, "it is shared by several cells of P1")
            elseif !all(I -> I[1] == I[2], cs_upper)
                block!(name, "it is shared by several cells of $mat, not all on the diagonal")
            end
        elseif mat == :a1 && length(cs) > 1
            block!(name, "it is shared by several cells of a1")
        end
    end
    # Tied diagonal cells of H or Q need a diagonal block.
    for (name, (mat, cs)) in cells
        (mat in (:H, :Q) && !haskey(blocked, name)) || continue
        cs_upper = unique(CartesianIndex(minmax(I[1], I[2])) for I in cs)
        length(cs_upper) > 1 || continue
        offdiag = any(n2 -> cells[n2][1] == mat && any(I -> I[1] != I[2], cells[n2][2]),
            keys(cells))
        offdiag && block!(name,
            "it is shared by several diagonal cells of $mat, which also has free off-diagonal cells")
    end
    for name in keys(blocked)
        delete!(cells, name)
    end

    return _EMLayout(cells, cov, blocked)
end

"""
    set_em_masks!(em_ws::EMWorkspace, spec::SSMSpec) -> em_ws

Set the free-cell masks and tie groups of `em_ws` from `spec`. A cell is free
when the spec fills it with a parameter (`FreeParam`); cells holding fixed
values keep them. Cells that share a parameter become a tie group, and a full
covariance built with `cov_free` frees its whole block.

Parameters that EM cannot update leave their cells fixed: parameters inside a
`MatrixExpr` (such as the decay `λ` of `dns_model`), in `R`, or in an
expression. `fit!(EM(), ...)` throws for those; `profile_em_ssm` profiles one of
them over a grid.
"""
set_em_masks!(em_ws::EMWorkspace, spec::SSMSpec) = _set_em_masks!(em_ws, _em_layout(spec))

function _set_em_masks!(em_ws::EMWorkspace, layout::_EMLayout)
    fill!(em_ws.Z_free, false)
    fill!(em_ws.H_free, false)
    fill!(em_ws.T_free, false)
    fill!(em_ws.Q_free, false)
    fill!(em_ws.a1_free, false)
    fill!(em_ws.P1_free, false)
    fill!(em_ws.H_zero, false)
    fill!(em_ws.Q_zero, false)
    for ties in (em_ws.Z_ties, em_ws.H_ties, em_ws.T_ties, em_ws.Q_ties)
        empty!(ties)
    end

    free = (Z = em_ws.Z_free, H = em_ws.H_free, T = em_ws.T_free, Q = em_ws.Q_free,
        P1 = em_ws.P1_free)
    ties = (Z = em_ws.Z_ties, H = em_ws.H_ties, T = em_ws.T_ties, Q = em_ws.Q_ties)
    for (mat, cs) in values(layout.cells)
        if mat == :a1
            em_ws.a1_free[cs[1][1]] = true
            continue
        end
        for I in cs
            free[mat][I] = true
        end
        if haskey(ties, mat)
            group = unique(mat in (:H, :Q) ?
                           (CartesianIndex(minmax(I[1], I[2])) for I in cs) : cs)
            length(group) > 1 && push!(ties[mat], group)
        end
    end
    haskey(layout.cov, :H) && fill!(em_ws.H_free, true)
    haskey(layout.cov, :Q) && fill!(em_ws.Q_free, true)
    em_ws.H_diag_only = !haskey(layout.cov, :H) &&
                        !any(I -> I[1] != I[2] && em_ws.H_free[I], CartesianIndices(em_ws.H_free))
    return em_ws
end

# Parameter values after an EM run: cells and covariance blocks are read from
# the workspace; every other parameter keeps its value in θ0.
function _em_theta(spec::SSMSpec, layout::_EMLayout, kf_ws::KalmanWorkspace{T},
        θ0::NamedTuple) where {T}
    mats = (Z = kf_ws.Z, H = kf_ws.H, T = kf_ws.Tmat, Q = kf_ws.Q, P1 = kf_ws.P1)
    from_cov = Dict{Symbol, T}()
    for (mat, expr) in layout.cov
        σ, corr = cov_expr_params(expr, mats[mat])
        for (name, v) in zip(expr.σ_param_names, σ)
            from_cov[name] = v
        end
        for (name, v) in zip(expr.corr_param_names, corr)
            from_cov[name] = v
        end
    end
    vals = map(spec.params) do prm
        name = prm.name
        if haskey(layout.cells, name)
            mat, cs = layout.cells[name]
            mat == :a1 ? kf_ws.a1[cs[1][1]] : mats[mat][cs[1]]
        elseif haskey(from_cov, name)
            from_cov[name]
        else
            T(θ0[name])
        end
    end
    return NamedTuple{keys(θ0)}(Tuple(vals))
end

# Run EM on `spec` from θ0 in the workspaces `ws`. Parameters in `held` keep
# their θ0 values; any other parameter that EM cannot update throws.
function _em_fit_spec!(ws::StateSpaceModelWorkspaces{T}, spec::SSMSpec, θ0::NamedTuple,
        y::AbstractMatrix; held = (), caller::AbstractString, kwargs...) where {T}
    layout = _em_layout(spec)
    unestimable = sort!([name for name in keys(layout.blocked) if !(name in held)])
    if !isempty(unestimable)
        reasons = join(("$name ($(layout.blocked[name]))" for name in unestimable), ", ")
        throw(ArgumentError(
            "$caller cannot estimate $reasons. Use fit!(MLE(), ...), or " *
            "profile_em_ssm for a parameter of a matrix expression."))
    end

    kf_ws, em_ws = ws.kf_ws, ws.em_ws
    kfparms = build_kfparms(spec, θ0)
    a1, P1 = build_initial_state(spec, θ0)
    copyto!(kf_ws.Z, kfparms.Z)
    copyto!(kf_ws.H, kfparms.H)
    copyto!(kf_ws.Tmat, kfparms.T)
    copyto!(kf_ws.R, kfparms.R)
    copyto!(kf_ws.Q, kfparms.Q)
    copyto!(kf_ws.a1, a1)
    copyto!(kf_ws.P1, P1)
    _update_RQR!(kf_ws)
    _set_em_masks!(em_ws, layout)

    result = em_estimate!(kf_ws, em_ws, y; kwargs...)
    return _em_theta(spec, layout, kf_ws, θ0), result
end

function _spec_init(spec::SSMSpec, ::Type{T}) where {T}
    NamedTuple{Tuple(prm.name for prm in spec.params)}(Tuple(T(prm.init)
    for prm in spec.params))
end

function _require_em_spec(::EKFSpec, caller::AbstractString)
    throw(ArgumentError(
        "$caller does not support EKF models; use fit!(MLE(), ...)."))
end
function _require_em_spec(spec::SSMSpec, caller::AbstractString)
    uses_exact_diffuse(spec) && throw(ArgumentError(
        "$caller does not support exact diffuse initialisation (diffuse = :exact). " *
        "Use fit!(MLE(), ...), or approximate diffuse initialisation (diffuse = true)."))
    return nothing
end

# ============================================
# StateSpaceModel fit!(EM(), ...)
# ============================================

"""
    fit!(::EM, model::StateSpaceModel, y::AbstractMatrix; kwargs...)

Fit a state-space model with the EM algorithm, starting from the parameters'
initial values.

EM updates every parameter that fills plain cells of `Z`, `H`, `T`, `Q`, `a1`
or `P1`, and full covariances built with `cov_free`. A parameter shared by
several cells keeps them equal. Parameters inside a `MatrixExpr` (such as the
decay `λ` of `dns_model`), in `R`, or in an expression have no closed-form
M-step, and `fit!` throws an `ArgumentError` naming them; use
`fit!(MLE(), ...)`, or `profile_em_ssm` for a matrix-expression parameter.

EM runs on the model's in-place `KalmanWorkspace` for every model size.

# Arguments
- `EM()`: Estimation method selector
- `model`: StateSpaceModel to fit (mutated in-place)
- `y`: Observations (p × n matrix), missing values as NaN or `missing`. Periods
  with some but not all observations missing are not supported.

# Keyword Arguments
- `maxiter::Int=500`: Maximum EM iterations
- `tol::Real=1e-6`: Convergence tolerance (relative change in log-likelihood)
- `verbose::Bool=false`: Log progress with `@info`
- `warn::Bool=true`: Warn when EM stops without converging, either at `maxiter` or
  because the filter failed at the current parameters
- `allow_degen::Bool=false`, `degen_lim::Real=1e-4`, `min_degen_iter::Int=50`:
  fix diagonal variances of `H` or `Q` at zero once they fall below `degen_lim`
  after `min_degen_iter` iterations, when that does not lower the
  log-likelihood

# Returns
The fitted `model` (same object, mutated)

# Example
```julia
spec = local_level()
model = StateSpaceModel(spec, 100)
fit!(EM(), model, randn(1, 100); maxiter=200)
```
"""
function fit!(
        ::EM,
        model::StateSpaceModel{T},
        y::AbstractMatrix;
        maxiter::Int = 500,
        tol::Real = 1e-6,
        verbose::Bool = false,
        warn::Bool = true,
        allow_degen::Bool = false,
        degen_lim::Real = 1e-4,
        min_degen_iter::Int = 50
) where {T}
    y = _nan_obs(y)
    _check_model_obs(model, "fit!(EM(), ...)", y)
    _require_no_partial_periods(y, "fit!(EM(), ...)",
        "Use fit!(MLE(), ...), which handles them.")
    _require_em_spec(model.spec, "fit!(EM(), ...)")

    ws = model.workspaces::StateSpaceModelWorkspaces{T}
    θ, result = _em_fit_spec!(ws, model.spec, _spec_init(model.spec, T), y;
        caller = "fit!(EM(), ...)", maxiter, tol, verbose, warn,
        allow_degen, degen_lim, min_degen_iter)

    copyto!(model.theta_values, values(θ))
    model.theta_fitted = true
    _ssm_copy_from_workspace!(model, ws.kf_ws)
    _ssm_store_data!(model, y)
    model.loglik = result.loglik
    model.converged = result.converged
    model.iterations = result.iterations
    model.fitted = true
    model.smoother_computed = false

    return model
end

# ============================================
# Profile EM
# ============================================

"""
    ProfileEMResult{T<:Real, NT<:NamedTuple}

Result of [`profile_em_ssm`](@ref).

# Fields
- `λ_optimal::T`: grid value with the highest profile log-likelihood
- `θ::NT`: all parameters at the optimum, in the order of the spec
- `loglik::T`: log-likelihood at the optimum
- `em_result::EMResult{T}`: EM run at the optimum
- `λ_grid::Vector{T}`: grid of values searched
- `loglik_profile::Vector{T}`: profile log-likelihood at each grid value
"""
struct ProfileEMResult{T <: Real, NT <: NamedTuple}
    λ_optimal::T
    θ::NT
    loglik::T
    em_result::EMResult{T}
    λ_grid::Vector{T}
    loglik_profile::Vector{T}
end

function Base.show(io::IO, r::ProfileEMResult)
    print(io, "ProfileEMResult(λ_optimal = $(r.λ_optimal), ",
        "loglik = $(round(r.loglik; sigdigits = 8)), $(length(r.λ_grid)) grid values)")
end

"""
    profile_em_ssm(spec::SSMSpec, y::AbstractMatrix; kwargs...) -> ProfileEMResult

Estimate a model by profile likelihood over one parameter that EM cannot update,
typically the decay `λ` of `dns_model`, which enters `Z` through a
`MatrixExpr`. For each value in `λ_grid`, EM (as in `fit!(EM(), ...)`)
estimates the other parameters with `λ` held fixed; the result is the grid value
with the highest log-likelihood.

Every other parameter that EM cannot update must be given a value in `fixed`;
for example the second decay `λ2` of a Svensson model.

# Keyword Arguments
- `λ_grid = 0.01:0.005:0.2`: values of the profiled parameter
- `λ_param::Symbol = :λ`: name of the profiled parameter
- `fixed::NamedTuple = (;)`: values of other parameters that EM cannot update
- `maxiter::Int = 500`: maximum EM iterations per grid value
- `tol::Real = 1e-6`: EM convergence tolerance (relative change in log-likelihood)
- `warm_start::Bool = true`: start each EM run from the previous grid value's
  estimates
- `verbose::Bool = false`: log each grid value with `@info`
- `warn::Bool = true`: warn when EM does not converge at some grid values

# Example
```julia
spec = dns_model(maturities; T_structure = :diagonal, Q_structure = :full)
result = profile_em_ssm(spec, yields; λ_grid = 0.02:0.005:0.15)
result.λ_optimal
ssm_loglik(spec, result.θ, yields) ≈ result.loglik
```
"""
function profile_em_ssm(
        spec::SSMSpec,
        y::AbstractMatrix;
        λ_grid = 0.01:0.005:0.2,
        λ_param::Symbol = :λ,
        fixed::NamedTuple = (;),
        maxiter::Int = 500,
        tol::Real = 1e-6,
        warm_start::Bool = true,
        verbose::Bool = false,
        warn::Bool = true
)
    y = _nan_obs(y)
    size(y, 1) == spec.n_obs || throw(DimensionMismatch(
        "profile_em_ssm: y has $(size(y, 1)) rows but the spec has $(spec.n_obs) observables"))
    _require_no_partial_periods(y, "profile_em_ssm",
        "Use optimize_ssm, which handles them.")
    _require_em_spec(spec, "profile_em_ssm")

    names = Tuple(prm.name for prm in spec.params)
    λ_param in names ||
        throw(ArgumentError("λ_param = :$λ_param is not a parameter of the spec"))
    haskey(_em_layout(spec).blocked, λ_param) || throw(ArgumentError(
        "profile_em_ssm profiles a parameter that EM cannot update, but EM updates " *
        ":$λ_param. Use fit!(EM(), ...)."))
    for name in keys(fixed)
        name in names ||
            throw(ArgumentError("fixed: :$name is not a parameter of the spec"))
        name == λ_param &&
            throw(ArgumentError("fixed must not contain λ_param = :$λ_param"))
    end
    λ_vec = collect(Float64, λ_grid)
    isempty(λ_vec) && throw(ArgumentError("λ_grid is empty"))

    θ_init = merge(_spec_init(spec, Float64), map(Float64, fixed))
    held = (λ_param, keys(fixed)...)
    with_λ(θ, λ) = merge(θ, NamedTuple{(λ_param,)}((λ,)))

    kfparms = build_kfparms(spec, θ_init)
    a1, P1 = build_initial_state(spec, θ_init)
    kf_ws = KalmanWorkspace(kfparms.Z, kfparms.H, kfparms.T, kfparms.R, kfparms.Q,
        a1, P1, size(y, 2))
    ws = StateSpaceModelWorkspaces{Float64}(kf_ws, EMWorkspace(kf_ws))

    loglik_profile = Vector{Float64}(undef, length(λ_vec))
    converged = falses(length(λ_vec))
    best = 0
    θ_best, result_best = θ_init, nothing
    θ_prev = θ_init
    for (k, λ) in enumerate(λ_vec)
        θ_start = with_λ(warm_start ? θ_prev : θ_init, λ)
        θ, result = _em_fit_spec!(ws, spec, θ_start, y; held,
            caller = "profile_em_ssm", maxiter, tol, warn = false)
        loglik_profile[k] = result.loglik
        converged[k] = result.converged
        verbose &&
            @info "profile_em_ssm" λ loglik=result.loglik converged=result.converged iterations=result.iterations
        isfinite(result.loglik) || continue
        θ_prev = θ
        if best == 0 || result.loglik > loglik_profile[best]
            best, θ_best, result_best = k, θ, result
        end
    end
    best == 0 && throw(ArgumentError(
        "profile_em_ssm: the filter failed at every value in λ_grid. " *
        "Try other initial values in the spec."))
    if warn && !all(converged)
        @warn "profile_em_ssm: EM did not converge at $(count(!, converged)) of " *
              "$(length(λ_vec)) grid values (λ = $(λ_vec[.!converged])). Increase " *
              "`maxiter` or pass `warn = false` to silence this warning."
    end

    return ProfileEMResult(λ_vec[best], θ_best, loglik_profile[best], result_best,
        λ_vec, loglik_profile)
end
