"""
    ekf_optim.jl

`EKFLogDensity` (LogDensityProblems-compatible) and `optimize_ekf`
(Optimization.jl-backed MLE) for `EKFSpec`. Mirrors `SSMLogDensity` /
`optimize_ssm`.
"""

# ============================================================================
# EKFLogDensity
# ============================================================================

"""
    EKFLogDensity(spec::EKFSpec, y; use_static=true)

Log-density evaluator for an `EKFSpec`, working in unconstrained ℝⁿ via the
`TransformVariables` transformation built from `spec.params`. Pairs with
`logdensity(ld, θ_u)` and integrates with `LogDensityProblems`.

Behaviour and contract match `SSMLogDensity` for the linear case.
"""
struct EKFLogDensity{S <: EKFSpec, Tr, Y <: AbstractMatrix}
    spec::S
    transformation::Tr
    y::Y
    use_static::Bool
end

function EKFLogDensity(spec::EKFSpec, y::AbstractMatrix; use_static::Bool = true)
    y = _nan_obs(y)
    t = build_transformation(spec)
    return EKFLogDensity{typeof(spec), typeof(t), typeof(y)}(spec, t, y, use_static)
end

# logdensity is defined in bayesian.jl as a generic function operating on
# SSMLogDensity. We add an EKF method here.
function logdensity(ld::EKFLogDensity, θ_u::AbstractVector)
    θ_nt, logjac = TransformVariables.transform_and_logjac(ld.transformation, θ_u)
    return _spec_loglik(ld.spec, θ_nt, ld.y, ld.use_static) + logjac
end

function _spec_loglik(spec::EKFSpec, θ::NamedTuple, y::AbstractMatrix, use_static::Bool)
    return ekf_loglik(spec, θ, y; use_static = use_static)
end

# Optional LogDensityProblems integration
LogDensityProblems.logdensity(ld::EKFLogDensity, θ_u::AbstractVector) = logdensity(
    ld, θ_u)
LogDensityProblems.dimension(ld::EKFLogDensity) = n_params(ld.spec)
function LogDensityProblems.capabilities(::Type{<:EKFLogDensity})
    LogDensityProblems.LogDensityOrder{0}()
end

# ============================================================================
# optimize_ekf
# ============================================================================

"""
    optimize_ekf(spec::EKFSpec, y; method=Optim.LBFGS(), θ0=nothing,
                 ad_backend=Optimization.AutoForwardDiff(),
                 use_static=true, prob_kwargs=NamedTuple(), warn=true, kwargs...)

Maximum-likelihood estimation for an `EKFSpec`. Works in unconstrained
parameter space via `TransformVariables`. Mirrors `optimize_ssm`.

# AD-backend choice
The default `AutoForwardDiff()` requires that the entire likelihood path is
ForwardDiff-compatible: in particular, the user's measurement methods must
accept `Dual` numbers. For nonsmooth measurements (e.g. B-DNS kink) or when
ForwardDiff propagation through user code is fragile, pass
`ad_backend = Optimization.AutoFiniteDiff()` for finite-difference parameter
gradients. The choice of `ad_backend` here affects only the *parameter*
gradient; the *state* Jacobian is controlled by `spec.jacobian_mode`.

# Returns
A NamedTuple with `θ` (constrained), `loglik`, `result`, and `converged`. A warning is
emitted when the optimiser does not report convergence, unless `warn = false`.
"""
function optimize_ekf(spec::EKFSpec, y::AbstractMatrix;
        method = Optim.LBFGS(),
        θ0::Union{Nothing, AbstractVector, NamedTuple} = nothing,
        ad_backend = Optimization.AutoForwardDiff(),
        use_static::Bool = true,
        prob_kwargs::NamedTuple = NamedTuple(),
        warn::Bool = true,
        kwargs...)
    y = _nan_obs(y)
    return _mle(spec, y, method, θ0, ad_backend, use_static, prob_kwargs; warn, kwargs...)
end
