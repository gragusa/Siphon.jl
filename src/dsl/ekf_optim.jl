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
    t = build_transformation(spec)
    return EKFLogDensity{typeof(spec), typeof(t), typeof(y)}(spec, t, y, use_static)
end

# logdensity is defined in bayesian.jl as a generic function operating on
# SSMLogDensity. We add an EKF method here.
function logdensity(ld::EKFLogDensity, θ_u::AbstractVector)
    θ_nt, logjac = TransformVariables.transform_and_logjac(ld.transformation, θ_u)
    ll = ekf_loglik(ld.spec, θ_nt, ld.y; use_static = ld.use_static)
    return ll + logjac
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
                 use_static=true, prob_kwargs=NamedTuple(), kwargs...)

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
A NamedTuple with `θ` (constrained), `loglik`, `result`, and `converged`.
"""
function optimize_ekf(spec::EKFSpec, y::AbstractMatrix;
        method = Optim.LBFGS(),
        θ0::Union{Nothing, AbstractVector, NamedTuple} = nothing,
        ad_backend = Optimization.AutoForwardDiff(),
        use_static::Bool = true,
        prob_kwargs::NamedTuple = NamedTuple(),
        kwargs...)
    ld = EKFLogDensity(spec, y; use_static = use_static)

    θ0_c = if θ0 === nothing
        initial_values(spec)
    elseif θ0 isa NamedTuple
        collect(Float64, θ0)
    else
        θ0
    end
    θ0_u = transform_to_unconstrained(spec, θ0_c)

    neglogdensity(θ_u, _) = -logdensity(ld, θ_u)
    optf = OptimizationFunction(neglogdensity, ad_backend)
    prob = OptimizationProblem(optf, θ0_u; prob_kwargs...)
    result = Optimization.solve(prob, method; kwargs...)

    θ_opt, _ = transform_to_constrained(spec, result.u)
    ll_opt = -result.objective
    return (
        θ = θ_opt,
        loglik = ll_opt,
        result = result,
        converged = result.retcode == Optimization.SciMLBase.ReturnCode.Success
    )
end
