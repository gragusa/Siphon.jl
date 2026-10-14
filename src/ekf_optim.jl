"""
    ekf_optim.jl

The `SSMLogDensity` constructor and the `optimize_ssm` method (Optimization.jl-backed
MLE) for `EKFSpec`.
"""

# ============================================================================
# SSMLogDensity
# ============================================================================

function SSMLogDensity(spec::EKFSpec, y::AbstractMatrix; prior = FlatPrior())
    y = _nan_obs(y)
    _check_prior(prior, spec)
    t = build_transformation(spec)
    return SSMLogDensity(spec, t, y, prior, :off)
end

# The EKF has no StaticArrays specialization; `static` is always `:off` here.
function _spec_loglik(spec::EKFSpec, θ::NamedTuple, y::AbstractMatrix, static::Symbol)
    ss = build_nonlinear_state_space(spec, θ, y)
    return kalman_loglik(ss.p, y, ss.a1, ss.P1)
end

# ============================================================================
# optimize_ssm for EKFSpec
# ============================================================================

"""
    optimize_ssm(spec::EKFSpec, y; method=Optim.LBFGS(), θ0=nothing,
                 ad_backend=Optimization.AutoForwardDiff(),
                 prob_kwargs=NamedTuple(), warn=true, kwargs...)

Maximum-likelihood estimation for an `EKFSpec`, with the same keywords and return
value as for an `SSMSpec`, except `static`: the EKF has no StaticArrays
specialization. Works in unconstrained parameter space via
`TransformVariables`.

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
function optimize_ssm(spec::EKFSpec, y::AbstractMatrix;
        method = Optim.LBFGS(),
        θ0::Union{Nothing, AbstractVector, NamedTuple} = nothing,
        ad_backend = Optimization.AutoForwardDiff(),
        prob_kwargs::NamedTuple = NamedTuple(),
        warn::Bool = true,
        kwargs...)
    y = _nan_obs(y)
    return _mle(spec, y, method, θ0, ad_backend, :off, prob_kwargs; warn, kwargs...)
end
