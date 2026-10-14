"""
    optimization.jl

Integration with Optimization.jl for maximum likelihood estimation of state-space models.

Provides convenient functions to optimize SSM parameters using various backends
(L-BFGS, Newton, etc.) with automatic differentiation.
"""

"""
    optimize_ssm(spec, y; method=Optim.LBFGS(), kwargs...)

Optimize state-space model parameters using Optimization.jl.

Works in unconstrained parameter space using automatic differentiation
for gradient computation.

# Arguments
- `spec::SSMSpec`: Model specification
- `y::AbstractMatrix`: Observations (p × n matrix)
- `method`: Optimization algorithm (default: L-BFGS)
- `θ0`: Initial parameter values (constrained). A vector in `param_names(spec)` order,
  or a NamedTuple giving any subset of the parameters by name (the rest start at
  their default starting values). Default: the spec initial values, with template
  default variances rescaled to the data (see below)
- `ad_backend`: AD backend for gradients (default: `Optimization.AutoForwardDiff()`)
- `static::Symbol=:auto`: `:auto`, `:on` or `:off`, passed to [`build_linear_state_space`](@ref)
- `prob_kwargs`: NamedTuple of kwargs passed to `OptimizationProblem`
- `warn::Bool=true`: Warn when the optimiser does not report convergence
- `kwargs...`: All other kwargs passed to `Optimization.solve`

# Default starting values
A variance on the diagonal of `H` or `Q` whose initial value was left unset (for
example `local_level()` rather than `local_level(var_obs = FreeParam(init = …))`)
starts at its default times the variance of the first differences of `y`. An
observation variance uses the series it applies to; a state variance uses the mean
over series. Differences keep the scale meaningful for trending series, where
`var(y)` is dominated by the trend. Initial values set by the user, and all other
parameters, are used as given.

# Common solve kwargs
- `maxiters`: Maximum iterations (default: 1000)
- `maxtime`: Maximum time in seconds
- `abstol`: Absolute tolerance
- `reltol`: Relative tolerance
- `callback`: Callback function `(state, loss) -> Bool` (return true to stop)
- `progress`: Show progress bar (requires ProgressLogging.jl)
- `show_trace`: Show optimization trace (Optim.jl specific)

# Returns
Named tuple with:
- `θ`: Optimal parameters (constrained space)
- `loglik`: Log-likelihood at the optimum (the maximised log-likelihood)
- `result`: Full Optimization.jl result object
- `converged`: Whether optimization converged

# Example

```julia
spec = local_level()
y = randn(1, 100) .* 10 .+ 100

# Basic usage
result = optimize_ssm(spec, y)

println("Optimal parameters: ", result.θ)
println("Log-likelihood: ", result.loglik)

# With custom initial values
result2 = optimize_ssm(spec, y; θ0=(var_obs=150.0^2, var_level=50.0^2))

# Different optimizer with options (Optim.jl, through OptimizationOptimJL)
using OptimizationOptimJL: Optim
result3 = optimize_ssm(spec, y;
    method=Optim.BFGS(),
    maxiters=500,
    show_trace=true
)

# With callback for monitoring
callback = (state, loss) -> begin
    println("Iteration: loss = \$loss")
    return false  # return true to stop
end
result4 = optimize_ssm(spec, y; callback=callback)

# Pass kwargs to OptimizationProblem (e.g., bounds in unconstrained space)
result5 = optimize_ssm(spec, y;
    prob_kwargs=(lb=fill(-10.0, n_params(spec)),
                 ub=fill(10.0, n_params(spec)))
)

# Disable StaticArrays (for debugging or if causing issues)
result6 = optimize_ssm(spec, y; static=:off)
```
"""
function optimize_ssm(
        spec::SSMSpec,
        y::AbstractMatrix;
        method = Optim.LBFGS(),
        θ0::Union{Nothing, AbstractVector, NamedTuple} = nothing,
        ad_backend = Optimization.AutoForwardDiff(),
        static::Symbol = :auto,
        prob_kwargs::NamedTuple = NamedTuple(),
        warn::Bool = true,
        kwargs...
)
    y = _nan_obs(y)
    return _mle(spec, y, method, θ0, ad_backend, static, prob_kwargs; warn, kwargs...)
end

# Maximum likelihood shared by the `optimize_ssm` methods for `SSMSpec` and `EKFSpec`.
# The search runs in unconstrained coordinates, but the objective is the log-likelihood alone:
# the log-Jacobian of the transform belongs to a density over parameters, not to
# the likelihood, and including it would move the optimum.
function _mle(spec, y, method, θ0, ad_backend, static, prob_kwargs; warn::Bool = true,
        kwargs...)
    t = build_transformation(spec)
    θ0_u = transform_to_unconstrained(spec, _initial_constrained(spec, θ0, y))

    negloglik(θ_u, _) = -_spec_loglik(spec, TransformVariables.transform(t, θ_u), y, static)
    # From a non-finite start the optimizer cannot move and returns θ0 with a
    # success code.
    ll0 = -negloglik(θ0_u, nothing)
    isfinite(ll0) ||
        throw(ArgumentError("the log-likelihood at the starting values is $ll0 (the " *
                            "filter failed, e.g. an innovation covariance that is not " *
                            "positive definite), so maximum likelihood cannot start. " *
                            "Starting values: $(TransformVariables.transform(t, θ0_u)). " *
                            "Pass other starting values with θ0."))
    optf = OptimizationFunction(negloglik, ad_backend)
    prob = OptimizationProblem(optf, θ0_u; prob_kwargs...)
    result = Optimization.solve(prob, method; kwargs...)
    converged = result.retcode == Optimization.SciMLBase.ReturnCode.Success &&
                isfinite(result.objective)
    if warn && !converged
        @warn "Maximum likelihood optimisation did not converge (return code " *
              "$(result.retcode), log-likelihood $(-result.objective)). Try other " *
              "starting values (θ0) or a larger `maxiters`. " *
              "Pass `warn = false` to silence this warning."
    end

    return (
        θ = TransformVariables.transform(t, result.u),
        loglik = -result.objective,
        result = result,
        converged = converged
    )
end

# Starting values in constrained space, in `param_names(spec)` order. A
# NamedTuple may give any subset of the parameters; the rest start at their
# default starting values.
function _initial_constrained(spec, θ0, y)
    θ0 isa AbstractVector && return θ0
    init = _default_initial_values(spec, y)
    θ0 === nothing && return init
    names = param_names(spec)
    unknown = setdiff(keys(θ0), names)
    isempty(unknown) ||
        throw(ArgumentError("θ0 has unknown parameters $(Tuple(unknown)); " *
                            "the model's parameters are $(Tuple(names))"))
    return [haskey(θ0, n) ? Float64(θ0[n]) : init[i] for (i, n) in enumerate(names)]
end

# Spec initial values, with each template-default variance on the diagonal of H
# or Q multiplied by the variance of the first differences of y. Template
# variances default to 1.0, and from there L-BFGS can stop at a degenerate
# optimum when the data's scale is far from 1.
_default_initial_values(spec, y) = initial_values(spec)

function _default_initial_values(spec::SSMSpec, y)
    init = initial_values(spec)
    any(p -> p.default_init, spec.params) || return init
    s2 = [_diff_variance(view(y, i, :)) for i in axes(y, 1)]
    for (k, p) in enumerate(spec.params)
        p.default_init || continue
        ref = ParameterRef(p.name)
        rows = [i for i in 1:spec.n_obs if get(spec.H.elements, (i, i), nothing) == ref]
        scale = if !isempty(rows)
            mean(s2[rows])
        elseif any(j -> get(spec.Q.elements, (j, j), nothing) == ref, 1:spec.n_shocks)
            mean(s2)
        else
            continue
        end
        v = p.init * scale
        p.lower < v < p.upper && (init[k] = v)
    end
    return init
end

# Variance of the differences between consecutive observed values of a series,
# falling back to the variance of the observed values and then to 1.0.
function _diff_variance(x)
    obs = filter(!isnan, x)
    for v in (length(obs) > 2 ? var(diff(obs)) : NaN, length(obs) > 1 ? var(obs) : NaN)
        isfinite(v) && v > 0 && return v
    end
    return 1.0
end

"""
    optimize_ssm_with_stderr(spec, y; static=:auto, kwargs...)

Optimize SSM and compute standard errors from the observed information.

Returns the same as `optimize_ssm` plus:
- `stderr`: standard errors of the parameters (constrained space)
- `vcov`: covariance matrix of the parameters (constrained space)
- `hessian`: Hessian of the negative log-likelihood (unconstrained space)

The Hessian is taken in the unconstrained coordinates the optimizer uses and
mapped to the constrained parameters with the delta method,
`vcov = J H⁻¹ Jᵀ`, where `J` is the Jacobian of the parameter transform.
"""
function optimize_ssm_with_stderr(
        spec::SSMSpec,
        y::AbstractMatrix;
        static::Symbol = :auto,
        warn::Bool = true,
        kwargs...
)
    y = _nan_obs(y)
    opt_result = optimize_ssm(spec, y; static, warn = false, kwargs...)

    if warn && !opt_result.converged
        @warn "Maximum likelihood optimisation did not converge (return code " *
              "$(opt_result.result.retcode)); the standard errors may be unreliable. " *
              "Pass `warn = false` to silence this warning."
    end

    V, H = _param_vcov(spec, opt_result.θ, y, static)

    return (
        θ = opt_result.θ,
        loglik = opt_result.loglik,
        result = opt_result.result,
        converged = opt_result.converged,
        stderr = sqrt.(diag(V)),
        vcov = V,
        hessian = H
    )
end

# Covariance of the constrained parameters at θ from the observed information,
# and the Hessian of the negative log-likelihood in the unconstrained
# coordinates. The covariance is NaN, with a warning, when that Hessian cannot be
# inverted or the result has a negative diagonal.
function _param_vcov(spec::AbstractSSMSpec, θ::NamedTuple, y::AbstractMatrix,
        static::Symbol)
    t = build_transformation(spec)
    θ_u = transform_to_unconstrained(spec, θ)
    to_θ(x) = collect(TransformVariables.transform(t, x))

    H = ForwardDiff.hessian(
        x -> -_spec_loglik(spec, TransformVariables.transform(t, x), y, static), θ_u)
    J = ForwardDiff.jacobian(to_θ, θ_u)

    V = try
        V = J * inv(H) * J'
        all(>=(0), diag(V)) || throw(DomainError(diag(V), "negative variance"))
        V
    catch e
        e isa LinearAlgebra.SingularException || e isa DomainError || rethrow()
        @warn "Failed to compute standard errors: the information matrix is not positive definite"
        k = length(θ_u)
        fill(NaN, k, k)
    end
    return V, H
end
