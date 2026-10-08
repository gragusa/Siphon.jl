"""
    optimization.jl

Integration with Optimization.jl for maximum likelihood estimation of state-space models.

Provides convenient functions to optimize SSM parameters using various backends
(L-BFGS, Newton, etc.) with automatic differentiation.
"""

using Optimization
using OptimizationOptimJL
using ForwardDiff

export optimize_ssm

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
  their spec initial values). Default: `initial_values(spec)`
- `ad_backend`: AD backend for gradients (default: `Optimization.AutoForwardDiff()`)
- `use_static::Bool=true`: Use StaticArrays for small matrices (dimensions ≤ 13)
- `prob_kwargs`: NamedTuple of kwargs passed to `OptimizationProblem`
- `kwargs...`: All other kwargs passed to `Optimization.solve`

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

# Different optimizer with options
result3 = optimize_ssm(spec, y;
    method=Optim.Newton(),
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
result6 = optimize_ssm(spec, y; use_static=false)
```
"""
function optimize_ssm(
        spec::SSMSpec,
        y::AbstractMatrix;
        method = Optim.LBFGS(),
        θ0::Union{Nothing, AbstractVector, NamedTuple} = nothing,
        ad_backend = Optimization.AutoForwardDiff(),
        use_static::Bool = true,
        prob_kwargs::NamedTuple = NamedTuple(),
        kwargs...
)
    return _mle(spec, y, method, θ0, ad_backend, use_static, prob_kwargs; kwargs...)
end

# Maximum likelihood shared by `optimize_ssm` and `optimize_ekf`. The search runs
# in unconstrained coordinates, but the objective is the log-likelihood alone:
# the log-Jacobian of the transform belongs to a density over parameters, not to
# the likelihood, and including it would move the optimum.
function _mle(spec, y, method, θ0, ad_backend, use_static, prob_kwargs; kwargs...)
    t = build_transformation(spec)
    θ0_u = transform_to_unconstrained(spec, _initial_constrained(spec, θ0))

    negloglik(θ_u, _) = -_spec_loglik(spec, TransformVariables.transform(t, θ_u), y,
        use_static)
    optf = OptimizationFunction(negloglik, ad_backend)
    prob = OptimizationProblem(optf, θ0_u; prob_kwargs...)
    result = Optimization.solve(prob, method; kwargs...)

    return (
        θ = TransformVariables.transform(t, result.u),
        loglik = -result.objective,
        result = result,
        converged = result.retcode == Optimization.SciMLBase.ReturnCode.Success
    )
end

# Starting values in constrained space, in `param_names(spec)` order. A
# NamedTuple may give any subset of the parameters; the rest start at their
# spec initial values.
function _initial_constrained(spec, θ0)
    θ0 === nothing && return initial_values(spec)
    θ0 isa NamedTuple || return θ0
    names = param_names(spec)
    unknown = setdiff(keys(θ0), names)
    isempty(unknown) ||
        throw(ArgumentError("θ0 has unknown parameters $(Tuple(unknown)); " *
                            "the model's parameters are $(Tuple(names))"))
    init = initial_values(spec)
    return [haskey(θ0, n) ? Float64(θ0[n]) : init[i] for (i, n) in enumerate(names)]
end

"""
    optimize_ssm_with_stderr(spec, y; use_static=true, kwargs...)

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
        use_static::Bool = true,
        kwargs...
)
    opt_result = optimize_ssm(spec, y; use_static = use_static, kwargs...)

    if !opt_result.converged
        @warn "Optimization did not converge; standard errors may be unreliable"
    end

    t = build_transformation(spec)
    θ_u_opt = transform_to_unconstrained(spec, opt_result.θ)
    to_θ(θ_u) = collect(TransformVariables.transform(t, θ_u))

    H = ForwardDiff.hessian(
        θ_u -> -_spec_loglik(spec, TransformVariables.transform(t, θ_u), y, use_static),
        θ_u_opt)
    J = ForwardDiff.jacobian(to_θ, θ_u_opt)

    vcov, se = try
        V = J * inv(H) * J'
        V, sqrt.(diag(V))
    catch e
        e isa LinearAlgebra.SingularException || e isa DomainError || rethrow()
        @warn "Failed to compute standard errors: the information matrix is not positive definite"
        k = length(θ_u_opt)
        fill(NaN, k, k), fill(NaN, k)
    end

    return (
        θ = opt_result.θ,
        loglik = opt_result.loglik,
        result = opt_result.result,
        converged = opt_result.converged,
        stderr = se,
        vcov = vcov,
        hessian = H
    )
end
