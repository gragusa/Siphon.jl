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
- `θ0`: Initial parameter values (constrained). Default: `initial_values(spec)`
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
- `loglik`: Log-likelihood at optimum
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
result2 = optimize_ssm(spec, y; θ0=(σ_obs=150.0, σ_level=50.0))

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

    # Create log-density object
    ld = SSMLogDensity(spec, y; use_static = use_static)

    # Initial values in constrained space
    θ0_c = if θ0 === nothing
        initial_values(spec)
    elseif θ0 isa NamedTuple
        collect(Float64, θ0)
    else
        θ0
    end

    # Transform to unconstrained space
    θ0_u = transform_to_unconstrained(spec, θ0_c)

    # Negative log-density for minimization
    neglogdensity(θ_u, p) = -logdensity(ld, θ_u)

    # Create OptimizationFunction with AD
    optf = OptimizationFunction(neglogdensity, ad_backend)

    # Create problem with optional kwargs
    prob = OptimizationProblem(optf, θ0_u; prob_kwargs...)

    # Solve with user kwargs
    result = Optimization.solve(prob, method; kwargs...)

    # Transform back to constrained space
    θ_opt, _ = transform_to_constrained(spec, result.u)
    ll_opt = -result.objective

    return (
        θ = θ_opt,
        loglik = ll_opt,
        result = result,
        converged = result.retcode == Optimization.SciMLBase.ReturnCode.Success
    )
end

"""
    optimize_ssm_with_stderr(spec, y; use_static=true, kwargs...)

Optimize SSM and compute standard errors using the Hessian.

Returns the same as `optimize_ssm` plus:
- `stderr`: Standard errors of parameters (in constrained space, approximate)
- `hessian`: Hessian matrix at the optimum (in unconstrained space)

Note: Standard errors are approximate when using parameter transformations.
For accurate standard errors on constrained parameters, use the delta method
or bootstrap.
"""
function optimize_ssm_with_stderr(
        spec::SSMSpec,
        y::AbstractMatrix;
        use_static::Bool = true,
        kwargs...
)
    # First optimize
    opt_result = optimize_ssm(spec, y; use_static = use_static, kwargs...)

    if !opt_result.converged
        @warn "Optimization did not converge; standard errors may be unreliable"
    end

    # Compute Hessian at optimum (in unconstrained space)
    ld = SSMLogDensity(spec, y; use_static = use_static)
    θ_u_opt = transform_to_unconstrained(spec, opt_result.θ)

    H = ForwardDiff.hessian(θ -> -logdensity(ld, θ), θ_u_opt)

    # Standard errors from inverse Hessian
    # This is in unconstrained space
    try
        Hinv = inv(H)
        stderr_u = sqrt.(diag(Hinv))

        # Transform to constrained space (approximate via chain rule)
        # For exp transform: se_c ≈ θ_c * se_u
        stderr_c = similar(stderr_u)
        for (i, p) in enumerate(spec.params)
            if p.lower > -Inf && p.upper == Inf
                # Exp transform: ∂θ_c/∂θ_u = θ_c - p.lower
                stderr_c[i] = (opt_result.θ[i] - p.lower) * stderr_u[i]
            elseif p.lower == -Inf && p.upper == Inf
                # Identity
                stderr_c[i] = stderr_u[i]
            else
                # Bounded: use unconstrained stderr as approximation
                stderr_c[i] = stderr_u[i]
            end
        end

        return (
            θ = opt_result.θ,
            loglik = opt_result.loglik,
            result = opt_result.result,
            converged = opt_result.converged,
            stderr = stderr_c,
            hessian = H
        )
    catch e
        @warn "Failed to compute standard errors: $(e)"
        return (
            θ = opt_result.θ,
            loglik = opt_result.loglik,
            result = opt_result.result,
            converged = opt_result.converged,
            stderr = fill(NaN, length(opt_result.θ)),
            hessian = H
        )
    end
end
