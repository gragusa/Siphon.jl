"""
    bayesian.jl

Integration with LogDensityProblems.jl and TransformVariables.jl
for likelihood-based estimation of state-space models.

The key design:
- SSMSpec defines parameters with bounds (lower, upper)
- build_transformation(spec) creates a TransformVariables transformation
- The transformation maps ℝⁿ → NamedTuple with constrained parameters
- LogDensityProblems works in unconstrained ℝⁿ space
"""

using LogDensityProblems
using TransformVariables

export SSMLogDensity
export logdensity
export transform_to_constrained, transform_to_unconstrained
export build_transformation
export FlatPrior, NormalPrior, InverseGammaPrior, CompositePrior

# ============================================
# Transformation Functions (using TransformVariables)
# ============================================

"""
    transform_to_constrained(spec, θ_unconstrained)

Transform from unconstrained ℝⁿ to constrained parameter space.
Returns (θ_constrained::NamedTuple, logjac::Real).

Uses TransformVariables.jl for the transformation.
"""
function transform_to_constrained(spec::SSMSpec, θ_u::AbstractVector)
    t = build_transformation(spec)
    θ_nt, logjac = TransformVariables.transform_and_logjac(t, θ_u)
    (θ_nt, logjac)
end

"""
    transform_to_unconstrained(spec, θ_constrained::NamedTuple)

Transform from constrained NamedTuple to unconstrained ℝⁿ.
Inverse of `transform_to_constrained`.
"""
function transform_to_unconstrained(spec::SSMSpec, θ_nt::NamedTuple)
    t = build_transformation(spec)
    TransformVariables.inverse(t, θ_nt)
end

"""
    transform_to_unconstrained(spec, θ_constrained::AbstractVector)

Transform from constrained vector to unconstrained ℝⁿ.
First converts vector to NamedTuple, then inverts.
"""
function transform_to_unconstrained(spec::SSMSpec, θ_c::AbstractVector)
    # Convert vector to NamedTuple
    names = Tuple(p.name for p in spec.params)
    θ_nt = NamedTuple{names}(Tuple(θ_c))
    transform_to_unconstrained(spec, θ_nt)
end

# ============================================
# Log-Density Type
# ============================================

"""
    SSMLogDensity(spec, y; prior=nothing, use_static=true)

Log-density for a state-space model, evaluated in UNCONSTRAINED ℝⁿ space.

This integrates seamlessly with LogDensityProblems.jl for optimization and
sampling. The transformation from unconstrained to constrained space is
handled automatically using TransformVariables.jl.

# Fields
- `spec`: Model specification (SSMSpec)
- `transformation`: TransformVariables transformation (ℝⁿ → NamedTuple)
- `y`: Observation data (p × n matrix)
- `prior`: Optional prior on CONSTRAINED parameters (θ::NamedTuple -> log_prior)
- `use_static`: Whether to use StaticArrays for small matrices (default: true)

# Usage

```julia
spec = local_level()
y = randn(1, 100)

# Create log-density (works in unconstrained space)
ld = SSMLogDensity(spec, y)

# Get initial point in unconstrained space
θ0 = transform_to_unconstrained(spec, initial_values(spec))

# Evaluate log-density
logdensity(ld, θ0)

# For optimization (in unconstrained space, no bounds needed!)
using Optim
result = optimize(θ -> -logdensity(ld, θ), θ0, LBFGS())

# Transform result back to NamedTuple
θ_hat, _ = transform_to_constrained(spec, result.minimizer)
# θ_hat is a NamedTuple like (σ_obs = 12.3, σ_level = 3.4)
```
"""
struct SSMLogDensity{S<:SSMSpec, T, Y<:AbstractMatrix, P}
    spec::S
    transformation::T
    y::Y
    prior::P
    use_static::Bool
end

function SSMLogDensity(spec::SSMSpec, y::AbstractMatrix; prior=nothing, use_static::Bool=true)
    t = build_transformation(spec)
    SSMLogDensity(spec, t, y, prior, use_static)
end

"""
    logdensity(ld::SSMLogDensity, θ_unconstrained)

Evaluate the log-density at unconstrained parameters.

Returns log p(y|θ) + log p(θ) + log|J| where J is the Jacobian
of the transformation from unconstrained to constrained space.

The parameters are automatically transformed to a NamedTuple and
passed to the likelihood function.
"""
function logdensity(ld::SSMLogDensity, θ_u::AbstractVector)
    # Transform to NamedTuple (constrained space)
    θ_nt, logjac = TransformVariables.transform_and_logjac(ld.transformation, θ_u)

    # Build state-space components (with optional StaticArrays)
    ss = build_linear_state_space(ld.spec, θ_nt, ld.y; use_static=ld.use_static)

    # Compute log-likelihood
    ll = kalman_loglik(ss.p, ld.y, ss.a1, ss.P1)

    # Add prior (on constrained parameters)
    if ld.prior !== nothing
        ll += ld.prior(θ_nt)
    end

    # Add Jacobian for change of variables
    return ll + logjac
end

# ============================================
# LogDensityProblems.jl Interface
# ============================================

LogDensityProblems.capabilities(::Type{<:SSMLogDensity}) =
    LogDensityProblems.LogDensityOrder{0}()

LogDensityProblems.dimension(ld::SSMLogDensity) = n_params(ld.spec)

LogDensityProblems.logdensity(ld::SSMLogDensity, θ::AbstractVector) =
    logdensity(ld, θ)

# ============================================
# Prior Types
# ============================================

"""
    FlatPrior()

Improper flat prior (log-density = 0 everywhere).
Works with both Vector and NamedTuple arguments.
"""
struct FlatPrior end
(::FlatPrior)(θ) = 0.0

"""
    NormalPrior(μ::NamedTuple, σ::NamedTuple)

Independent normal priors for parameters, specified by name.

# Example
```julia
prior = NormalPrior(
    (σ_obs = 10.0, σ_level = 5.0),   # means
    (σ_obs = 5.0, σ_level = 2.0)     # standard deviations
)
```
"""
struct NormalPrior{M<:NamedTuple, S<:NamedTuple}
    μ::M
    σ::S
end

function (p::NormalPrior)(θ::NamedTuple)
    result = 0.0
    for name in keys(p.μ)
        μ_i = getproperty(p.μ, name)
        σ_i = getproperty(p.σ, name)
        θ_i = getproperty(θ, name)
        result += -0.5 * ((θ_i - μ_i) / σ_i)^2 - log(σ_i)
    end
    result - length(keys(p.μ)) * 0.5 * log(2π)
end

# Also support vector-based priors for backwards compatibility
struct NormalPriorVec{M<:AbstractVector, S<:AbstractVector}
    μ::M
    σ::S
end

function (p::NormalPriorVec)(θ::NamedTuple)
    θ_vec = collect(values(θ))
    _normal_logpdf(θ_vec, p.μ, p.σ)
end

function (p::NormalPriorVec)(θ::AbstractVector)
    _normal_logpdf(θ, p.μ, p.σ)
end

function _normal_logpdf(θ, μ, σ)
    n = length(θ)
    result = zero(eltype(θ))
    for i in 1:n
        result += -0.5 * ((θ[i] - μ[i]) / σ[i])^2 - log(σ[i])
    end
    result - n * 0.5 * log(2π)
end

"""
    InverseGammaPrior(α, β, param_names)

Inverse gamma priors for variance parameters at specified parameter names.
Assumes θ contains standard deviations (will square them).

# Example
```julia
prior = InverseGammaPrior(2.0, 1.0, (:σ_obs, :σ_level))
```
"""
struct InverseGammaPrior{A<:Real, B<:Real, N}
    α::A
    β::B
    param_names::N  # Tuple of Symbols
end

function (p::InverseGammaPrior)(θ::NamedTuple)
    result = 0.0
    for name in p.param_names
        σ = getproperty(θ, name)
        σ² = σ^2
        result += p.α * log(p.β) - lgamma(p.α) - (p.α + 1) * log(σ²) - p.β / σ²
    end
    result
end

"""
    CompositePrior(priors...)

Combine multiple priors by summing their log-densities.
"""
struct CompositePrior{T<:Tuple}
    priors::T
end

CompositePrior(priors...) = CompositePrior(priors)

(cp::CompositePrior)(θ) = sum(p(θ) for p in cp.priors)
