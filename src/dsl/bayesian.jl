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

# ============================================
# Transformation Functions (using TransformVariables)
# ============================================

"""
    transform_to_constrained(spec, θ_unconstrained)

Transform from unconstrained ℝⁿ to constrained parameter space.
Returns (θ_constrained::NamedTuple, logjac::Real).

Uses TransformVariables.jl for the transformation.
"""
function transform_to_constrained(spec::AbstractSSMSpec, θ_u::AbstractVector)
    t = build_transformation(spec)
    θ_nt, logjac = TransformVariables.transform_and_logjac(t, θ_u)
    (θ_nt, logjac)
end

"""
    transform_to_unconstrained(spec, θ_constrained::NamedTuple)

Transform from constrained NamedTuple to unconstrained ℝⁿ.
Inverse of `transform_to_constrained`.
"""
function transform_to_unconstrained(spec::AbstractSSMSpec, θ_nt::NamedTuple)
    t = build_transformation(spec)
    TransformVariables.inverse(t, θ_nt)
end

"""
    transform_to_unconstrained(spec, θ_constrained::AbstractVector)

Transform from constrained vector to unconstrained ℝⁿ.
First converts vector to NamedTuple, then inverts.
"""
function transform_to_unconstrained(spec::AbstractSSMSpec, θ_c::AbstractVector)
    # Convert vector to NamedTuple
    names = Tuple(p.name for p in spec.params)
    θ_nt = NamedTuple{names}(Tuple(θ_c))
    transform_to_unconstrained(spec, θ_nt)
end

# ============================================
# Log-Density Type
# ============================================

"""
    SSMLogDensity(spec::SSMSpec, y; prior = FlatPrior(), static = :auto)
    SSMLogDensity(spec::EKFSpec, y; prior = FlatPrior())

Log-posterior density of a state-space model, evaluated at unconstrained
parameters θ_u ∈ ℝⁿ:

    logdensity(ld, θ_u) = log p(y | θ) + log p(θ) + log|J(θ_u)|

where θ is the constrained parameter NamedTuple obtained from θ_u by
[`build_transformation`](@ref)`(spec)` and `log|J|` is the log-Jacobian of
that map. The prior is evaluated on θ as estimated, so a prior on `var_obs`
is a prior on the variance. Implements the LogDensityProblems.jl interface.

`prior` is an [`AbstractPrior`](@ref) (usually [`ParameterPriors`](@ref)) or
any callable `θ::NamedTuple -> log p(θ)`. Parameter names in a
`ParameterPriors` must belong to `spec`; parameters without a prior get a flat
one. `static` (`:auto`, `:on` or `:off`) is passed to
[`build_linear_state_space`](@ref); the extended Kalman filter has no
StaticArrays specialization and takes no `static` keyword.

# Example

```julia
spec = local_level()
prior = ParameterPriors(var_obs = InverseGammaPrior(2.0, 1.0e4),
                        var_level = InverseGammaPrior(2.0, 1.0e3))
ld = SSMLogDensity(spec, y; prior)

θ_u = transform_to_unconstrained(spec, initial_values(spec))
logdensity(ld, θ_u)
```
"""
struct SSMLogDensity{S <: AbstractSSMSpec, T, Y <: AbstractMatrix, P}
    spec::S
    transformation::T
    y::Y
    prior::P
    static::Symbol
end

function SSMLogDensity(
        spec::SSMSpec,
        y::AbstractMatrix;
        prior = FlatPrior(),
        static::Symbol = :auto
)
    y = _nan_obs(y)
    _use_static(static, spec.n_states, spec.n_obs, spec.n_shocks)  # validate the keyword
    _check_prior(prior, spec)
    t = build_transformation(spec)
    SSMLogDensity(spec, t, y, prior, static)
end

"""
    logdensity(ld::SSMLogDensity, θ_unconstrained)

Log-likelihood plus log-prior plus the log-Jacobian of the map from
unconstrained to constrained parameters, at `θ_unconstrained`. Returns `-Inf`
when the filter fails or θ lies outside the prior's support.
"""
function logdensity(ld::SSMLogDensity, θ_u::AbstractVector)
    θ_nt, logjac = TransformVariables.transform_and_logjac(ld.transformation, θ_u)
    lp = ld.prior(θ_nt)
    # θ is outside the prior's support; the likelihood cannot change the result.
    lp == -Inf && return lp + logjac
    return _spec_loglik(ld.spec, θ_nt, ld.y, ld.static) + lp + logjac
end

# ============================================
# LogDensityProblems.jl Interface
# ============================================

function LogDensityProblems.capabilities(::Type{<:SSMLogDensity})
    LogDensityProblems.LogDensityOrder{0}()
end

LogDensityProblems.dimension(ld::SSMLogDensity) = n_params(ld.spec)

LogDensityProblems.logdensity(ld::SSMLogDensity, θ::AbstractVector) = logdensity(ld, θ)

# ============================================
# Priors
# ============================================

"""
    AbstractPrior

Supertype of Siphon's priors. A prior `p` is callable: `p(θ)` returns the log
prior density. Joint priors ([`FlatPrior`](@ref), [`ParameterPriors`](@ref),
[`CompositePrior`](@ref)) take the parameter NamedTuple θ; univariate priors
([`NormalPrior`](@ref), [`InverseGammaPrior`](@ref)) take one parameter value
and are combined by name with `ParameterPriors`.
"""
abstract type AbstractPrior end

"""
    FlatPrior()

Improper flat prior: `FlatPrior()(θ) == 0` for every θ. The default prior of
[`SSMLogDensity`](@ref).
"""
struct FlatPrior <: AbstractPrior end
(::FlatPrior)(θ) = 0.0

"""
    NormalPrior(μ, σ)

Univariate normal prior with mean `μ` and standard deviation `σ > 0`, for use
in [`ParameterPriors`](@ref). `NormalPrior(μ, σ)(x)` is the log density at `x`.
"""
struct NormalPrior{T <: Real} <: AbstractPrior
    μ::T
    σ::T
    function NormalPrior{T}(μ, σ) where {T <: Real}
        σ > 0 || throw(ArgumentError("NormalPrior: σ must be positive, got σ = $σ"))
        new{T}(μ, σ)
    end
end
NormalPrior(μ::Real, σ::Real) = NormalPrior{float(promote_type(typeof(μ), typeof(σ)))}(μ, σ)

(p::NormalPrior)(x::Real) = -((x - p.μ) / p.σ)^2 / 2 - log(p.σ) - log(2π) / 2

"""
    InverseGammaPrior(α, β)

Univariate inverse-gamma prior with shape `α > 0` and scale `β > 0`, for use in
[`ParameterPriors`](@ref), usually on a variance. `InverseGammaPrior(α, β)(x)`
is the log density at `x`: `α log β - log Γ(α) - (α + 1) log x - β / x` for
`x > 0`, and `-Inf` otherwise.
"""
struct InverseGammaPrior{T <: Real} <: AbstractPrior
    α::T
    β::T
    function InverseGammaPrior{T}(α, β) where {T <: Real}
        (α > 0 && β > 0) || throw(ArgumentError(
            "InverseGammaPrior: α and β must be positive, got α = $α, β = $β"))
        new{T}(α, β)
    end
end
function InverseGammaPrior(α::Real, β::Real)
    InverseGammaPrior{float(promote_type(typeof(α), typeof(β)))}(α, β)
end

function (p::InverseGammaPrior)(x::Real)
    x > 0 || return oftype(float(x) * p.β, -Inf)
    return p.α * log(p.β) - loggamma(p.α) - (p.α + 1) * log(x) - p.β / x
end

# A prior that `ParameterPriors` accepts for one parameter. The Distributions
# extension adds `UnivariateDistribution`.
_is_univariate_prior(p) = false
_is_univariate_prior(::Union{NormalPrior, InverseGammaPrior}) = true
_univariate_logpdf(p::AbstractPrior, x) = p(x)

"""
    ParameterPriors(; name = prior, ...)

Independent priors on the named parameters. Each `prior` is a univariate prior
([`NormalPrior`](@ref), [`InverseGammaPrior`](@ref)) or, with Distributions.jl
loaded, any `Distributions.UnivariateDistribution`. `ParameterPriors(...)(θ)`
is the sum of the log densities at the named entries of θ; parameters not named
contribute nothing (a flat prior). [`SSMLogDensity`](@ref) checks the names
against the spec.

# Example

```julia
prior = ParameterPriors(var_obs = InverseGammaPrior(2.0, 1.0),
                        ρ = NormalPrior(0.0, 0.5))

using Distributions
prior = ParameterPriors(var_obs = InverseGamma(2, 1), ρ = Beta(2, 2))
```
"""
struct ParameterPriors{N <: NamedTuple} <: AbstractPrior
    priors::N
    function ParameterPriors{N}(priors) where {N <: NamedTuple}
        isempty(priors) && throw(ArgumentError(
            "ParameterPriors needs at least one `name = prior`; use FlatPrior() for no prior"))
        for (name, p) in pairs(priors)
            _is_univariate_prior(p) || throw(ArgumentError(
                "ParameterPriors: the prior for `$name` is a $(typeof(p)); use " *
                "NormalPrior, InverseGammaPrior or (with Distributions loaded) " *
                "a univariate distribution"))
        end
        new{N}(priors)
    end
end
ParameterPriors(priors::NamedTuple) = ParameterPriors{typeof(priors)}(priors)
ParameterPriors(; priors...) = ParameterPriors(NamedTuple(priors))

function (p::ParameterPriors)(θ::NamedTuple)
    θsub = NamedTuple{keys(p.priors)}(θ)
    return sum(map(_univariate_logpdf, values(p.priors), values(θsub)))
end

"""
    CompositePrior(priors...)

Joint prior whose log density is the sum of those of `priors`, each an
[`AbstractPrior`](@ref) or a callable `θ -> log p(θ)`.
"""
struct CompositePrior{T <: Tuple} <: AbstractPrior
    priors::T
end

CompositePrior(priors...) = CompositePrior(priors)

(cp::CompositePrior)(θ) = sum(p(θ) for p in cp.priors)

# Fail at construction, not at the first evaluation, when a prior names a
# parameter the spec lacks.
_check_prior(prior, spec) = nothing
function _check_prior(::Nothing, spec)
    throw(ArgumentError("prior = nothing is not supported; use FlatPrior() for no prior"))
end
function _check_prior(p::ParameterPriors, spec)
    names = param_names(spec)
    unknown = [k for k in keys(p.priors) if k ∉ names]
    isempty(unknown) || throw(ArgumentError(
        "ParameterPriors names $(join(unknown, ", ")), which the spec does not " *
        "estimate; its parameters are $(join(names, ", "))"))
    return nothing
end
function _check_prior(p::CompositePrior, spec)
    foreach(q -> _check_prior(q, spec), p.priors)
    return nothing
end
