"""
    types.jl

Type definitions for the state-space model DSL.
These types form the intermediate representation (IR) for model specification.
"""

"""
    SSMParameter{T<:Real}

Represents a single estimable parameter in a state-space model.

# Fields
- `name::Symbol`: Parameter name for identification
- `lower::T`: Lower bound for optimization (use -Inf for unbounded)
- `upper::T`: Upper bound for optimization (use Inf for unbounded)
- `init::T`: Initial value for optimization
- `default_init::Bool`: `true` when `init` is a template default rather than a
  value the user gave. Maximum likelihood then rescales a default variance to the
  data before optimising (see [`optimize_ssm`](@ref)).
"""
struct SSMParameter{T <: Real}
    name::Symbol
    lower::T
    upper::T
    init::T
    default_init::Bool

    function SSMParameter(name::Symbol, lower::T, upper::T, init::T,
            default_init::Bool = false) where {T <: Real}
        lower <= init <= upper || throw(
            ArgumentError("Initial value $init for $name must be in [$lower, $upper]"),
        )
        new{T}(name, lower, upper, init, default_init)
    end
end

# Convenience constructors
function SSMParameter(name::Symbol; lower = -Inf, upper = Inf, init = 0.0,
        default_init::Bool = false)
    SSMParameter(name, Float64(lower), Float64(upper), Float64(init), default_init)
end

SSMParameter(name::Symbol, init::Real) = SSMParameter(name, -Inf, Inf, Float64(init))

"""
    FixedValue{T}

Represents a fixed (non-estimable) value in a state-space model.
"""
struct FixedValue{T <: Real}
    value::T
end

"""
    ParameterRef

Reference to a parameter by name, used for building matrix mappings.
"""
struct ParameterRef
    name::Symbol
end

"""
    MatrixElement

A single element of a state-space matrix: a `FixedValue` or a `ParameterRef`.
"""
const MatrixElement = Union{FixedValue, ParameterRef}

"""
    SSMMatrixSpec

Specification for a state-space matrix with fixed/free elements.

# Fields
- `dims::Tuple{Int,Int}`: Matrix dimensions
- `elements::Dict{Tuple{Int,Int}, MatrixElement}`: Mapping from (row, col) to element spec
- `default::MatrixElement`: Default value for unspecified elements
"""
struct SSMMatrixSpec
    dims::Tuple{Int, Int}
    elements::Dict{Tuple{Int, Int}, MatrixElement}
    default::MatrixElement
end

# Default constructor with zero default
function SSMMatrixSpec(dims::Tuple{Int, Int})
    SSMMatrixSpec(dims, Dict{Tuple{Int, Int}, MatrixElement}(), FixedValue(0.0))
end

"""
    AbstractMatrixExpr

Supertype of system matrices computed from parameters as a whole rather than
cell by cell: `MatrixExpr` and `CovMatrixExpr`. A subtype `E` implements
`build_from_expr(::E, θ::NamedTuple)` and
`build_from_expr(::E, θ::AbstractVector, param_map::Dict{Symbol, Int})`, both
returning the matrix.
"""
abstract type AbstractMatrixExpr end

"""
    SSMSpec

Complete specification of a state-space model.

# Fields
- `name::Symbol`: Model name
- `n_states::Int`: Number of state variables
- `n_obs::Int`: Number of observables
- `n_shocks::Int`: Number of shocks
- `params::Vector{SSMParameter}`: Free parameters to estimate
- `Z::SSMMatrixSpec`: Observation matrix specification
- `H::SSMMatrixSpec`: Observation covariance specification
- `T::SSMMatrixSpec`: Transition matrix specification
- `R::SSMMatrixSpec`: Selection matrix specification
- `Q::SSMMatrixSpec`: State covariance specification
- `a1::Vector{MatrixElement}`: Initial state mean
- `P1::SSMMatrixSpec`: Initial state covariance (finite part for exact diffuse)
- `P1_inf::Union{Nothing,SSMMatrixSpec}`: Diffuse covariance (nothing = approximate diffuse)
- `matrix_exprs::Dict{Symbol,AbstractMatrixExpr}`: Matrices computed from parameters as a whole (DNS loadings, full covariances)

When `P1_inf` is not `nothing`, exact diffuse initialization is used:
- `P1` becomes P1_star (finite part)
- `P1_inf` is the diffuse/infinite part

When `P1_inf` is `nothing`, approximate diffuse is used (P1 is the full initial covariance).
"""
struct SSMSpec
    name::Symbol
    n_states::Int
    n_obs::Int
    n_shocks::Int
    params::Vector{SSMParameter{Float64}}
    Z::SSMMatrixSpec
    H::SSMMatrixSpec
    T::SSMMatrixSpec
    R::SSMMatrixSpec
    Q::SSMMatrixSpec
    a1::Vector{MatrixElement}
    P1::SSMMatrixSpec
    P1_inf::Union{Nothing, SSMMatrixSpec}  # nothing = approximate, SSMMatrixSpec = exact diffuse
    matrix_exprs::Dict{Symbol, AbstractMatrixExpr}  # :Z, :H, … → expression; overrides the SSMMatrixSpec field
end

# Convenience constructor without matrix_exprs or P1_inf (approximate diffuse)
function SSMSpec(name, n_states, n_obs, n_shocks, params, Z, H, T, R, Q, a1, P1)
    SSMSpec(
        name,
        n_states,
        n_obs,
        n_shocks,
        params,
        Z,
        H,
        T,
        R,
        Q,
        a1,
        P1,
        nothing,
        Dict{Symbol, AbstractMatrixExpr}()
    )
end

# Constructor with P1_inf but no matrix_exprs
function SSMSpec(
        name,
        n_states,
        n_obs,
        n_shocks,
        params,
        Z,
        H,
        T,
        R,
        Q,
        a1,
        P1,
        P1_inf::Union{Nothing, SSMMatrixSpec}
)
    SSMSpec(
        name,
        n_states,
        n_obs,
        n_shocks,
        params,
        Z,
        H,
        T,
        R,
        Q,
        a1,
        P1,
        P1_inf,
        Dict{Symbol, AbstractMatrixExpr}()
    )
end

# Constructor with matrix_exprs but no P1_inf (for custom_ssm backwards compatibility)
function SSMSpec(
        name,
        n_states,
        n_obs,
        n_shocks,
        params,
        Z,
        H,
        T,
        R,
        Q,
        a1,
        P1,
        matrix_exprs::AbstractDict{Symbol}
)
    SSMSpec(
        name,
        n_states,
        n_obs,
        n_shocks,
        params,
        Z,
        H,
        T,
        R,
        Q,
        a1,
        P1,
        nothing,
        matrix_exprs
    )
end

"""
    param_names(spec::SSMSpec)

Return the names of all free parameters.
"""
param_names(spec::SSMSpec) = [p.name for p in spec.params]

"""
    n_params(spec::SSMSpec)

Return the number of free parameters.
"""
n_params(spec::SSMSpec) = length(spec.params)

"""
    initial_values(spec::SSMSpec)

Return a vector of initial parameter values.
"""
initial_values(spec::SSMSpec) = [p.init for p in spec.params]

"""
    param_bounds(spec::SSMSpec)

Return lower and upper bound vectors.
"""
function param_bounds(spec::SSMSpec)
    lower = [p.lower for p in spec.params]
    upper = [p.upper for p in spec.params]
    (lower, upper)
end

"""
    uses_exact_diffuse(spec::SSMSpec) -> Bool

Check if the model specification uses exact diffuse initialization.
Returns `true` if `P1_inf` is specified, `false` otherwise.
"""
uses_exact_diffuse(spec::SSMSpec) = spec.P1_inf !== nothing

"""
    param_index(spec::SSMSpec, name::Symbol)

Return the index of a parameter by name.
"""
function param_index(spec::SSMSpec, name::Symbol)
    idx = findfirst(p -> p.name == name, spec.params)
    idx === nothing && throw(ArgumentError("Unknown parameter: $name"))
    idx
end

# ============================================
# Covariance Matrix Parameterization
# ============================================

"""
    CovFree

Marker type for a full positive definite covariance matrix with free parameters.

Used in `custom_ssm` to specify that a covariance matrix (Q or H) should be
parameterized as Σ = D * Corr * D where:
- D = Diagonal(σ) with n positive standard deviation parameters
- Corr = L'L where L comes from `corr_cholesky_factor(n)` with n(n-1)/2 correlation parameters

Total parameters: n + n(n-1)/2 = n(n+1)/2

# Fields
- `n::Int`: Matrix dimension
- `prefix::Symbol`: Prefix for parameter names (e.g., :Q creates :Q_σ_1, :Q_corr_1, etc.)
- `init_σ::Float64`: Initial value for standard deviation parameters (default: 1.0)

# Example
```julia
Q = cov_free(3, :Q)  # Creates 3×3 covariance with 6 parameters
```
"""
struct CovFree
    n::Int
    prefix::Symbol
    init_σ::Float64
end
