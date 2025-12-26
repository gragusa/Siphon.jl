"""
    codegen.jl

Code generation for state-space model specifications.
Converts SSMSpec into KFParms and initial state components.

Supports both:
- Vector{T} parameters (flat vector, indexed by position)
- NamedTuple parameters (accessed by name, from TransformVariables)
"""

export CovMatrixExpr

using LinearAlgebra
using TransformVariables
using TransformVariables: corr_cholesky_factor

# ============================================
# Transformation Builder
# ============================================

"""
    build_transformation(spec::SSMSpec)

Build a TransformVariables transformation from an SSMSpec.

Returns a transformation that maps ℝⁿ → NamedTuple with the parameter names
and appropriate constraints (positive for variances, bounded for others).

# Example
```julia
spec = local_level()
t = build_transformation(spec)
# t transforms ℝ² → (σ_obs = ..., σ_level = ...)

θ_nt = transform(t, randn(2))
# θ_nt is a NamedTuple with positive values
```
"""
function build_transformation(spec::SSMSpec)
    transforms = [_param_to_transform(p) for p in spec.params]
    names = Tuple(p.name for p in spec.params)
    as(NamedTuple{names}(Tuple(transforms)))
end

"""
    _param_to_transform(p::SSMParameter)

Convert a parameter's bounds to a TransformVariables scalar transformation.

Note: For semi-infinite intervals `[a, ∞)` where `a >= 0`, we use `asℝ₊` (exp transform).
Small positive lower bounds like 1e-6 are treated as zero since they are typically
numerical safeguards rather than meaningful constraints.
"""
function _param_to_transform(p::SSMParameter)
    if p.lower == -Inf && p.upper == Inf
        # Unbounded
        asℝ
    elseif p.lower >= 0.0 && p.upper == Inf
        # Positive (common for standard deviations, variances)
        # Use exp transform - works for any non-negative lower bound
        asℝ₊
    elseif p.lower == -Inf && p.upper < Inf
        # Upper bounded
        as(Real, -Inf, p.upper)
    elseif p.lower > -Inf && p.upper < Inf
        # Bounded interval [a, b] with a < b
        as(Real, p.lower, p.upper)
    else
        # Fallback to unbounded
        asℝ
    end
end

# ============================================
# Covariance Matrix Expression
# ============================================

"""
    CovMatrixExpr

Expression type for building positive definite covariance matrices.

Represents Σ = D * Corr * D where:
- D = Diagonal(σ) with standard deviation parameters
- Corr = L'L where L comes from `corr_cholesky_factor(n)`

# Fields
- `n::Int`: Matrix dimension
- `σ_param_names::Vector{Symbol}`: Names of standard deviation parameters
- `corr_param_names::Vector{Symbol}`: Names of correlation parameters (unconstrained)
- `var_init::Vector{Float64}`: Initial variance values (to avoid sqrt/square roundoff)
"""
struct CovMatrixExpr
    n::Int
    σ_param_names::Vector{Symbol}
    corr_param_names::Vector{Symbol}
    var_init::Vector{Float64}

    # Constructor with var_init
    function CovMatrixExpr(n::Int, σ_param_names::Vector{Symbol},
                           corr_param_names::Vector{Symbol}, var_init::Vector{Float64})
        new(n, σ_param_names, corr_param_names, var_init)
    end

    # Backwards-compatible constructor without var_init
    function CovMatrixExpr(n::Int, σ_param_names::Vector{Symbol},
                           corr_param_names::Vector{Symbol})
        new(n, σ_param_names, corr_param_names, Float64[])
    end
end

"""
    build_from_expr(expr::CovMatrixExpr, θ::NamedTuple)

Build a positive definite covariance matrix from CovMatrixExpr.

Uses Σ = D * Corr * D parameterization where the correlation matrix
is constructed via TransformVariables.corr_cholesky_factor.

Returns a regular Matrix (not Symmetric) to maintain AD compatibility,
but the result is numerically symmetric within floating-point precision.
"""
function build_from_expr(expr::CovMatrixExpr, θ::NamedTuple)
    n = expr.n
    T = _eltype(θ)

    # Build D from σ parameters
    σ = T[getproperty(θ, name) for name in expr.σ_param_names]
    D = Diagonal(σ)

    # Build Corr from correlation parameters
    if n > 1
        corr_vec = T[getproperty(θ, name) for name in expr.corr_param_names]
        t_corr = corr_cholesky_factor(n)
        L = TransformVariables.transform(t_corr, corr_vec)
        Corr = L' * L
    else
        # n=1: correlation is trivially 1
        Corr = ones(T, 1, 1)
    end

    # Compute covariance and symmetrize to avoid floating-point asymmetry
    Σ = D * Corr * D
    # Explicitly symmetrize: (Σ + Σ')/2
    return (Σ + Σ') / 2
end

"""
    build_from_expr(expr::CovMatrixExpr, theta::AbstractVector, param_map::Dict)

Build covariance matrix from vector parameters (legacy API).
"""
function build_from_expr(expr::CovMatrixExpr, theta::AbstractVector{T},
                         param_map::Dict{Symbol,Int}) where T
    n = expr.n

    # Build D from σ parameters
    σ = T[theta[param_map[name]] for name in expr.σ_param_names]
    D = Diagonal(σ)

    # Build Corr from correlation parameters
    if n > 1
        corr_vec = T[theta[param_map[name]] for name in expr.corr_param_names]
        t_corr = corr_cholesky_factor(n)
        L = TransformVariables.transform(t_corr, corr_vec)
        Corr = L' * L
    else
        Corr = ones(T, 1, 1)
    end

    # Compute covariance and symmetrize to avoid floating-point asymmetry
    Σ = D * Corr * D
    return (Σ + Σ') / 2
end

# ============================================
# Matrix Building - Core Functions
# ============================================

"""
    build_matrix(spec::SSMMatrixSpec, θ::NamedTuple)

Build a matrix from specification using NamedTuple parameters.
"""
function build_matrix(spec::SSMMatrixSpec, θ::NamedTuple)
    T = _eltype(θ)
    m, n = spec.dims
    M = zeros(T, m, n)

    # Fill with default
    default_val = evaluate_element(spec.default, θ)
    fill!(M, default_val)

    # Fill specified elements
    for ((i, j), elem) in spec.elements
        M[i, j] = evaluate_element(elem, θ)
    end

    return M
end

# Helper to get element type from NamedTuple
_eltype(θ::NamedTuple) = promote_type(typeof.(values(θ))...)
_eltype(θ::NamedTuple{(), Tuple{}}) = Float64  # Empty NamedTuple

"""
    evaluate_element(elem::MatrixElement, θ::NamedTuple)

Evaluate a matrix element given NamedTuple parameters.
"""
function evaluate_element(elem::FixedValue, θ::NamedTuple)
    elem.value
end

function evaluate_element(elem::ParameterRef, θ::NamedTuple)
    getproperty(θ, elem.name)
end

function evaluate_element(elem::Expr, θ::NamedTuple)
    evaluate_expr(elem, θ)
end

function evaluate_expr(ex::Expr, θ::NamedTuple)
    if ex.head == :call
        func = ex.args[1]
        args = [evaluate_expr(a, θ) for a in ex.args[2:end]]
        return eval(Expr(:call, func, args...))
    elseif ex.head == :ref
        error("Array indexing not supported in parameter expressions")
    else
        return eval(ex)
    end
end

function evaluate_expr(sym::Symbol, θ::NamedTuple)
    if haskey(θ, sym)
        return getproperty(θ, sym)
    else
        return eval(sym)
    end
end

function evaluate_expr(val::Number, θ::NamedTuple)
    val
end

# ============================================
# Building KFParms and Initial State
# ============================================

"""
    build_kfparms(spec::SSMSpec, θ::NamedTuple)

Build KFParms from specification and NamedTuple parameters.
"""
function build_kfparms(spec::SSMSpec, θ::NamedTuple)
    Z = build_matrix_or_expr(:Z, spec, θ)
    H = build_matrix_or_expr(:H, spec, θ)
    Tr = build_matrix_or_expr(:T, spec, θ)
    R = build_matrix_or_expr(:R, spec, θ)
    Q = build_matrix_or_expr(:Q, spec, θ)

    KFParms(Z, H, Tr, R, Q)
end

"""
    build_matrix_or_expr(name, spec, θ::NamedTuple)

Build a matrix from spec or MatrixExpr using NamedTuple parameters.
"""
function build_matrix_or_expr(name::Symbol, spec::SSMSpec, θ::NamedTuple)
    if haskey(spec.matrix_exprs, name)
        expr = spec.matrix_exprs[name]
        return build_from_expr(expr, θ)
    else
        mat_spec = getfield(spec, name)
        return build_matrix(mat_spec, θ)
    end
end

"""
    build_from_expr(expr::MatrixExpr, θ::NamedTuple)

Build a matrix from a MatrixExpr using NamedTuple parameters.
"""
function build_from_expr(expr, θ::NamedTuple)
    T = _eltype(θ)
    # Create a Dict for the builder function (it expects Dict)
    θ_dict = Dict{Symbol,T}()
    for p in expr.params
        if haskey(θ, p.name)
            θ_dict[p.name] = getproperty(θ, p.name)
        end
    end
    return expr.builder(θ_dict, expr.data)
end

"""
    build_initial_state(spec::SSMSpec, θ::NamedTuple)

Build initial state (a1, P1) from specification and NamedTuple parameters.
"""
function build_initial_state(spec::SSMSpec, θ::NamedTuple)
    T = _eltype(θ)
    a1 = T[evaluate_element(elem, θ) for elem in spec.a1]
    P1 = build_matrix(spec.P1, θ)
    (a1, P1)
end

# ============================================
# Vector-based API (for backwards compatibility)
# ============================================

"""
    build_matrix(spec::SSMMatrixSpec, theta::AbstractVector, param_map::Dict{Symbol,Int})

Build a matrix from specification and parameter vector (legacy API).
"""
function build_matrix(spec::SSMMatrixSpec, theta::AbstractVector{T},
                      param_map::Dict{Symbol,Int}) where T
    m, n = spec.dims
    M = zeros(T, m, n)

    default_val = evaluate_element(spec.default, theta, param_map)
    fill!(M, default_val)

    for ((i, j), elem) in spec.elements
        M[i, j] = evaluate_element(elem, theta, param_map)
    end

    return M
end

function evaluate_element(elem::FixedValue, theta::AbstractVector, param_map::Dict)
    elem.value
end

function evaluate_element(elem::ParameterRef, theta::AbstractVector, param_map::Dict)
    idx = param_map[elem.name]
    theta[idx]
end

function evaluate_element(elem::Expr, theta::AbstractVector, param_map::Dict)
    evaluate_expr(elem, theta, param_map)
end

function evaluate_expr(ex::Expr, theta::AbstractVector, param_map::Dict)
    if ex.head == :call
        func = ex.args[1]
        args = [evaluate_expr(a, theta, param_map) for a in ex.args[2:end]]
        return eval(Expr(:call, func, args...))
    elseif ex.head == :ref
        error("Array indexing not supported in parameter expressions")
    else
        return eval(ex)
    end
end

function evaluate_expr(sym::Symbol, theta::AbstractVector, param_map::Dict)
    if haskey(param_map, sym)
        return theta[param_map[sym]]
    else
        return eval(sym)
    end
end

function evaluate_expr(val::Number, theta::AbstractVector, param_map::Dict)
    val
end

"""
    build_kfparms(spec::SSMSpec, theta::AbstractVector)

Build KFParms from specification and parameter vector.
"""
function build_kfparms(spec::SSMSpec, theta::AbstractVector{T}) where T
    param_map = Dict{Symbol,Int}(p.name => i for (i, p) in enumerate(spec.params))

    Z = build_matrix_or_expr(:Z, spec, theta, param_map)
    H = build_matrix_or_expr(:H, spec, theta, param_map)
    Tr = build_matrix_or_expr(:T, spec, theta, param_map)
    R = build_matrix_or_expr(:R, spec, theta, param_map)
    Q = build_matrix_or_expr(:Q, spec, theta, param_map)

    KFParms(Z, H, Tr, R, Q)
end

function build_matrix_or_expr(name::Symbol, spec::SSMSpec, theta::AbstractVector{T},
                               param_map::Dict{Symbol,Int}) where T
    if haskey(spec.matrix_exprs, name)
        expr = spec.matrix_exprs[name]
        return build_from_expr(expr, theta, param_map)
    else
        mat_spec = getfield(spec, name)
        return build_matrix(mat_spec, theta, param_map)
    end
end

function build_from_expr(expr, theta::AbstractVector{T}, param_map::Dict{Symbol,Int}) where T
    θ_dict = Dict{Symbol,T}()
    for p in expr.params
        if haskey(param_map, p.name)
            θ_dict[p.name] = theta[param_map[p.name]]
        end
    end
    return expr.builder(θ_dict, expr.data)
end

"""
    build_initial_state(spec::SSMSpec, theta::AbstractVector)

Build initial state (a1, P1) from specification and parameter vector.
"""
function build_initial_state(spec::SSMSpec, theta::AbstractVector{T}) where T
    param_map = Dict{Symbol,Int}(p.name => i for (i, p) in enumerate(spec.params))

    a1 = T[evaluate_element(elem, theta, param_map) for elem in spec.a1]
    P1 = build_matrix(spec.P1, theta, param_map)

    (a1, P1)
end

# ============================================
# High-Level API
# ============================================

"""
    build_linear_state_space(spec::SSMSpec, θ, y; use_static=true)

Build state-space model components from specification.

Accepts either a Vector or NamedTuple of parameters.

# Arguments
- `spec::SSMSpec`: Model specification
- `θ`: Parameters (Vector or NamedTuple)
- `y`: Observations (used for dimension inference in some models)
- `use_static::Bool=true`: If true, automatically convert small matrices (dimensions ≤ 13)
  to StaticArrays for better performance

Returns a NamedTuple with:
- `p`: KFParms struct with system matrices (Z, H, T, R, Q)
- `a1`: Initial state mean vector
- `P1`: Initial state covariance matrix
"""
function build_linear_state_space(spec::SSMSpec, θ::Union{AbstractVector, NamedTuple}, y;
                                   use_static::Bool=true)
    kfparms = build_kfparms(spec, θ)
    a1, P1 = build_initial_state(spec, θ)

    if use_static
        # Convert to StaticArrays if dimensions are small
        p = KFParms_static(kfparms.Z, kfparms.H, kfparms.T, kfparms.R, kfparms.Q)
        a1_out = to_static_if_small(a1)
        P1_out = to_static_if_small(P1)
        return (p=p, a1=a1_out, P1=P1_out)
    else
        return (p=kfparms, a1=a1, P1=P1)
    end
end

"""
    objective_function(spec::SSMSpec, y)

Create a negative log-likelihood function for optimization.

Returns a callable that computes -loglik for a given parameter vector theta.
Uses the AD-compatible kalman_loglik function.
"""
function objective_function(spec::SSMSpec, y)
    function negloglik(theta)
        kfparms = build_kfparms(spec, theta)
        a1, P1 = build_initial_state(spec, theta)
        ll = kalman_loglik(kfparms, y, a1, P1)
        return -ll
    end
    return negloglik
end

"""
    ssm_loglik(spec::SSMSpec, θ::NamedTuple, y::AbstractMatrix)

Compute log-likelihood directly from spec and NamedTuple parameters.

This is the recommended high-level API for likelihood evaluation.
"""
function ssm_loglik(spec::SSMSpec, θ::NamedTuple, y::AbstractMatrix)
    kfparms = build_kfparms(spec, θ)
    a1, P1 = build_initial_state(spec, θ)
    kalman_loglik(kfparms, y, a1, P1)
end

# Alias for consistency
const kalman_loglik_spec = ssm_loglik
