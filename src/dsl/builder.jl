"""
    builder.jl

Matrix-based state-space model specification for custom models.

Use `custom_ssm` when the pre-built templates don't fit your model.
"""

export custom_ssm, FreeParam, @P

"""
    FreeParam(name; init=0.0, lower=-Inf, upper=Inf)

Mark a matrix element as a free parameter to be estimated.

# Arguments
- `name::Symbol`: Parameter name
- `init`: Initial value for optimization
- `lower`, `upper`: Bounds for optimization (TransformVariables.jl handles transformations)

# Example
```julia
# A variance parameter (estimate variance directly, lower=0 triggers asℝ₊)
H = [FreeParam(:var_obs, init=100.0, lower=0.0)]

# A coefficient with bounds
T = [FreeParam(:ρ, init=0.8, lower=-0.99, upper=0.99)]
```
"""
struct FreeParam
    name::Symbol
    init::Float64
    lower::Float64
    upper::Float64
end

function FreeParam(name::Symbol; init = 0.0, lower = -Inf, upper = Inf)
    FreeParam(name, Float64(init), Float64(lower), Float64(upper))
end

# Short alias for convenience
const P = FreeParam

"""
    @P(name, init=0.0)

Shorthand macro for creating a FreeParam.

# Examples
```julia
@P(:σ_obs, 10.0)           # Same as FreeParam(:σ_obs, init=10.0)
@P(:ρ, 0.8)                # Same as FreeParam(:ρ, init=0.8)
```
"""
macro P(name, init = 0.0)
    :(FreeParam($(esc(name)), init = $(esc(init))))
end

# Type for matrix elements that can be Real or FreeParam
const MatrixInput = Union{Real,FreeParam}

"""
    custom_ssm(; Z, H, T, R, Q, a1, P1, name=:CustomSSM) -> SSMSpec

Create a state-space model specification from explicit matrices.

Use `FreeParam(:name, init=val, lower=lb, upper=ub)` to mark elements for estimation.
Use regular numbers for fixed values.

# Arguments
- `Z`: Observation matrix (p × m)
- `H`: Observation covariance (p × p)
- `T`: Transition matrix (m × m)
- `R`: Selection matrix (m × r)
- `Q`: State covariance (r × r)
- `a1`: Initial state mean (m-vector)
- `P1`: Initial state covariance (m × m)
- `name`: Model name (default: :CustomSSM)

# Example
```julia
# Local level model
spec = custom_ssm(
    Z = [1.0],
    H = [FreeParam(:var_obs, init=225.0, lower=0.0)],
    T = [1.0],
    R = [1.0],
    Q = [FreeParam(:var_level, init=100.0, lower=0.0)],
    a1 = [0.0],
    P1 = [1e7]
)
param_names(spec)  # [:var_obs, :var_level]
```

See also: `local_level`, `ar1`, `arma`, `dynamic_factor` for pre-built templates.
"""
function custom_ssm(; Z, H, T, R, Q, a1, P1, name::Symbol = :CustomSSM)
    # Collect parameters and MatrixExpr objects
    params = SSMParameter{Float64}[]
    param_set = Set{Symbol}()
    matrix_exprs = Dict{Symbol,Any}()

    # Process each matrix, handling MatrixExpr specially
    Z_spec, Z_dims = _process_matrix_input(:Z, Z, params, param_set, matrix_exprs)
    H_spec, H_dims = _process_matrix_input(:H, H, params, param_set, matrix_exprs)
    T_spec, T_dims = _process_matrix_input(:T, T, params, param_set, matrix_exprs)
    R_spec, R_dims = _process_matrix_input(:R, R, params, param_set, matrix_exprs)
    Q_spec, Q_dims = _process_matrix_input(:Q, Q, params, param_set, matrix_exprs)
    P1_spec, P1_dims = _process_matrix_input(:P1, P1, params, param_set, matrix_exprs)

    # Process a1 vector
    a1_vec = _to_vector(a1)
    a1_elems = _build_vector_spec(a1_vec, params, param_set)

    # Infer dimensions
    p, m = Z_dims
    r = Q_dims[1]
    n_states = m
    n_obs = p
    n_shocks = r

    # Validate dimensions
    H_dims == (p, p) || throw(DimensionMismatch("H must be ($p, $p), got $H_dims"))
    T_dims == (m, m) || throw(DimensionMismatch("T must be ($m, $m), got $T_dims"))
    R_dims == (m, r) || throw(DimensionMismatch("R must be ($m, $r), got $R_dims"))
    Q_dims == (r, r) || throw(DimensionMismatch("Q must be ($r, $r), got $Q_dims"))
    length(a1_vec) == m ||
        throw(DimensionMismatch("a1 must have length $m, got $(length(a1_vec))"))
    P1_dims == (m, m) || throw(DimensionMismatch("P1 must be ($m, $m), got $P1_dims"))

    SSMSpec(
        name,
        n_states,
        n_obs,
        n_shocks,
        params,
        Z_spec,
        H_spec,
        T_spec,
        R_spec,
        Q_spec,
        a1_elems,
        P1_spec,
        matrix_exprs,
    )
end

# Check if something is a MatrixExpr (duck typing to avoid circular dependency)
_is_matrix_expr(x) =
    hasfield(typeof(x), :builder) &&
    hasfield(typeof(x), :dims) &&
    hasfield(typeof(x), :params)

# Check if something is a CovFree specification
_is_cov_free(x) = x isa CovFree

"""
Process a matrix input, handling regular matrices, MatrixExpr, and CovFree.
Returns (SSMMatrixSpec, dims).
"""
function _process_matrix_input(name::Symbol, input, params, param_set, matrix_exprs)
    if _is_cov_free(input)
        # Handle CovFree - create parameters and CovMatrixExpr
        n = input.n
        prefix = input.prefix

        # Create σ parameters (positive, use lower=0.0 for asℝ₊ transform)
        σ_param_names = Symbol[]
        for i = 1:n
            pname = Symbol("$(prefix)_σ_$i")
            if !(pname in param_set)
                push!(
                    params,
                    SSMParameter(pname; lower = 0.0, upper = Inf, init = input.init_σ),
                )
                push!(param_set, pname)
            end
            push!(σ_param_names, pname)
        end

        # Create correlation parameters (unconstrained)
        n_corr = n * (n - 1) ÷ 2
        corr_param_names = Symbol[]
        for i = 1:n_corr
            pname = Symbol("$(prefix)_corr_$i")
            if !(pname in param_set)
                push!(params, SSMParameter(pname; lower = -Inf, upper = Inf, init = 0.0))
                push!(param_set, pname)
            end
            push!(corr_param_names, pname)
        end

        # Create CovMatrixExpr and store it
        expr = CovMatrixExpr(n, σ_param_names, corr_param_names)
        matrix_exprs[name] = expr

        # Return placeholder spec with correct dimensions
        return (SSMMatrixSpec((n, n)), (n, n))

    elseif _is_matrix_expr(input)
        # It's a MatrixExpr - extract params and store it
        for p in input.params
            if !(p.name in param_set)
                push!(params, p)
                push!(param_set, p.name)
            end
        end
        matrix_exprs[name] = input
        # Return a placeholder spec with correct dimensions
        return (SSMMatrixSpec(input.dims), input.dims)
    else
        # Regular matrix - convert and build spec
        mat = _to_matrix(input)
        spec = _build_matrix_spec(mat, params, param_set)
        return (spec, size(mat))
    end
end

# Helper to convert input to matrix
function _to_matrix(x::AbstractMatrix{<:MatrixInput})
    convert(Matrix{MatrixInput}, x)
end

function _to_matrix(x::AbstractMatrix{<:Real})
    Matrix{MatrixInput}(convert.(MatrixInput, x))
end

# Handle mixed matrices (e.g., Matrix{Any} from [FreeParam(...) 0.0; ...])
function _to_matrix(x::AbstractMatrix)
    mat = Matrix{MatrixInput}(undef, size(x))
    for i in eachindex(x)
        elem = x[i]
        if elem isa FreeParam
            mat[i] = elem
        elseif elem isa Real
            mat[i] = Float64(elem)
        else
            throw(
                ArgumentError(
                    "Matrix element must be Real or FreeParam, got $(typeof(elem))",
                ),
            )
        end
    end
    mat
end

function _to_matrix(x::AbstractVector{<:MatrixInput})
    # Treat vector as 1×n row vector (common for Z matrix)
    n = length(x)
    mat = Matrix{MatrixInput}(undef, 1, n)
    for i = 1:n
        mat[1, i] = x[i]
    end
    mat
end

function _to_matrix(x::AbstractVector{<:Real})
    n = length(x)
    mat = Matrix{MatrixInput}(undef, 1, n)
    for i = 1:n
        mat[1, i] = Float64(x[i])
    end
    mat
end

function _to_matrix(x::Real)
    Matrix{MatrixInput}(fill(Float64(x), 1, 1))
end

function _to_matrix(x::FreeParam)
    Matrix{MatrixInput}(fill(x, 1, 1))
end

# Helper to convert input to vector
function _to_vector(x::AbstractVector{<:MatrixInput})
    convert(Vector{MatrixInput}, x)
end

function _to_vector(x::AbstractVector{<:Real})
    Vector{MatrixInput}(convert.(Float64, x))
end

function _to_vector(x::Real)
    Vector{MatrixInput}([Float64(x)])
end

# Build SSMMatrixSpec from a matrix that may contain FreeParams
function _build_matrix_spec(
    mat::Matrix{MatrixInput},
    params::Vector{SSMParameter{Float64}},
    param_set::Set{Symbol},
)
    m, n = size(mat)
    spec = SSMMatrixSpec((m, n))

    for j = 1:n, i = 1:m
        elem = mat[i, j]
        if elem isa FreeParam
            # Add parameter if not already added
            if !(elem.name in param_set)
                push!(
                    params,
                    SSMParameter(
                        elem.name;
                        init = elem.init,
                        lower = elem.lower,
                        upper = elem.upper,
                    ),
                )
                push!(param_set, elem.name)
            end
            # Create parameter reference
            spec.elements[(i, j)] = ParameterRef(elem.name)
        else
            # Fixed value
            spec.elements[(i, j)] = FixedValue(Float64(elem))
        end
    end

    spec
end

# Build vector spec
function _build_vector_spec(
    vec::Vector{MatrixInput},
    params::Vector{SSMParameter{Float64}},
    param_set::Set{Symbol},
)
    elems = MatrixElement[]

    for elem in vec
        if elem isa FreeParam
            if !(elem.name in param_set)
                push!(
                    params,
                    SSMParameter(
                        elem.name;
                        init = elem.init,
                        lower = elem.lower,
                        upper = elem.upper,
                    ),
                )
                push!(param_set, elem.name)
            end
            push!(elems, ParameterRef(elem.name))
        else
            push!(elems, FixedValue(Float64(elem)))
        end
    end

    elems
end
