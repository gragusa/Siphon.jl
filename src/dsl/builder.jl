"""
    builder.jl

Matrix-based state-space model specification for custom models.

Use `custom_ssm` when the pre-built templates don't fit your model.
"""

"""
    FreeParam(name; lower, upper, init)
    FreeParam(name, lower, upper, init)

Mark a matrix element as a free parameter to be estimated; a plain number marks
a fixed element. The arguments are in the same order as [`SSMParameter`](@ref).

Any field may be left unset (`nothing`). Unset bounds mean unbounded. An unset
`init` is a point inside the bounds: the midpoint of two finite bounds, one unit
inside a single finite bound, or `0.0`. Maximum likelihood rescales an unset
`init` on the diagonal of `H` or `Q` to the data (see [`optimize_ssm`](@ref)).
`custom_ssm` and `custom_ekf` require a name.

The same name may appear in several cells, which then share one parameter. Every
occurrence must give the same bounds and `init`.

# Example
```julia
# A variance (lower = 0 gives a positive transformation); init defaults to 1.0
H = [FreeParam(:var_obs; lower = 0.0)]

# A bounded coefficient
T = [FreeParam(:ρ; lower = -0.99, upper = 0.99, init = 0.8)]
```
"""
struct FreeParam
    name::Union{Symbol, Nothing}
    lower::Union{Float64, Nothing}
    upper::Union{Float64, Nothing}
    init::Union{Float64, Nothing}

    function FreeParam(name, lower, upper, init)
        lo = _float_or_nothing(lower)
        up = _float_or_nothing(upper)
        x = _float_or_nothing(init)
        lo !== nothing && up !== nothing && !(lo < up) &&
            throw(ArgumentError("FreeParam $(repr(name)): lower bound $lo must be below upper bound $up"))
        if x !== nothing
            _check_init(name, something(lo, -Inf), something(up, Inf), x)
        end
        new(name, lo, up, x)
    end
end

function FreeParam(name::Union{Symbol, Nothing} = nothing;
        lower = nothing, upper = nothing, init = nothing)
    FreeParam(name, lower, upper, init)
end

_float_or_nothing(::Nothing) = nothing
_float_or_nothing(x::Real) = Float64(x)

# The parameter a FreeParam declares in `custom_ssm`/`custom_ekf`.
function SSMParameter(fp::FreeParam)
    fp.name === nothing &&
        throw(ArgumentError("a FreeParam in a model matrix needs a name, e.g. FreeParam(:ρ)"))
    SSMParameter(fp.name; lower = something(fp.lower, -Inf),
        upper = something(fp.upper, Inf), init = fp.init)
end

# Add `p` to `params` unless a parameter of that name is already there, in which
# case the two must agree.
function _add_param!(params::Vector{SSMParameter{Float64}}, param_set::Set{Symbol},
        p::SSMParameter)
    if p.name in param_set
        q = params[findfirst(x -> x.name == p.name, params)]
        (q.lower, q.upper, q.init) == (p.lower, p.upper, p.init) || throw(ArgumentError(
            "parameter :$(p.name) is declared twice with different settings: " *
            "(lower, upper, init) = $((q.lower, q.upper, q.init)) and " *
            "$((p.lower, p.upper, p.init))"))
        return params
    end
    push!(param_set, p.name)
    push!(params, p)
end

# Type for matrix elements that can be Real or FreeParam
const MatrixInput = Union{Real, FreeParam}

"""
    custom_ssm(; Z, H, T, R, Q, a1, P1, name=:CustomSSM) -> SSMSpec

Create a state-space model specification from explicit matrices.

Use `FreeParam(:name; lower, upper, init)` to mark elements for estimation and
plain numbers for fixed values. Each matrix may be a matrix or vector of such
elements, a `Diagonal`, a scaled identity `I` (sized from the other inputs), a
[`cov_free`](@ref) marker, or a `MatrixExpr`.

# Arguments
- `Z`: Observation matrix (p × m)
- `H`: Observation covariance (p × p)
- `T`: Transition matrix (m × m)
- `R`: Selection matrix (m × r)
- `Q`: State covariance (r × r)
- `a1`: Initial state mean (m-vector); it sets the state dimension m
- `P1`: Initial state covariance (m × m)
- `name`: Model name (default: :CustomSSM)

`I` is sized as m × m for `Z`, `T`, `R` and `P1`, as p × p for `H`, and as
r × r for `Q`, where p and r are the row count of `Z` and the column count of `R`.

# Example
```julia
# Local level model
spec = custom_ssm(
    Z = [1.0],
    H = [FreeParam(:var_obs; lower = 0.0, init = 225.0)],
    T = [1.0],
    R = [1.0],
    Q = [FreeParam(:var_level; lower = 0.0, init = 100.0)],
    a1 = [0.0],
    P1 = [1e7]
)
param_names(spec)  # [:var_obs, :var_level]

# Local linear trend with identity selection and diffuse P1
spec = custom_ssm(
    Z = [1.0 0.0],
    H = [FreeParam(:var_obs; lower = 0.0)],
    T = [1.0 1.0; 0.0 1.0],
    R = I,
    Q = diag_free([:var_level, :var_slope]),
    a1 = zeros(2),
    P1 = 1e7I
)
```

See also: `local_level`, `ar1`, `arma`, `dynamic_factor` for pre-built templates.
"""
function custom_ssm(; Z, H, T, R, Q, a1, P1, name::Symbol = :CustomSSM)
    # Collect parameters and MatrixExpr objects
    params = SSMParameter{Float64}[]
    param_set = Set{Symbol}()
    matrix_exprs = Dict{Symbol, AbstractMatrixExpr}()

    # Size any `I` inputs: a1 gives m, Z gives p, R gives r.
    a1_vec = _to_vector(a1)
    m = length(a1_vec)
    Z, T, R, P1 = map(x -> _sized(x, m), (Z, T, R, P1))
    H = _sized(H, _input_dims(Z)[1])
    Q = _sized(Q, _input_dims(R)[2])

    # Process each matrix, handling MatrixExpr specially
    Z_spec, Z_dims = _process_matrix_input(:Z, Z, params, param_set, matrix_exprs)
    H_spec, H_dims = _process_matrix_input(:H, H, params, param_set, matrix_exprs)
    T_spec, T_dims = _process_matrix_input(:T, T, params, param_set, matrix_exprs)
    R_spec, R_dims = _process_matrix_input(:R, R, params, param_set, matrix_exprs)
    Q_spec, Q_dims = _process_matrix_input(:Q, Q, params, param_set, matrix_exprs)
    P1_spec, P1_dims = _process_matrix_input(:P1, P1, params, param_set, matrix_exprs)

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
        matrix_exprs
    )
end

# A scaled identity becomes an n × n matrix; other inputs pass through.
_sized(x, n::Int) = x
_sized(J::UniformScaling, n::Int) = Matrix(J, n, n)

_input_dims(x::CovFree) = (x.n, x.n)
_input_dims(x::MatrixExpr) = x.dims
_input_dims(x) = size(_to_matrix(x))

"""
Process a matrix input, handling regular matrices, MatrixExpr, and CovFree.
Returns (SSMMatrixSpec, dims).
"""
function _process_matrix_input(name::Symbol, input, params, param_set, matrix_exprs)
    if input isa CovFree
        n = input.n
        prefix = input.prefix
        sd_names = [Symbol("$(prefix)_sd_$i") for i in 1:n]
        corr_names = [Symbol("$(prefix)_corr_$i") for i in 1:(n * (n - 1) ÷ 2)]
        for pname in sd_names
            _add_param!(params, param_set,
                SSMParameter(pname; lower = 0.0, upper = Inf, init = input.init_sd))
        end
        for pname in corr_names
            _add_param!(params, param_set, SSMParameter(pname; init = 0.0))
        end
        matrix_exprs[name] = CovMatrixExpr(n, sd_names, corr_names)

        # Return placeholder spec with correct dimensions
        return (SSMMatrixSpec((n, n)), (n, n))

    elseif input isa MatrixExpr
        # It's a MatrixExpr - extract params and store it
        for p in input.params
            _add_param!(params, param_set, p)
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
    for i in 1:n
        mat[1, i] = x[i]
    end
    mat
end

function _to_matrix(x::AbstractVector{<:Real})
    n = length(x)
    mat = Matrix{MatrixInput}(undef, 1, n)
    for i in 1:n
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

function _to_matrix(::UniformScaling)
    throw(ArgumentError("cannot size `I` here; pass an explicit matrix such as Matrix(1.0I, n, n)"))
end

# Helper to convert input to vector
function _to_vector(x::AbstractVector{<:MatrixInput})
    convert(Vector{MatrixInput}, x)
end

function _to_vector(x::AbstractVector{<:Real})
    Vector{MatrixInput}(convert.(Float64, x))
end

# Mixed vector (e.g. Vector{Any} from [FreeParam(...), 0.0, FreeParam(...)])
function _to_vector(x::AbstractVector)
    out = Vector{MatrixInput}(undef, length(x))
    for i in eachindex(x)
        elem = x[i]
        if elem isa FreeParam
            out[i] = elem
        elseif elem isa Real
            out[i] = Float64(elem)
        else
            throw(ArgumentError(
                "Vector element must be Real or FreeParam, got $(typeof(elem))"))
        end
    end
    out
end

function _to_vector(x::Real)
    Vector{MatrixInput}([Float64(x)])
end

# Build SSMMatrixSpec from a matrix that may contain FreeParams
function _build_matrix_spec(
        mat::Matrix{MatrixInput},
        params::Vector{SSMParameter{Float64}},
        param_set::Set{Symbol}
)
    m, n = size(mat)
    spec = SSMMatrixSpec((m, n))

    for j in 1:n, i in 1:m

        elem = mat[i, j]
        if elem isa FreeParam
            _add_param!(params, param_set, SSMParameter(elem))
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
        param_set::Set{Symbol}
)
    elems = MatrixElement[]

    for elem in vec
        if elem isa FreeParam
            _add_param!(params, param_set, SSMParameter(elem))
            push!(elems, ParameterRef(elem.name))
        else
            push!(elems, FixedValue(Float64(elem)))
        end
    end

    elems
end
