"""
    matrix_helpers.jl

Syntactic sugar for common state-space matrix structures.
"""

export diag_free, scalar_free, identity_mat, zeros_mat, ones_mat
export lower_triangular_free, symmetric_free
export cov_free
export block_diag

"""
    diag_free(names; init=1.0, lower=0.0, upper=Inf)
    diag_free(n, prefix; init=1.0, lower=0.0, upper=Inf)

Create a diagonal matrix with free parameters on the diagonal.

# Arguments
- `names`: Vector of parameter names, e.g., `[:var_level, :var_slope]`
- OR `n, prefix`: Number of elements and prefix, creates `:prefix_1`, `:prefix_2`, etc.
- `init`: Initial value(s) - scalar or vector (variance values)
- `lower`, `upper`: Bounds - scalar or vector (lower=0.0 triggers asℝ₊ transform)

# Examples
```julia
# 2×2 diagonal with named parameters (variance values)
Q = diag_free([:var_level, :var_slope], init=[0.01, 0.0001])

# 3×3 diagonal with auto-generated names
Q = diag_free(3, :var, init=1.0)  # Creates :var_1, :var_2, :var_3

# 1×1 scalar variance
H = diag_free([:var_obs], init=100.0)
# Or equivalently:
H = scalar_free(:var_obs, init=100.0)
```
"""
function diag_free(names::AbstractVector{Symbol}; init = 1.0, lower = 0.0, upper = Inf)
    n = length(names)

    # Broadcast scalar arguments to vectors
    inits = _to_vec(init, n)
    lowers = _to_vec(lower, n)
    uppers = _to_vec(upper, n)

    # Build matrix
    mat = Matrix{Any}(undef, n, n)
    fill!(mat, 0.0)
    for i in 1:n
        mat[i, i] = FreeParam(names[i], init = inits[i], lower = lowers[i], upper = uppers[i])
    end
    mat
end

function diag_free(n::Int, prefix::Symbol; init = 1.0, lower = 0.0, upper = Inf)
    names = [Symbol("$(prefix)_$i") for i in 1:n]
    diag_free(names; init = init, lower = lower, upper = upper)
end

# Single parameter case
function diag_free(name::Symbol; init = 1.0, lower = 0.0, upper = Inf)
    diag_free([name]; init = init, lower = lower, upper = upper)
end

"""
    scalar_free(name; init=1.0, lower=0.0, upper=Inf)

Create a 1×1 matrix with a single free parameter. Convenience wrapper for `diag_free`.

# Example
```julia
H = scalar_free(:var_obs, init=225.0)
# Equivalent to: H = [FreeParam(:var_obs, init=225.0, lower=0.0)]
```
"""
function scalar_free(name::Symbol; init = 1.0, lower = 0.0, upper = Inf)
    [FreeParam(name, init = init, lower = lower, upper = upper)]
end

"""
    identity_mat(n)

Create an n×n identity matrix (fixed values).

# Example
```julia
R = identity_mat(3)  # 3×3 identity
```
"""
identity_mat(n::Int) = Matrix{Float64}(LinearAlgebra.I, n, n)

"""
    zeros_mat(m, n=m)

Create an m×n zero matrix.
"""
zeros_mat(m::Int, n::Int = m) = zeros(Float64, m, n)

"""
    ones_mat(m, n=m)

Create an m×n matrix of ones.
"""
ones_mat(m::Int, n::Int = m) = ones(Float64, m, n)

"""
    diag_fixed(values)

Create a diagonal matrix with fixed values on the diagonal.

# Example
```julia
Q = diag_fixed([0.1, 0.01, 0.001])  # Fixed 3×3 diagonal
```
"""
function diag_fixed(values::AbstractVector{<:Real})
    n = length(values)
    mat = zeros(Float64, n, n)
    for i in 1:n
        mat[i, i] = values[i]
    end
    mat
end

diag_fixed(value::Real, n::Int) = diag_fixed(fill(value, n))

"""
    lower_triangular_free(n, prefix; init=0.0, lower=-Inf, upper=Inf)

Create a lower triangular matrix with free parameters below and on the diagonal.
Useful for Cholesky factors.

# Example
```julia
# Cholesky factor of covariance matrix
L = lower_triangular_free(2, :L)
# Creates:
# [FreeParam(:L_1_1)  0.0            ]
# [FreeParam(:L_2_1)  FreeParam(:L_2_2)]
```
"""
function lower_triangular_free(
        n::Int,
        prefix::Symbol;
        init = 0.0,
        lower = -Inf,
        upper = Inf
)
    mat = Matrix{Any}(undef, n, n)
    fill!(mat, 0.0)

    for j in 1:n
        for i in j:n
            name = Symbol("$(prefix)_$(i)_$(j)")
            mat[i, j] = FreeParam(name, init = init, lower = lower, upper = upper)
        end
    end
    mat
end

"""
    symmetric_free(n, prefix; init_diag=1.0, init_offdiag=0.0,
                   lower_diag=0.0, lower_offdiag=-Inf,
                   upper_diag=Inf, upper_offdiag=Inf)

Create a symmetric matrix with free parameters.
Diagonal and off-diagonal elements can have different settings.

# Example
```julia
# Symmetric covariance with variance on diagonal, covariances off-diagonal
Σ = symmetric_free(2, :Σ, init_diag=1.0, init_offdiag=0.1)
# Creates a symmetric matrix where Σ[1,2] = Σ[2,1] (same parameter)
```
"""
function symmetric_free(
        n::Int,
        prefix::Symbol;
        init_diag = 1.0,
        init_offdiag = 0.0,
        lower_diag = 0.0,
        lower_offdiag = -Inf,
        upper_diag = Inf,
        upper_offdiag = Inf
)
    mat = Matrix{Any}(undef, n, n)

    # Diagonal elements
    for i in 1:n
        name = Symbol("$(prefix)_$(i)_$(i)")
        mat[i, i] = FreeParam(name, init = init_diag, lower = lower_diag, upper = upper_diag)
    end

    # Off-diagonal (lower triangle, upper references same params)
    for j in 1:n
        for i in (j + 1):n
            name = Symbol("$(prefix)_$(i)_$(j)")
            param = FreeParam(
                name,
                init = init_offdiag,
                lower = lower_offdiag,
                upper = upper_offdiag
            )
            mat[i, j] = param
            mat[j, i] = param  # Symmetric - same parameter reference
        end
    end
    mat
end

"""
    selection_mat(m, r)

Create a selection matrix R of size m×r.
Common patterns:
- `selection_mat(m, m)` → Identity (all states have shocks)
- `selection_mat(m, r)` where r < m → First r states have shocks

# Example
```julia
# 3 states, 2 shocks (first 2 states)
R = selection_mat(3, 2)
# [1 0]
# [0 1]
# [0 0]
```
"""
function selection_mat(m::Int, r::Int)
    R = zeros(Float64, m, r)
    for i in 1:min(m, r)
        R[i, i] = 1.0
    end
    R
end

"""
    companion_mat(n)

Create an n×n companion matrix structure (for AR(n) models).
Returns a matrix with 1s on the superdiagonal and FreeParams in the first column.

# Example
```julia
T = companion_mat(2, :φ)  # AR(2) transition
# [FreeParam(:φ_1)  1.0]
# [FreeParam(:φ_2)  0.0]
```
"""
function companion_mat(n::Int, prefix::Symbol; init = 0.5, lower = -Inf, upper = Inf)
    mat = Matrix{Any}(undef, n, n)
    fill!(mat, 0.0)

    # First column: AR coefficients
    for i in 1:n
        name = Symbol("$(prefix)_$i")
        mat[i, 1] = FreeParam(name, init = init/i, lower = lower, upper = upper)
    end

    # Superdiagonal: 1s
    for i in 1:(n - 1)
        mat[i, i + 1] = 1.0
    end
    mat
end

# Helper to broadcast scalar to vector
_to_vec(x::Real, n::Int) = fill(x, n)
_to_vec(x::AbstractVector, n::Int) = (@assert length(x) == n; x)

"""
    block_diag(blocks...) -> Matrix{Any}

Combine matrices into a block-diagonal layout, splicing free parameters and
fixed values from each block.

Each `block` may be:

- An `AbstractMatrix` of `Real`, `FreeParam`, or mixed (`Matrix{Any}` from
  helpers like [`diag_free`](@ref), [`companion_mat`](@ref),
  [`symmetric_free`](@ref), [`lower_triangular_free`](@ref)).
- An `AbstractVector` (treated as a column-vector block of size `length × 1`).
- A `Real` (treated as a 1×1 fixed block).
- A `FreeParam` (treated as a 1×1 free-parameter block).

The result is a `Matrix{Any}` of size
`(Σ size(b, 1), Σ size(b, 2))`, with each block placed on the diagonal and
zeros elsewhere. The result is accepted directly by [`custom_ssm`](@ref) for
any of `Z`, `T`, `H`, `Q`, `R`, `P1`.

`FreeParam` identity is preserved: if two cells inside an input block reference
the same `FreeParam` object, the corresponding cells in the output remain
tied. This matters for blocks built with [`symmetric_free`](@ref).

`CovFree` and `MatrixExpr` blocks are not supported here — wrap their parameters
into `custom_ssm` directly when you need them inside a block-diagonal layout.

# Examples

```julia
# Q with two independent free shocks plus a fixed third shock
Q = block_diag(diag_free([:q1, :q2]), [3.0;;])
# 3×3, free on (1,1)/(2,2), fixed 3.0 on (3,3), zeros elsewhere

# T as block-diagonal: AR(2) companion plus a stationary AR(1)
T = block_diag(companion_mat(2, :φ),
               FreeParam(:ρ; init=0.9, lower=-0.99, upper=0.99))

# Rectangular Z: independent loadings for two factor groups
Z = block_diag([FreeParam(:λ1); FreeParam(:λ2)],
               [FreeParam(:λ3); FreeParam(:λ4); FreeParam(:λ5)])
# size(Z) == (5, 2)
```
"""
function block_diag(blocks...)
    isempty(blocks) && throw(ArgumentError("block_diag requires at least one block"))
    mats = map(_block_to_matrix, blocks)
    n_rows = sum(size(m, 1) for m in mats)
    n_cols = sum(size(m, 2) for m in mats)
    out = Matrix{Any}(undef, n_rows, n_cols)
    fill!(out, 0.0)
    row_offset = 0
    col_offset = 0
    @inbounds for m in mats
        nr, nc = size(m)
        for j in 1:nc, i in 1:nr
            out[row_offset + i, col_offset + j] = m[i, j]
        end
        row_offset += nr
        col_offset += nc
    end
    return out
end

# Block normalisation. CovFree / MatrixExpr inputs are special-cased so the
# error is informative instead of "got typeof(x)" from a missing method.
_block_to_matrix(x::AbstractMatrix) = x
_block_to_matrix(x::AbstractVector) = reshape(collect(x), :, 1)
_block_to_matrix(x::Real) = fill(Float64(x), 1, 1)
_block_to_matrix(x::FreeParam) = fill(x, 1, 1)
function _block_to_matrix(x)
    if _is_cov_free(x) || _is_matrix_expr(x)
        throw(ArgumentError(
            "block_diag does not yet support CovFree or MatrixExpr blocks; " *
            "got $(typeof(x)). Place these matrices directly in custom_ssm instead."))
    end
    throw(ArgumentError(
        "block_diag block must be an AbstractMatrix, AbstractVector, Real, " *
        "or FreeParam; got $(typeof(x))."))
end

"""
    cov_free(n, prefix; init_σ=1.0)

Create a full n×n positive definite covariance matrix specification.

Uses Σ = D * Corr * D decomposition where:
- D = Diagonal(σ) with n standard deviation parameters (positive, named `prefix_σ_1`, etc.)
- Corr = L'L from `corr_cholesky_factor(n)` with n(n-1)/2 correlation parameters (unconstrained)

Total parameters: n + n(n-1)/2 = n(n+1)/2

# Arguments
- `n::Int`: Matrix dimension
- `prefix::Symbol`: Prefix for parameter names
- `init_σ::Float64=1.0`: Initial value for standard deviation parameters

# Example
```julia
# 2×2 covariance with 3 parameters: Q_σ_1, Q_σ_2, Q_corr_1
Q = cov_free(2, :Q)

# 3×3 covariance with 6 parameters
Q = cov_free(3, :Q, init_σ=0.5)
```

# See also
- [`diag_free`](@ref): For diagonal covariance matrices (no correlation)
"""
cov_free(n::Int, prefix::Symbol; init_σ::Float64 = 1.0) = CovFree(n, prefix, init_σ)
