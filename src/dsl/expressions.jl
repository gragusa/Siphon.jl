"""
    expressions.jl

Support for functional dependencies between parameters and matrix elements.
Useful for models like Dynamic Nelson-Siegel where matrix entries are
complex functions of parameters and external data.
"""

export ParamExpr, MatrixExpr, build_dns_loadings

"""
    ParamExpr(params, data, expr)

A matrix element that is a function of parameters and external data.

# Arguments
- `params`: Symbol or tuple of Symbols for parameter names
- `data`: NamedTuple of external data used in the expression
- `expr`: Function `(param_values..., data...) -> scalar`

# Example: Nelson-Siegel loading
```julia
τ = 30  # maturity in months
# Loading: (1 - exp(-τ*λ)) / (τ*λ)
elem = ParamExpr(:λ, (τ=τ,), (λ, τ) -> (1 - exp(-τ*λ)) / (τ*λ))
```
"""
struct ParamExpr
    params::Tuple{Vararg{Symbol}}
    data::NamedTuple
    expr::Function
end

# Convenience constructors
ParamExpr(param::Symbol, data::NamedTuple, expr::Function) = ParamExpr((param,), data, expr)

ParamExpr(params::Tuple{Vararg{Symbol}}, data::Real, expr::Function) =
    ParamExpr(params, (val = data,), (args...) -> expr(args[1:length(params)]..., data))

ParamExpr(param::Symbol, data::Real, expr::Function) =
    ParamExpr((param,), (val = data,), (p, v) -> expr(p, data))

"""
    MatrixExpr(params, data, builder)

A full matrix that is built from parameters and external data.

# Arguments
- `params`: Vector of SSMParameter specs for parameters used
- `data`: NamedTuple of external data
- `builder`: Function `(θ_dict, data) -> Matrix` where θ_dict maps param names to values

# Example: Dynamic Nelson-Siegel factor loadings
```julia
maturities = [3, 6, 12, 24, 60, 120]  # months

function dns_loadings(θ, data)
    λ = θ[:λ]
    τ = data.maturities
    p = length(τ)
    Z = ones(p, 3)
    for i in 1:p
        x = τ[i] * λ
        Z[i, 2] = (1 - exp(-x)) / x
        Z[i, 3] = Z[i, 2] - exp(-x)
    end
    Z
end

Z = MatrixExpr(
    [SSMParameter(:λ, init=0.0609, lower=0.001, upper=1.0)],
    (maturities=maturities,),
    dns_loadings
)
```
"""
struct MatrixExpr
    params::Vector{SSMParameter{Float64}}
    data::NamedTuple
    builder::Function
    dims::Tuple{Int,Int}  # Expected output dimensions
end

function MatrixExpr(params, data::NamedTuple, builder::Function; dims = nothing)
    # If dims not provided, try to infer by calling builder with dummy values
    if dims === nothing
        θ_dummy = Dict(p.name => p.init for p in params)
        test_mat = builder(θ_dummy, data)
        dims = size(test_mat)
    end
    MatrixExpr(params, data, builder, dims)
end

# ============================================
# Nelson-Siegel / Svensson helpers
# ============================================

"""
    dns_loading1(λ, τ)

First slope loading for Dynamic Nelson-Siegel: `(1 - exp(-λτ)) / (λτ)`
"""
function dns_loading1(λ, τ)
    x = λ * τ
    x < 1e-10 ? 1.0 - x/2 : (1 - exp(-x)) / x
end

"""
    dns_loading2(λ, τ)

Curvature loading for Dynamic Nelson-Siegel: `(1 - exp(-λτ)) / (λτ) - exp(-λτ)`
"""
function dns_loading2(λ, τ)
    dns_loading1(λ, τ) - exp(-λ * τ)
end

"""
    build_dns_loadings(maturities; λ_init=0.0609, λ_lower=0.001, λ_upper=1.0)

Create a MatrixExpr for Dynamic Nelson-Siegel factor loadings.

# Arguments
- `maturities`: Vector of maturities (e.g., in months)
- `λ_init`, `λ_lower`, `λ_upper`: Parameter settings for decay rate λ

# Returns
A MatrixExpr that builds the p×3 loading matrix Z where:
- Column 1: Level factor (all 1s)
- Column 2: Slope factor `(1 - exp(-λτ)) / (λτ)`
- Column 3: Curvature factor `(1 - exp(-λτ)) / (λτ) - exp(-λτ)`

# Example
```julia
Z = build_dns_loadings([3, 6, 12, 24, 60, 120])

spec = custom_ssm(
    Z = Z,
    H = diag_free(6, :var_obs, init=0.01),
    T = [FreeParam(:φ_L, init=0.99, lower=0.0, upper=0.9999) 0.0 0.0;
         0.0 FreeParam(:φ_S, init=0.99, lower=0.0, upper=0.9999) 0.0;
         0.0 0.0 FreeParam(:φ_C, init=0.99, lower=0.0, upper=0.9999)],
    R = identity_mat(3),
    Q = diag_free([:var_L, :var_S, :var_C], init=0.01),
    a1 = [0.0, 0.0, 0.0],
    P1 = 1e4 * identity_mat(3)
)
```
"""
function build_dns_loadings(
    maturities::AbstractVector{<:Real};
    λ_init = 0.0609,
    λ_lower = 0.001,
    λ_upper = 1.0,
)
    p = length(maturities)

    function builder(θ, data)
        λ = θ[:λ]
        τ = data.maturities
        Z = ones(Float64, length(τ), 3)
        for i in eachindex(τ)
            Z[i, 2] = dns_loading1(λ, τ[i])
            Z[i, 3] = dns_loading2(λ, τ[i])
        end
        Z
    end

    params = [SSMParameter(:λ; init = λ_init, lower = λ_lower, upper = λ_upper)]

    MatrixExpr(params, (maturities = collect(Float64, maturities),), builder; dims = (p, 3))
end

"""
    build_svensson_loadings(maturities; λ1_init=0.0609, λ2_init=0.03, ...)

Create a MatrixExpr for Svensson (4-factor) yield curve loadings.

Adds a second curvature factor with separate decay rate λ2.
"""
function build_svensson_loadings(
    maturities::AbstractVector{<:Real};
    λ1_init = 0.0609,
    λ1_lower = 0.001,
    λ1_upper = 1.0,
    λ2_init = 0.03,
    λ2_lower = 0.001,
    λ2_upper = 1.0,
)
    p = length(maturities)

    function builder(θ, data)
        λ1, λ2 = θ[:λ1], θ[:λ2]
        τ = data.maturities
        Z = ones(Float64, length(τ), 4)
        for i in eachindex(τ)
            Z[i, 2] = dns_loading1(λ1, τ[i])
            Z[i, 3] = dns_loading2(λ1, τ[i])
            Z[i, 4] = dns_loading2(λ2, τ[i])
        end
        Z
    end

    params = [
        SSMParameter(:λ1; init = λ1_init, lower = λ1_lower, upper = λ1_upper),
        SSMParameter(:λ2; init = λ2_init, lower = λ2_lower, upper = λ2_upper),
    ]

    MatrixExpr(params, (maturities = collect(Float64, maturities),), builder; dims = (p, 4))
end
