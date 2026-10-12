"""
    templates.jl

Pre-built state-space model templates for common models.

# Fixed and free parameters

Every parameter keyword of a template takes either

- a number, which fixes the parameter at that value, or
- a `FreeParam`, which marks it for estimation. Unset fields take the
  template's defaults: `FreeParam()` uses the template's bounds and starting
  value, and `FreeParam(init = 225.0)` changes only the starting value. The
  name may be omitted or repeat the template's name for that parameter.

The default for every parameter keyword is `FreeParam()`. A keyword that covers
a group of parameters (ARMA coefficients, factor loadings, …) takes one value
for every member, or a vector with one value per member.

```julia
spec = local_level()                                   # estimate both variances
spec = local_level(var_obs = 225.0)                    # fix var_obs
spec = local_level(var_obs = FreeParam(init = 225.0))  # estimate, from 225
spec = local_level(var_obs = FreeParam(lower = 1.0, upper = 4e4))
spec = arma(2, 1; ar = [FreeParam(), 0.0])            # φ2 fixed at 0
```
"""

# Add the template parameter `name`, given by the user as `x`, to `params` and
# return the matrix element that refers to it. `lower`, `upper` and `init` are
# the template's defaults for the fields `x` leaves unset; `kw` is the keyword
# that `x` was passed as.
function _template_param!(params, param_set, x, name::Symbol, lower, upper, init;
        kw::Symbol = name)
    x isa Real && return FixedValue(Float64(x))
    x isa FreeParam || throw(ArgumentError(
        "`$kw` must be a number (to fix the parameter) or a FreeParam (to estimate " *
        "it), e.g. FreeParam(init = 1.0); got $(repr(x))"))
    x.name === nothing || x.name === name ||
        throw(ArgumentError(
            "`$kw` sets the parameter :$name; pass a FreeParam without a name, " *
            "got FreeParam(:$(x.name))"))
    lo = something(x.lower, lower)
    up = something(x.upper, upper)
    p = if x.init === nothing
        SSMParameter(name; lower = lo, upper = up, default_init = true,
            init = lo < init < up ? init : _default_init(lo, up))
    else
        SSMParameter(name; lower = lo, upper = up, init = x.init)
    end
    _add_param!(params, param_set, p)
    return ParameterRef(name)
end

# A group of template parameters: `x` is one value for every member or a vector
# with one value per member. `inits` holds the template's starting values.
function _template_params!(params, param_set, x, names, lower, upper, inits; kw::Symbol)
    xs = _group_values(x, length(names), kw)
    return MatrixElement[_template_param!(params, param_set, xs[i], names[i], lower,
                             upper, inits[i]; kw) for i in eachindex(names, xs, inits)]
end

function _group_values(x::AbstractVector, n::Int, kw::Symbol)
    length(x) == n || throw(DimensionMismatch(
        "`$kw` covers $n parameters, got a vector of length $(length(x))"))
    return x
end

function _group_values(x, n::Int, kw::Symbol)
    n > 1 && x isa FreeParam && x.name !== nothing &&
        throw(ArgumentError(
            "one FreeParam for all $n parameters of `$kw` cannot carry a name; " *
            "pass a vector to set each parameter separately"))
    return fill(x, n)
end

# Starting standard deviations of a full covariance (D * Corr * D) whose keyword
# `kw` gives the variances. Only the starting variances can be set there.
function _cov_sd_inits(x, n::Int, init; kw::Symbol)
    map(_group_values(x, n, kw)) do xi
        xi isa FreeParam && xi.name === nothing && xi.lower === nothing &&
        xi.upper === nothing || throw(ArgumentError(
            "with a full covariance, `$kw` sets only starting variances, as " *
            "FreeParam(init = v); got $(repr(xi))"))
        sqrt(something(xi.init, init))
    end
end

# Standard deviation and correlation parameters of a full n × n covariance named
# `<prefix>_sd_<label>` and `<prefix>_corr_<i>`.
function _cov_params!(params, param_set, prefix::Symbol, labels, sd_inits)
    sd_names = [Symbol("$(prefix)_sd_$l") for l in labels]
    n = length(labels)
    corr_names = [Symbol("$(prefix)_corr_$i") for i in 1:(n * (n - 1) ÷ 2)]
    for (name, init) in zip(sd_names, sd_inits)
        _add_param!(params, param_set, SSMParameter(name; lower = 0.0, init))
    end
    for name in corr_names
        _add_param!(params, param_set, SSMParameter(name; init = 0.0))
    end
    return sd_names, corr_names
end

"""
    DiffuseSpec

Union type for diffuse initialization specification:
- `Bool` - `true` for approximate diffuse (large P1), `false` for non-diffuse
- `Symbol` - `:exact` for exact diffuse initialization (Durbin-Koopman method)
"""
const DiffuseSpec = Union{Bool, Symbol}

function _check_diffuse(diffuse::DiffuseSpec)
    diffuse isa Bool || diffuse === :exact ||
        throw(ArgumentError(
            "diffuse must be true, false or :exact, got $(repr(diffuse))"))
    return nothing
end

"""
    local_level(; var_obs=FreeParam(), var_level=FreeParam(), diffuse=true)

Create specification for local level (random walk + noise) model.

Model:
    yₜ = μₜ + εₜ,  εₜ ~ N(0, var_obs)
    μₜ₊₁ = μₜ + ηₜ,  ηₜ ~ N(0, var_level)

# Arguments
- `var_obs`: Observation noise variance. A number fixes it; a `FreeParam`
  estimates it (defaults: `lower = 0`, `init = 1.0`).
- `var_level`: Level noise variance. Same options and defaults as `var_obs`.
- `diffuse`: Initialization mode:
  - `true` (default): Approximate diffuse (P1 = 1e7)
  - `false`: Non-diffuse (P1 = 1e4)
  - `:exact`: Exact diffuse initialization (P1_star=0, P1_inf=I)

A starting value left unset is rescaled to the data by maximum likelihood (see
[`optimize_ssm`](@ref)).

# Examples
```julia
# Estimate both (default)
spec = local_level()

# Fix observation variance, estimate level variance
spec = local_level(var_obs = 225.0)

# Use exact diffuse initialization
spec = local_level(diffuse = :exact)

# Custom starting value and bounds
spec = local_level(var_obs = FreeParam(lower = 1.0, upper = 1e4, init = 225.0))
```
"""
function local_level(;
        var_obs = FreeParam(),
        var_level = FreeParam(),
        diffuse::DiffuseSpec = true
)
    _check_diffuse(diffuse)
    params = SSMParameter{Float64}[]
    param_set = Set{Symbol}()

    Z = SSMMatrixSpec((1, 1))
    Z.elements[(1, 1)] = FixedValue(1.0)
    H = SSMMatrixSpec((1, 1))
    H.elements[(1, 1)] = _template_param!(
        params, param_set, var_obs, :var_obs, 0.0, Inf, 1.0)
    T = SSMMatrixSpec((1, 1))
    T.elements[(1, 1)] = FixedValue(1.0)
    R = SSMMatrixSpec((1, 1))
    R.elements[(1, 1)] = FixedValue(1.0)
    Q = SSMMatrixSpec((1, 1))
    Q.elements[(1, 1)] = _template_param!(
        params, param_set, var_level, :var_level, 0.0, Inf, 1.0)
    a1 = [FixedValue(0.0)]
    return _template_spec(:LocalLevel, 1, 1, 1, params, Z, H, T, R, Q, a1, diffuse)
end

# The spec with the initial state covariance that `diffuse` selects: 1e7 I
# (true), 1e4 I (false), or P1_star = 0 with P1_inf = I (:exact).
function _template_spec(name, m, p, r, params, Z, H, T, R, Q, a1, diffuse::DiffuseSpec)
    P1 = SSMMatrixSpec((m, m))
    if diffuse === :exact
        P1_inf = SSMMatrixSpec((m, m))
        for i in 1:m
            P1.elements[(i, i)] = FixedValue(0.0)
            P1_inf.elements[(i, i)] = FixedValue(1.0)
        end
        return SSMSpec(name, m, p, r, params, Z, H, T, R, Q, a1, P1, P1_inf)
    end
    for i in 1:m
        P1.elements[(i, i)] = FixedValue(diffuse ? 1e7 : 1e4)
    end
    return SSMSpec(name, m, p, r, params, Z, H, T, R, Q, a1, P1)
end

"""
    local_linear_trend(; var_obs=FreeParam(), var_level=FreeParam(),
                       var_slope=FreeParam(), diffuse=true)

Create specification for local linear trend model.

Model:
    yₜ = μₜ + εₜ,  εₜ ~ N(0, var_obs)
    μₜ₊₁ = μₜ + νₜ + ηₜ,  ηₜ ~ N(0, var_level)
    νₜ₊₁ = νₜ + ζₜ,  ζₜ ~ N(0, var_slope)

State: [μₜ, νₜ]

# Arguments
- `var_obs`, `var_level`, `var_slope`: Variances. A number fixes one; a
  `FreeParam` estimates it (defaults: `lower = 0` and `init` = 1.0, 0.01 and
  1e-4 respectively).
- `diffuse`: Initialization mode:
  - `true` (default): Approximate diffuse (P1 = 1e7*I)
  - `false`: Non-diffuse (P1 = 1e4*I)
  - `:exact`: Exact diffuse initialization (P1_star=0, P1_inf=I)
"""
function local_linear_trend(;
        var_obs = FreeParam(),
        var_level = FreeParam(),
        var_slope = FreeParam(),
        diffuse::DiffuseSpec = true
)
    _check_diffuse(diffuse)
    params = SSMParameter{Float64}[]
    param_set = Set{Symbol}()

    Z = SSMMatrixSpec((1, 2))
    Z.elements[(1, 1)] = FixedValue(1.0)
    H = SSMMatrixSpec((1, 1))
    H.elements[(1, 1)] = _template_param!(
        params, param_set, var_obs, :var_obs, 0.0, Inf, 1.0)
    T = SSMMatrixSpec((2, 2))
    T.elements[(1, 1)] = FixedValue(1.0)
    T.elements[(1, 2)] = FixedValue(1.0)
    T.elements[(2, 2)] = FixedValue(1.0)
    R = SSMMatrixSpec((2, 2))
    R.elements[(1, 1)] = FixedValue(1.0)
    R.elements[(2, 2)] = FixedValue(1.0)
    Q = SSMMatrixSpec((2, 2))
    Q.elements[(1, 1)] = _template_param!(params, param_set, var_level, :var_level,
        0.0, Inf, 0.01)
    Q.elements[(2, 2)] = _template_param!(params, param_set, var_slope, :var_slope,
        0.0, Inf, 1e-4)
    a1 = [FixedValue(0.0), FixedValue(0.0)]
    return _template_spec(:LocalLinearTrend, 2, 1, 2, params, Z, H, T, R, Q, a1, diffuse)
end

"""
    ar1(; ρ=FreeParam(), var_obs=FreeParam(), var_state=FreeParam(), diffuse=true)

Create specification for AR(1) plus noise model.

Model:
    yₜ = xₜ + εₜ,  εₜ ~ N(0, var_obs)
    xₜ₊₁ = ρ xₜ + ηₜ,  ηₜ ~ N(0, var_state)

# Arguments
- `ρ`: AR coefficient. A number fixes it; a `FreeParam` estimates it (defaults:
  bounds ±0.9999, `init = 0.8`).
- `var_obs`, `var_state`: Noise variances (defaults: `lower = 0`, `init = 1.0`).
- `diffuse`: Initialization mode:
  - `true` (default): Approximate diffuse (P1 = 1e7)
  - `false`: Non-diffuse (P1 = 1e4)
  - `:exact`: Exact diffuse initialization (P1_star=0, P1_inf=I)
"""
function ar1(;
        ρ = FreeParam(),
        var_obs = FreeParam(),
        var_state = FreeParam(),
        diffuse::DiffuseSpec = true
)
    _check_diffuse(diffuse)
    params = SSMParameter{Float64}[]
    param_set = Set{Symbol}()

    T = SSMMatrixSpec((1, 1))
    T.elements[(1, 1)] = _template_param!(params, param_set, ρ, :ρ, -0.9999, 0.9999, 0.8)
    Z = SSMMatrixSpec((1, 1))
    Z.elements[(1, 1)] = FixedValue(1.0)
    H = SSMMatrixSpec((1, 1))
    H.elements[(1, 1)] = _template_param!(
        params, param_set, var_obs, :var_obs, 0.0, Inf, 1.0)
    R = SSMMatrixSpec((1, 1))
    R.elements[(1, 1)] = FixedValue(1.0)
    Q = SSMMatrixSpec((1, 1))
    Q.elements[(1, 1)] = _template_param!(params, param_set, var_state, :var_state,
        0.0, Inf, 1.0)
    a1 = [FixedValue(0.0)]
    return _template_spec(:AR1, 1, 1, 1, params, Z, H, T, R, Q, a1, diffuse)
end

"""
    arma(p::Int, q::Int; ar=FreeParam(), ma=FreeParam(), var=FreeParam(), diffuse=true)

Create specification for ARMA(p,q) model in state-space form.

Uses the innovations state-space representation.

Parameters: AR coefficients `φ1`, …, `φp`, MA coefficients `θ1`, …, `θq`, and
the innovation variance `var`.

# Arguments
- `ar`: AR coefficients, as one value for all of them or a vector of length `p`.
  A number fixes a coefficient; a `FreeParam` estimates it (defaults:
  unbounded, `init = 0.5 / p`).
- `ma`: MA coefficients, as for `ar` with length `q` (default `init = 0.0`).
- `var`: Innovation variance (defaults: `lower = 0`, `init = 1.0`). A starting
  value left unset is rescaled to the data by maximum likelihood (see
  [`optimize_ssm`](@ref)).
- `diffuse`: Initial state covariance:
  - `true` (default): Approximate diffuse (P1 = 1e7*I)
  - `false`: Non-diffuse (P1 = 1e4*I)
  - `:exact`: Exact diffuse initialization (P1_star=0, P1_inf=I)

# Example
```julia
spec = arma(2, 1)                                  # estimate everything
spec = arma(2, 0; ar = [FreeParam(init = 0.5), 0.0])  # φ2 fixed at 0
```
"""
function arma(p::Int, q::Int; ar = FreeParam(), ma = FreeParam(), var = FreeParam(),
        diffuse::DiffuseSpec = true)
    p >= 0 && q >= 0 || throw(ArgumentError("arma orders must be ≥ 0, got ($p, $q)"))
    _check_diffuse(diffuse)
    # State dimension is max(p, q+1)
    r = max(p, q + 1)
    params = SSMParameter{Float64}[]
    param_set = Set{Symbol}()

    φ = _template_params!(params, param_set, ar, [Symbol("φ$i") for i in 1:p],
        -Inf, Inf, fill(0.5 / max(p, 1), p); kw = :ar)
    θ = _template_params!(params, param_set, ma, [Symbol("θ$i") for i in 1:q],
        -Inf, Inf, zeros(q); kw = :ma)
    var_elem = _template_param!(params, param_set, var, :var, 0.0, Inf, 1.0)

    # Z = [1 0 ... 0]
    Z = SSMMatrixSpec((1, r))
    Z.elements[(1, 1)] = FixedValue(1.0)

    # H = 0 (no observation noise in pure ARMA)
    H = SSMMatrixSpec((1, 1))
    H.elements[(1, 1)] = FixedValue(0.0)

    # T = [φ₁ 1 0 ... 0]
    #     [φ₂ 0 1 ... 0]
    #     [⋮  ⋮ ⋮ ⋱  ⋮]
    #     [φᵣ 0 0 ... 0]
    T = SSMMatrixSpec((r, r))
    for i in 1:r
        i <= p && (T.elements[(i, 1)] = φ[i])
        i < r && (T.elements[(i, i + 1)] = FixedValue(1.0))
    end

    # R = [1; θ₁; θ₂; ...; θ_q; 0; ...]
    R = SSMMatrixSpec((r, 1))
    R.elements[(1, 1)] = FixedValue(1.0)
    for i in 1:q
        R.elements[(i + 1, 1)] = θ[i]
    end

    Q = SSMMatrixSpec((1, 1))
    Q.elements[(1, 1)] = var_elem

    a1 = [FixedValue(0.0) for _ in 1:r]
    return _template_spec(
        Symbol("ARMA($p,$q)"), r, 1, 1, params, Z, H, T, R, Q, a1, diffuse)
end

"""
    dynamic_factor(n_obs, n_factors; factor_lags=1, loading_lags=0, correlated_errors=false,
                   loadings=FreeParam(), ar=FreeParam(), var_obs=FreeParam(),
                   var_factor=FreeParam(), diffuse=true)

Create specification for a dynamic factor model with VAR factor dynamics and
optional lagged factor loadings.

Model:
    yₜ = Λ₀ fₜ + Λ₁ fₜ₋₁ + ... + Λₚ fₜ₋ₚ + εₜ,  εₜ ~ N(0, H)
    fₜ = Φ₁ fₜ₋₁ + Φ₂ fₜ₋₂ + ... + Φq fₜ₋q + ηₜ,  ηₜ ~ N(0, Q)

where:
- yₜ is n_obs × 1 vector of observations
- fₜ is n_factors × 1 vector of latent factors
- Λₗ is n_obs × n_factors factor loadings matrix for lag l (l = 0, ..., p)
- Φₗ is n_factors × n_factors AR coefficient matrix for lag l (diagonal)
- H is observation error covariance (diagonal or full if correlated_errors=true)
- Q is factor innovation covariance (diagonal)
- p = loading_lags, q = factor_lags (the notation of `DynamicFactorModel`)

The model is cast in companion form with state vector:
    αₜ = [fₜ', fₜ₋₁', ..., fₜ₋ₛ₊₁']'  where s = max(q, p+1)

# Identification

For identification, the first n_factors rows of Λ₀ (contemporaneous loadings)
form a lower-triangular block with unit diagonal:
- λ₀_{i,i} = 1 for i ≤ n_factors
- λ₀_{i,j} = 0 for i < j ≤ n_factors

All other loadings (including lagged) are free parameters.

# Arguments
- `n_obs::Int`: Number of observable variables
- `n_factors::Int`: Number of latent factors (must be < n_obs)
- `factor_lags::Int=1`: Number of lags in factor VAR dynamics (q)
- `loading_lags::Int=0`: Number of lagged factors that load onto observations (p).
                         If p > 0, observations depend on fₜ, fₜ₋₁, ..., fₜ₋ₚ.
- `correlated_errors::Bool=false`: If true, H is a full covariance matrix (inexact DFM).
                                   If false, H is diagonal (exact DFM).
- `loadings`: The free loadings, in the order of `param_names` (`λ0_i_j`, then
  `λ1_i_j`, …). One value for all of them or a vector; a number fixes a loading
  and a `FreeParam` estimates it (defaults: unbounded, `init` = 0.5 for `Λ₀` and
  0.5 / (l + 1) for `Λₗ`).
- `ar`: AR coefficients `φ_i_l`, ordered by factor and then lag (defaults:
  bounds ±0.9999, `init = 0.5 / l`).
- `var_obs`: Observation error variances `var_obs_i` (defaults: `lower = 0`,
  `init = 1.0`). With `correlated_errors = true`, H is parameterized by standard
  deviations `H_sd_i` and correlations `H_corr_i`, and `var_obs` can only set the
  starting variances, as `FreeParam(init = v)`.
- `var_factor`: Factor innovation variances `var_factor_i` (defaults:
  `lower = 0`, `init = 1.0`).

  Variance starting values left unset are rescaled to the data by maximum
  likelihood (see [`optimize_ssm`](@ref)).
- `diffuse::Bool=true`: Use diffuse initialization for factors

# State Space Representation

State dimension is `n_factors * max(factor_lags, loading_lags + 1)`.

State vector: αₜ = [fₜ', fₜ₋₁', ..., fₜ₋ₛ₊₁']'

Transition matrix (companion form):
    T = [Φ₁  Φ₂  ... Φq  0  ... 0]   (padded with zeros if s > q)
        [I   0   ... 0   0  ... 0]
        [0   I   ... 0   0  ... 0]
        [⋮   ⋮   ⋱   ⋮   ⋮  ⋱   ⋮]
        [0   0   ... I   0  ... 0]

Observation matrix: Z = [Λ₀  Λ₁  ...  Λₚ  0  ...  0]

# Example
```julia
# Standard 1-factor model (no lagged loadings)
spec = dynamic_factor(4, 1)

# 1-factor model with 2 factor lags, no lagged loadings
spec = dynamic_factor(4, 1; factor_lags=2)
# yₜ = Λ₀ fₜ + εₜ
# fₜ = φ₁ fₜ₋₁ + φ₂ fₜ₋₂ + ηₜ

# 1-factor model with lagged loadings (p=1)
spec = dynamic_factor(4, 1; factor_lags=2, loading_lags=1)
# yₜ = Λ₀ fₜ + Λ₁ fₜ₋₁ + εₜ
# fₜ = φ₁ fₜ₋₁ + φ₂ fₜ₋₂ + ηₜ
# Parameters: λ0_i_j (contemporaneous), λ1_i_j (lag 1 loadings)

# 2-factor model with loading_lags=2, factor_lags=1
spec = dynamic_factor(5, 2; factor_lags=1, loading_lags=2)
# yₜ = Λ₀ fₜ + Λ₁ fₜ₋₁ + Λ₂ fₜ₋₂ + εₜ
# State: [f₁ₜ, f₂ₜ, f₁ₜ₋₁, f₂ₜ₋₁, f₁ₜ₋₂, f₂ₜ₋₂]
```

# Parameters Created
- `λ0_i_j`: Contemporaneous loadings (i > j for identification in first k rows)
- `λl_i_j`: Loadings for lag l (l = 1, ..., loading_lags), all free
- `φ_i_l`: AR coefficient for factor i at lag l (l = 1, ..., factor_lags)
- `var_obs_i`: Observation error variances (if correlated_errors=false)
- `H_sd_i`, `H_corr_i`: Standard deviations and correlation parameters of H (if correlated_errors=true)
- `var_factor_i`: Factor innovation variances
"""
function dynamic_factor(
        n_obs::Int,
        n_factors::Int;
        factor_lags::Int = 1,
        loading_lags::Int = 0,
        correlated_errors::Bool = false,
        loadings = FreeParam(),
        ar = FreeParam(),
        var_obs = FreeParam(),
        var_factor = FreeParam(),
        diffuse::Bool = true
)
    # Validation
    n_factors < n_obs ||
        throw(ArgumentError("n_factors must be < n_obs for identification"))
    n_factors >= 1 || throw(ArgumentError("n_factors must be >= 1"))
    factor_lags >= 1 || throw(ArgumentError("factor_lags must be >= 1"))
    loading_lags >= 0 || throw(ArgumentError("loading_lags must be >= 0"))

    # State dimension = n_factors * max(factor_lags, loading_lags + 1)
    # We need enough lags in the state to support both factor dynamics and observation loadings
    n_state_lags = max(factor_lags, loading_lags + 1)
    n_states = n_factors * n_state_lags
    # Number of shocks = n_factors (only current factors have innovations)
    n_shocks = n_factors

    # Build using custom_ssm approach for flexibility
    params = SSMParameter{Float64}[]
    param_set = Set{Symbol}()
    matrix_exprs = Dict{Symbol, AbstractMatrixExpr}()

    # ==========================================
    # Z matrix (loadings): n_obs × n_states
    # ==========================================
    # Z = [Λ₀  Λ₁  ...  Λₛ  0  ...  0]
    # where each Λₗ is n_obs × n_factors
    #
    # For identification: upper-left block of Λ₀ is identity
    Z = SSMMatrixSpec((n_obs, n_states))

    # Free loadings: Λ₀ outside the identification block, then Λ₁, …, Λₚ
    cells = Tuple{Int, Int}[]
    names = Symbol[]
    inits = Float64[]
    for i in 1:n_obs, j in 1:n_factors

        if i <= n_factors && j >= i
            # Identification block: unit diagonal, zero upper triangle
            Z.elements[(i, j)] = FixedValue(i == j ? 1.0 : 0.0)
        else
            push!(cells, (i, j))
            push!(names, Symbol("λ0_$(i)_$(j)"))
            push!(inits, 0.5)
        end
    end
    for lag in 1:loading_lags, i in 1:n_obs, j in 1:n_factors
        push!(cells, (i, lag * n_factors + j))
        push!(names, Symbol("λ$(lag)_$(i)_$(j)"))
        push!(inits, 0.5 / (lag + 1))
    end
    elems = _template_params!(params, param_set, loadings, names, -Inf, Inf, inits;
        kw = :loadings)
    for (cell, elem) in zip(cells, elems)
        Z.elements[cell] = elem
    end
    # Remaining columns (beyond loading_lags) are zero - use default

    # ==========================================
    # H matrix (observation errors): n_obs × n_obs
    # ==========================================
    if correlated_errors
        # Full covariance D * Corr * D, parameterized by standard deviations
        sd_inits = _cov_sd_inits(var_obs, n_obs, 1.0; kw = :var_obs)
        sd_names, corr_names = _cov_params!(params, param_set, :H, 1:n_obs, sd_inits)
        matrix_exprs[:H] = CovMatrixExpr(n_obs, sd_names, corr_names)
        H = SSMMatrixSpec((n_obs, n_obs))  # Placeholder
    else
        H = SSMMatrixSpec((n_obs, n_obs))
        elems = _template_params!(params, param_set, var_obs,
            [Symbol("var_obs_$i") for i in 1:n_obs], 0.0, Inf, ones(n_obs); kw = :var_obs)
        for i in 1:n_obs
            H.elements[(i, i)] = elems[i]
        end
    end

    # ==========================================
    # T matrix (factor dynamics): n_states × n_states
    # ==========================================
    # Companion form:
    # T = [Φ₁  Φ₂  ... Φₚ  0  ... 0]   <- n_factors rows (AR coefficients, padded)
    #     [I   0   ... 0   0  ... 0]
    #     [0   I   ... 0   0  ... 0]
    #     [⋮   ⋮   ⋱   ⋮   ⋮  ⋱   ⋮]
    #     [0   0   ... I   0  ... 0]
    #
    # Each Φₗ is n_factors × n_factors diagonal matrix
    T = SSMMatrixSpec((n_states, n_states))

    # First n_factors rows: AR coefficients for lags 1:factor_lags; the
    # remaining columns are zero (default)
    ar_cells = [(i, (lag - 1) * n_factors + i) for i in 1:n_factors
                for lag in 1:factor_lags]
    elems = _template_params!(params, param_set, ar,
        [Symbol("φ_$(i)_$(lag)") for i in 1:n_factors for lag in 1:factor_lags],
        -0.9999, 0.9999, [0.5 / lag for i in 1:n_factors for lag in 1:factor_lags];
        kw = :ar)
    for (cell, elem) in zip(ar_cells, elems)
        T.elements[cell] = elem
    end

    # Identity blocks for shifting lags (rows n_factors+1 to n_states)
    if n_state_lags > 1
        for lag in 1:(n_state_lags - 1)
            for i in 1:n_factors
                row = lag * n_factors + i
                col = (lag - 1) * n_factors + i
                T.elements[(row, col)] = FixedValue(1.0)
            end
        end
    end

    # ==========================================
    # R matrix (selection): n_states × n_shocks
    # ==========================================
    # Only the first n_factors states (current factors) receive shocks
    # R = [I]   <- n_factors × n_factors identity
    #     [0]   <- (n_states - n_factors) × n_factors zeros
    R = SSMMatrixSpec((n_states, n_shocks))
    for i in 1:n_factors
        R.elements[(i, i)] = FixedValue(1.0)
    end

    # ==========================================
    # Q matrix (factor innovations): n_shocks × n_shocks
    # ==========================================
    # Diagonal covariance for factor innovations (variance parameters)
    Q = SSMMatrixSpec((n_shocks, n_shocks))
    elems = _template_params!(params, param_set, var_factor,
        [Symbol("var_factor_$i") for i in 1:n_factors], 0.0, Inf, ones(n_factors);
        kw = :var_factor)
    for i in 1:n_factors
        Q.elements[(i, i)] = elems[i]
    end

    # ==========================================
    # Initial state
    # ==========================================
    a1 = [FixedValue(0.0) for _ in 1:n_states]
    P1_val = diffuse ? 1e7 : 1e4
    P1 = SSMMatrixSpec((n_states, n_states))
    for i in 1:n_states
        P1.elements[(i, i)] = FixedValue(P1_val)
    end

    SSMSpec(
        :DynamicFactor,
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
        matrix_exprs
    )
end

# ============================================
# Dynamic Nelson-Siegel Model
# ============================================

"""
    dns_model(maturities; T_structure=:diagonal, H_structure=:diagonal,
              Q_structure=:full, λ=FreeParam(), T=FreeParam(), H=FreeParam(),
              Q=FreeParam(), diffuse=true)

Create specification for Dynamic Nelson-Siegel yield curve model.

Model:
    yₜ = Z(λ) fₜ + εₜ,  εₜ ~ N(0, H)
    fₜ₊₁ = T fₜ + ηₜ,   ηₜ ~ N(0, Q)

where:
- yₜ is the p×1 vector of yields at different maturities
- fₜ = [Lₜ, Sₜ, Cₜ]' is the 3×1 factor vector (Level, Slope, Curvature)
- Z(λ) is the p×3 loading matrix dependent on decay parameter λ:
  - Column 1: Level loading (all 1s)
  - Column 2: Slope loading `(1 - exp(-λτ)) / (λτ)`
  - Column 3: Curvature loading `(1 - exp(-λτ)) / (λτ) - exp(-λτ)`
- H is the p×p observation covariance
- T is the 3×3 factor dynamics matrix
- Q is the 3×3 state innovation covariance

# Arguments
- `maturities::AbstractVector`: Vector of maturities (e.g., in months)
- `T_structure::Symbol=:diagonal`: Factor dynamics structure
  - `:full` - Full 3×3 VAR matrix (9 parameters: `T_1_1`, `T_1_2`, ..., `T_3_3`)
  - `:diagonal` - Diagonal AR(1) (3 parameters: `T_L`, `T_S`, `T_C`)
  - `:ar1` - Single AR coefficient for all factors (1 parameter: `φ`)
- `H_structure::Symbol=:diagonal`: Observation error structure
  - `:diagonal` - Diagonal (p variances: `var_obs_1`, ..., `var_obs_p`)
  - `:scalar` - Single variance for all (1 parameter: `var_obs`)
- `Q_structure::Symbol=:full`: State covariance structure
  - `:full` - Full 3×3 covariance (6 parameters: standard deviations `Q_sd_L`, `Q_sd_S`, `Q_sd_C` and correlation parameters `Q_corr_1`, …, `Q_corr_3`)
  - `:diagonal` - Diagonal (3 variances: `var_L`, `var_S`, `var_C`)
- `λ`: Decay parameter. A `FreeParam` estimates it (defaults: bounds
  `[0.001, 1.0]`, `init = 0.0609`); a number fixes it, and `Z` is then a fixed
  matrix.
- `T`: The parameters of `T` in the order of `param_names`, as one value for all
  or a vector (9, 3 or 1 values for `:full`, `:diagonal`, `:ar1`). A number fixes
  an entry; a `FreeParam` estimates it (defaults: bounds ±0.9999, `init` = 0.95
  on the diagonal and 0.0 off it).
- `H`: Observation variances (`p` values for `:diagonal`, 1 for `:scalar`;
  defaults: `lower = 0`, `init = 0.01`).
- `Q`: State variances (3 values; defaults: `lower = 0`, `init = 0.01`). With
  `Q_structure = :full`, Q is parameterized by standard deviations and
  correlations, and `Q` can only set the starting variances, as
  `FreeParam(init = v)`.
- `diffuse::Bool=true`: Use diffuse initialization

# Returns
SSMSpec with Z as MatrixExpr (dependent on λ)

# Estimation
The model can be estimated via:
1. `optimize_ssm(spec, y)` - Full MLE via gradient-based optimization
2. `profile_em_ssm(spec, y)` - Profile likelihood over λ with EM for other parameters

# Example
```julia
maturities = [3, 6, 12, 24, 60, 120]  # months

# Standard DNS with diagonal T
spec = dns_model(maturities)
result = optimize_ssm(spec, yields)

# DNS with full VAR dynamics (for profile EM)
spec = dns_model(maturities; T_structure=:full, Q_structure=:full)
result = profile_em_ssm(spec, yields; λ_grid=0.01:0.01:0.2)

# Extract factors
model = StateSpaceModel(spec, result.θ, size(yields, 2))
kalman_filter!(model, yields)
kalman_smoother!(model)
smooth = smoothed_states(model)
```

# Parameters Created
- `λ`: Nelson-Siegel decay parameter
- T matrix elements (depends on T_structure)
- H variance elements (depends on H_structure)
- Q covariance elements (depends on Q_structure)
"""
function dns_model(
        maturities::AbstractVector{<:Real};
        T_structure::Symbol = :diagonal,
        H_structure::Symbol = :diagonal,
        Q_structure::Symbol = :full,
        λ = FreeParam(),
        T = FreeParam(),
        H = FreeParam(),
        Q = FreeParam(),
        diffuse::Bool = true
)
    p = length(maturities)
    n_states = 3
    n_shocks = 3

    params = SSMParameter{Float64}[]
    param_set = Set{Symbol}()
    matrix_exprs = Dict{Symbol, AbstractMatrixExpr}()

    # Z: Nelson-Siegel loadings, computed from λ when it is estimated
    Z = SSMMatrixSpec((p, n_states))
    λ_elem = _template_param!(params, param_set, λ, :λ, 0.001, 1.0, 0.0609)
    if λ_elem isa FixedValue
        for (i, τ) in enumerate(maturities)
            Z.elements[(i, 1)] = FixedValue(1.0)
            Z.elements[(i, 2)] = FixedValue(dns_loading1(λ_elem.value, τ))
            Z.elements[(i, 3)] = FixedValue(dns_loading2(λ_elem.value, τ))
        end
    else
        λp = only(params)
        matrix_exprs[:Z] = build_dns_loadings(maturities; λ_init = λp.init,
            λ_lower = λp.lower, λ_upper = λp.upper)
    end

    Tmat = _build_dns_T_spec(T_structure, T, params, param_set)
    Hmat = _build_dns_H_spec(H_structure, p, H, params, param_set)
    Qmat = _build_dns_Q_spec(Q_structure, Q, params, param_set, matrix_exprs)

    # ==========================================
    # R matrix: Identity selection
    # ==========================================
    R = SSMMatrixSpec((n_states, n_shocks))
    for i in 1:n_states
        R.elements[(i, i)] = FixedValue(1.0)
    end

    # ==========================================
    # Initial state
    # ==========================================
    a1 = [FixedValue(0.0) for _ in 1:n_states]
    P1_val = diffuse ? 1e7 : 1e4
    P1 = SSMMatrixSpec((n_states, n_states))
    for i in 1:n_states
        P1.elements[(i, i)] = FixedValue(P1_val)
    end

    SSMSpec(:DNS, n_states, p, n_shocks, params, Z, Hmat, Tmat, R, Qmat, a1, P1,
        matrix_exprs)
end

# T, H and Q of `dns_model` for each structure; `x` is the user's keyword value.
function _build_dns_T_spec(structure::Symbol, x, params, param_set)
    T = SSMMatrixSpec((3, 3))
    if structure == :full
        cells = [(i, j) for i in 1:3 for j in 1:3]
        names = [Symbol("T_$(i)_$(j)") for (i, j) in cells]
        inits = [i == j ? 0.95 : 0.0 for (i, j) in cells]
    elseif structure == :diagonal
        cells = [(i, i) for i in 1:3]
        names = [:T_L, :T_S, :T_C]
        inits = fill(0.95, 3)
    elseif structure == :ar1
        cells = [(1, 1)]
        names = [:φ]
        inits = [0.95]
    else
        throw(ArgumentError("T_structure must be :full, :diagonal, or :ar1, got :$structure"))
    end
    elems = _template_params!(params, param_set, x, names, -0.9999, 0.9999, inits; kw = :T)
    for (cell, elem) in zip(cells, elems)
        T.elements[cell] = elem
    end
    if structure == :ar1
        T.elements[(2, 2)] = T.elements[(3, 3)] = elems[1]
    end
    return T
end

function _build_dns_H_spec(structure::Symbol, p::Int, x, params, param_set)
    H = SSMMatrixSpec((p, p))
    if structure == :diagonal
        elems = _template_params!(params, param_set, x,
            [Symbol("var_obs_$i") for i in 1:p], 0.0, Inf, fill(0.01, p); kw = :H)
    elseif structure == :scalar
        elems = fill(
            _template_param!(params, param_set, x, :var_obs, 0.0, Inf, 0.01;
                kw = :H), p)
    else
        throw(ArgumentError("H_structure must be :diagonal or :scalar, got :$structure"))
    end
    for i in 1:p
        H.elements[(i, i)] = elems[i]
    end
    return H
end

function _build_dns_Q_spec(structure::Symbol, x, params, param_set, matrix_exprs)
    Q = SSMMatrixSpec((3, 3))
    if structure == :full
        # D * Corr * D, parameterized by standard deviations
        sd_inits = _cov_sd_inits(x, 3, 0.01; kw = :Q)
        sd_names, corr_names = _cov_params!(params, param_set, :Q, (:L, :S, :C), sd_inits)
        matrix_exprs[:Q] = CovMatrixExpr(3, sd_names, corr_names, sd_inits .^ 2)
    elseif structure == :diagonal
        elems = _template_params!(params, param_set, x, [:var_L, :var_S, :var_C],
            0.0, Inf, fill(0.01, 3); kw = :Q)
        for i in 1:3
            Q.elements[(i, i)] = elems[i]
        end
    else
        throw(ArgumentError("Q_structure must be :full or :diagonal, got :$structure"))
    end
    return Q
end
