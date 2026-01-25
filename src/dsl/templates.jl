"""
    templates.jl

Pre-built state-space model templates for common models.

# Specifying Fixed vs Free Parameters

All template functions accept keyword arguments to control which parameters are
estimated vs fixed:

- Pass a `Symbol` or `nothing` to estimate (optimize) that parameter
- Pass a `Real` value to fix the parameter at that value

## Examples

```julia
# Estimate both var_obs and var_level (default)
spec = local_level()

# Fix var_obs at 225.0, estimate var_level starting from 100.0
spec = local_level(var_obs=225.0, var_level=100.0)
# Wait, that's ambiguous! Use explicit :free marker:

# Fix var_obs at 225.0, estimate var_level
spec = local_level(var_obs=225.0, var_level=:free)

# Estimate var_obs, fix var_level at 100.0
spec = local_level(var_obs=:free, var_level=100.0)

# Set initial values for estimation
spec = local_level(var_obs=(init=225.0,), var_level=(init=100.0,))

# Full control: bounds + initial value
spec = local_level(var_obs=(init=225.0, lower=1.0, upper=40000.0), var_level=:free)
```
"""

export local_level, local_linear_trend, ar1, arma, dynamic_factor, dns_model

"""
    ParamSpec

Union type for parameter specification in templates:
- `Symbol` (:free) - estimate with default bounds/init
- `Real` - fix at this value
- `NamedTuple` - estimate with custom (init=, lower=, upper=)
"""
const ParamSpec = Union{Symbol, Real, NamedTuple}

"""
    parse_param_spec(name, spec, default_init, default_lower, default_upper)

Parse a parameter specification into either an SSMParameter (free) or FixedValue.
Returns (is_free::Bool, param_or_value)
"""
function parse_param_spec(name::Symbol, spec::Symbol, default_init, default_lower, default_upper)
    spec == :free || throw(ArgumentError("Unknown symbol :$spec for parameter $name. Use :free or a numeric value."))
    (true, SSMParameter(name; lower=default_lower, upper=default_upper, init=default_init))
end

function parse_param_spec(name::Symbol, spec::Real, default_init, default_lower, default_upper)
    (false, FixedValue(Float64(spec)))
end

function parse_param_spec(name::Symbol, spec::NamedTuple, default_init, default_lower, default_upper)
    init = get(spec, :init, default_init)
    lower = get(spec, :lower, default_lower)
    upper = get(spec, :upper, default_upper)
    (true, SSMParameter(name; lower=lower, upper=upper, init=init))
end

"""
    DiffuseSpec

Union type for diffuse initialization specification:
- `Bool` - `true` for approximate diffuse (large P1), `false` for non-diffuse
- `Symbol` - `:exact` for exact diffuse initialization (Durbin-Koopman method)
"""
const DiffuseSpec = Union{Bool, Symbol}

"""
    local_level(; var_obs=:free, var_level=:free, diffuse=true)

Create specification for local level (random walk + noise) model.

Model:
    yₜ = μₜ + εₜ,  εₜ ~ N(0, var_obs)
    μₜ₊₁ = μₜ + ηₜ,  ηₜ ~ N(0, var_level)

# Arguments
- `var_obs`: Observation noise variance. Use `:free` to estimate, a number to fix,
             or `(init=, lower=, upper=)` for custom estimation settings.
- `var_level`: Level noise variance. Same options as var_obs.
- `diffuse`: Initialization mode:
  - `true` (default): Approximate diffuse (P1 = 1e7)
  - `false`: Non-diffuse (P1 = small value)
  - `:exact`: Exact diffuse initialization (P1_star=0, P1_inf=I)

# Examples
```julia
# Estimate both (default)
spec = local_level()

# Fix observation variance, estimate level variance
spec = local_level(var_obs=225.0, var_level=:free)

# Use exact diffuse initialization
spec = local_level(diffuse=:exact)

# Custom initial value and bounds
spec = local_level(var_obs=(init=225.0, lower=1.0, upper=10000.0))
```
"""
function local_level(; var_obs::ParamSpec=:free, var_level::ParamSpec=:free, diffuse::DiffuseSpec=true)
    # Parse parameter specifications (variance parameters, lower=0.0)
    obs_free, obs_spec = parse_param_spec(:var_obs, var_obs, 1.0, 0.0, Inf)
    level_free, level_spec = parse_param_spec(:var_level, var_level, 1.0, 0.0, Inf)

    # Build parameter list (only free parameters)
    params = SSMParameter{Float64}[]
    obs_free && push!(params, obs_spec)
    level_free && push!(params, level_spec)

    # Z = [1]
    Z = SSMMatrixSpec((1, 1))
    Z.elements[(1, 1)] = FixedValue(1.0)

    # H = var_obs (either ParameterRef or FixedValue)
    H = SSMMatrixSpec((1, 1))
    if obs_free
        H.elements[(1, 1)] = ParameterRef(:var_obs)
    else
        H.elements[(1, 1)] = FixedValue(obs_spec.value)
    end

    # T = [1]
    T = SSMMatrixSpec((1, 1))
    T.elements[(1, 1)] = FixedValue(1.0)

    # R = [1]
    R = SSMMatrixSpec((1, 1))
    R.elements[(1, 1)] = FixedValue(1.0)

    # Q = var_level (either ParameterRef or FixedValue)
    Q = SSMMatrixSpec((1, 1))
    if level_free
        Q.elements[(1, 1)] = ParameterRef(:var_level)
    else
        Q.elements[(1, 1)] = FixedValue(level_spec.value)
    end

    # Initial state mean
    a1 = [FixedValue(0.0)]

    # Initial state covariance - depends on diffuse mode
    if diffuse === :exact
        # Exact diffuse: P1_star = 0, P1_inf = I
        P1 = SSMMatrixSpec((1, 1))
        P1.elements[(1, 1)] = FixedValue(0.0)  # P1_star = 0
        P1_inf = SSMMatrixSpec((1, 1))
        P1_inf.elements[(1, 1)] = FixedValue(1.0)  # P1_inf = I
        return SSMSpec(:LocalLevel, 1, 1, 1, params, Z, H, T, R, Q, a1, P1, P1_inf)
    else
        # Approximate diffuse or non-diffuse
        P1_val = diffuse ? 1e7 : 1e4
        P1 = SSMMatrixSpec((1, 1))
        P1.elements[(1, 1)] = FixedValue(P1_val)
        return SSMSpec(:LocalLevel, 1, 1, 1, params, Z, H, T, R, Q, a1, P1)
    end
end

"""
    local_linear_trend(; var_obs=:free, var_level=:free, var_slope=:free, diffuse=true)

Create specification for local linear trend model.

Model:
    yₜ = μₜ + εₜ,  εₜ ~ N(0, var_obs)
    μₜ₊₁ = μₜ + νₜ + ηₜ,  ηₜ ~ N(0, var_level)
    νₜ₊₁ = νₜ + ζₜ,  ζₜ ~ N(0, var_slope)

State: [μₜ, νₜ]

# Arguments
- `var_obs`, `var_level`, `var_slope`: Use `:free` to estimate, a number to fix,
  or `(init=, lower=, upper=)` for custom settings.
- `diffuse`: Initialization mode:
  - `true` (default): Approximate diffuse (P1 = 1e7*I)
  - `false`: Non-diffuse (P1 = 1e4*I)
  - `:exact`: Exact diffuse initialization (P1_star=0, P1_inf=I)
"""
function local_linear_trend(; var_obs::ParamSpec=:free, var_level::ParamSpec=:free,
                              var_slope::ParamSpec=:free, diffuse::DiffuseSpec=true)
    # Parse parameter specifications (variance parameters, lower=0.0)
    obs_free, obs_spec = parse_param_spec(:var_obs, var_obs, 1.0, 0.0, Inf)
    level_free, level_spec = parse_param_spec(:var_level, var_level, 0.01, 0.0, Inf)
    slope_free, slope_spec = parse_param_spec(:var_slope, var_slope, 0.0001, 0.0, Inf)

    # Build parameter list (only free parameters)
    params = SSMParameter{Float64}[]
    obs_free && push!(params, obs_spec)
    level_free && push!(params, level_spec)
    slope_free && push!(params, slope_spec)

    # Z = [1 0]
    Z = SSMMatrixSpec((1, 2))
    Z.elements[(1, 1)] = FixedValue(1.0)

    # H = var_obs
    H = SSMMatrixSpec((1, 1))
    H.elements[(1, 1)] = obs_free ? ParameterRef(:var_obs) : FixedValue(obs_spec.value)

    # T = [1 1; 0 1]
    T = SSMMatrixSpec((2, 2))
    T.elements[(1, 1)] = FixedValue(1.0)
    T.elements[(1, 2)] = FixedValue(1.0)
    T.elements[(2, 2)] = FixedValue(1.0)

    # R = I
    R = SSMMatrixSpec((2, 2))
    R.elements[(1, 1)] = FixedValue(1.0)
    R.elements[(2, 2)] = FixedValue(1.0)

    # Q = diag(var_level, var_slope)
    Q = SSMMatrixSpec((2, 2))
    Q.elements[(1, 1)] = level_free ? ParameterRef(:var_level) : FixedValue(level_spec.value)
    Q.elements[(2, 2)] = slope_free ? ParameterRef(:var_slope) : FixedValue(slope_spec.value)

    # Initial state mean
    a1 = [FixedValue(0.0), FixedValue(0.0)]

    # Initial state covariance - depends on diffuse mode
    if diffuse === :exact
        # Exact diffuse: P1_star = 0, P1_inf = I
        P1 = SSMMatrixSpec((2, 2))
        P1.elements[(1, 1)] = FixedValue(0.0)
        P1.elements[(2, 2)] = FixedValue(0.0)
        P1_inf = SSMMatrixSpec((2, 2))
        P1_inf.elements[(1, 1)] = FixedValue(1.0)
        P1_inf.elements[(2, 2)] = FixedValue(1.0)
        return SSMSpec(:LocalLinearTrend, 2, 1, 2, params, Z, H, T, R, Q, a1, P1, P1_inf)
    else
        P1_val = diffuse ? 1e7 : 1e4
        P1 = SSMMatrixSpec((2, 2))
        P1.elements[(1, 1)] = FixedValue(P1_val)
        P1.elements[(2, 2)] = FixedValue(P1_val)
        return SSMSpec(:LocalLinearTrend, 2, 1, 2, params, Z, H, T, R, Q, a1, P1)
    end
end

"""
    ar1(; ρ=:free, var_obs=:free, var_state=:free, diffuse=true)

Create specification for AR(1) plus noise model.

Model:
    yₜ = xₜ + εₜ,  εₜ ~ N(0, var_obs)
    xₜ₊₁ = ρ xₜ + ηₜ,  ηₜ ~ N(0, var_state)

# Arguments
- `ρ`: AR coefficient (|ρ| < 1). Use `:free`, a number, or `(init=, lower=, upper=)`.
- `var_obs`, `var_state`: Noise variances. Same options.
- `diffuse`: Initialization mode:
  - `true` (default): Approximate diffuse (P1 = 1e7)
  - `false`: Non-diffuse (P1 = 1e4)
  - `:exact`: Exact diffuse initialization (P1_star=0, P1_inf=I)
"""
function ar1(; ρ::ParamSpec=:free, var_obs::ParamSpec=:free, var_state::ParamSpec=:free, diffuse::DiffuseSpec=true)
    # Parse parameter specifications
    rho_free, rho_spec = parse_param_spec(:ρ, ρ, 0.8, -0.9999, 0.9999)
    obs_free, obs_spec = parse_param_spec(:var_obs, var_obs, 1.0, 0.0, Inf)
    state_free, state_spec = parse_param_spec(:var_state, var_state, 1.0, 0.0, Inf)

    # Build parameter list
    params = SSMParameter{Float64}[]
    rho_free && push!(params, rho_spec)
    obs_free && push!(params, obs_spec)
    state_free && push!(params, state_spec)

    # Z = [1]
    Z = SSMMatrixSpec((1, 1))
    Z.elements[(1, 1)] = FixedValue(1.0)

    # H = var_obs
    H = SSMMatrixSpec((1, 1))
    H.elements[(1, 1)] = obs_free ? ParameterRef(:var_obs) : FixedValue(obs_spec.value)

    # T = [ρ]
    T = SSMMatrixSpec((1, 1))
    T.elements[(1, 1)] = rho_free ? ParameterRef(:ρ) : FixedValue(rho_spec.value)

    # R = [1]
    R = SSMMatrixSpec((1, 1))
    R.elements[(1, 1)] = FixedValue(1.0)

    # Q = var_state
    Q = SSMMatrixSpec((1, 1))
    Q.elements[(1, 1)] = state_free ? ParameterRef(:var_state) : FixedValue(state_spec.value)

    # Initial state mean
    a1 = [FixedValue(0.0)]

    # Initial state covariance - depends on diffuse mode
    if diffuse === :exact
        P1 = SSMMatrixSpec((1, 1))
        P1.elements[(1, 1)] = FixedValue(0.0)
        P1_inf = SSMMatrixSpec((1, 1))
        P1_inf.elements[(1, 1)] = FixedValue(1.0)
        return SSMSpec(:AR1, 1, 1, 1, params, Z, H, T, R, Q, a1, P1, P1_inf)
    else
        P1_val = diffuse ? 1e7 : 1e4
        P1 = SSMMatrixSpec((1, 1))
        P1.elements[(1, 1)] = FixedValue(P1_val)
        return SSMSpec(:AR1, 1, 1, 1, params, Z, H, T, R, Q, a1, P1)
    end
end

"""
    arma(p::Int, q::Int; ar_init=nothing, ma_init=nothing, var_init=1.0, diffuse=true)

Create specification for ARMA(p,q) model in state-space form.

Uses the innovations state-space representation.

Parameters: ar coefficients φ₁...φₚ, ma coefficients θ₁...θᵧ, var (innovation variance)
"""
function arma(p::Int, q::Int; ar_init=nothing, ma_init=nothing, var_init=1.0)
    # State dimension is max(p, q+1)
    r = max(p, q + 1)

    # Default initial values
    if ar_init === nothing
        ar_init = fill(0.5 / p, p)
    end
    if ma_init === nothing
        ma_init = fill(0.0, q)
    end

    # Parameters
    params = SSMParameter{Float64}[]

    # AR coefficients
    for i in 1:p
        push!(params, SSMParameter(Symbol("φ$i"); lower=-Inf, upper=Inf, init=ar_init[i]))
    end

    # MA coefficients
    for i in 1:q
        push!(params, SSMParameter(Symbol("θ$i"); lower=-Inf, upper=Inf, init=ma_init[i]))
    end

    # Innovation variance
    push!(params, SSMParameter(:var; lower=0.0, upper=Inf, init=var_init))

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
        if i <= p
            T.elements[(i, 1)] = ParameterRef(Symbol("φ$i"))
        end
        if i < r
            T.elements[(i, i + 1)] = FixedValue(1.0)
        end
    end

    # R = [1; θ₁; θ₂; ...; θᵧ; 0; ...]
    R = SSMMatrixSpec((r, 1))
    R.elements[(1, 1)] = FixedValue(1.0)
    for i in 1:q
        R.elements[(i + 1, 1)] = ParameterRef(Symbol("θ$i"))
    end

    # Q = var
    Q = SSMMatrixSpec((1, 1))
    Q.elements[(1, 1)] = ParameterRef(:var)

    # Initial state (diffuse)
    a1 = [FixedValue(0.0) for _ in 1:r]
    P1 = SSMMatrixSpec((r, r))
    for i in 1:r
        P1.elements[(i, i)] = FixedValue(1e7)
    end

    SSMSpec(Symbol("ARMA($p,$q)"), r, 1, 1, params, Z, H, T, R, Q, a1, P1)
end

"""
    dynamic_factor(n_obs, n_factors; factor_lags=1, obs_lags=0, correlated_errors=false,
                   loadings_init=0.5, ar_init=0.5, var_obs_init=1.0,
                   var_factor_init=1.0, diffuse=true)

Create specification for a dynamic factor model with VAR factor dynamics and
optional lagged factor loadings.

Model:
    yₜ = Λ₀ fₜ + Λ₁ fₜ₋₁ + ... + Λₛ fₜ₋ₛ + εₜ,  εₜ ~ N(0, H)
    fₜ = Φ₁ fₜ₋₁ + Φ₂ fₜ₋₂ + ... + Φₚ fₜ₋ₚ + ηₜ,  ηₜ ~ N(0, Q)

where:
- yₜ is n_obs × 1 vector of observations
- fₜ is n_factors × 1 vector of latent factors
- Λₗ is n_obs × n_factors factor loadings matrix for lag l (l = 0, ..., s)
- Φₗ is n_factors × n_factors AR coefficient matrix for lag l (diagonal)
- H is observation error covariance (diagonal or full if correlated_errors=true)
- Q is factor innovation covariance (diagonal)
- s = obs_lags, p = factor_lags

The model is cast in companion form with state vector:
    αₜ = [fₜ', fₜ₋₁', ..., fₜ₋ₘ₊₁']'  where m = max(p, s+1)

# Identification

For identification, the first n_factors rows of Λ₀ (contemporaneous loadings)
are set to an identity matrix:
- λ₀_{i,i} = 1 for i ≤ n_factors
- λ₀_{i,j} = 0 for i < j ≤ n_factors

All other loadings (including lagged) are free parameters.

# Arguments
- `n_obs::Int`: Number of observable variables
- `n_factors::Int`: Number of latent factors (must be < n_obs)
- `factor_lags::Int=1`: Number of lags in factor VAR dynamics (p)
- `obs_lags::Int=0`: Number of lagged factors that load onto observations (s).
                     If s > 0, observations depend on fₜ, fₜ₋₁, ..., fₜ₋ₛ.
- `correlated_errors::Bool=false`: If true, H is a full covariance matrix (inexact DFM).
                                   If false, H is diagonal (exact DFM).
- `loadings_init::Real=0.5`: Initial value for free factor loadings
- `ar_init::Real=0.5`: Initial value for AR coefficients (divided by lag number)
- `var_obs_init::Real=1.0`: Initial value for observation error variances
- `var_factor_init::Real=1.0`: Initial value for factor innovation variances
- `diffuse::Bool=true`: Use diffuse initialization for factors

# State Space Representation

State dimension is `n_factors * max(factor_lags, obs_lags + 1)`.

State vector: αₜ = [fₜ', fₜ₋₁', ..., fₜ₋ₘ₊₁']'

Transition matrix (companion form):
    T = [Φ₁  Φ₂  ... Φₚ  0  ... 0]   (padded with zeros if m > p)
        [I   0   ... 0   0  ... 0]
        [0   I   ... 0   0  ... 0]
        [⋮   ⋮   ⋱   ⋮   ⋮  ⋱   ⋮]
        [0   0   ... I   0  ... 0]

Observation matrix: Z = [Λ₀  Λ₁  ...  Λₛ  0  ...  0]

# Example
```julia
# Standard 1-factor model (no lagged loadings)
spec = dynamic_factor(4, 1)

# 1-factor model with 2 factor lags, no lagged loadings
spec = dynamic_factor(4, 1; factor_lags=2)
# yₜ = Λ₀ fₜ + εₜ
# fₜ = φ₁ fₜ₋₁ + φ₂ fₜ₋₂ + ηₜ

# 1-factor model with lagged loadings (s=1)
spec = dynamic_factor(4, 1; factor_lags=2, obs_lags=1)
# yₜ = Λ₀ fₜ + Λ₁ fₜ₋₁ + εₜ
# fₜ = φ₁ fₜ₋₁ + φ₂ fₜ₋₂ + ηₜ
# Parameters: λ0_i_j (contemporaneous), λ1_i_j (lag 1 loadings)

# 2-factor model with obs_lags=2, factor_lags=1
spec = dynamic_factor(5, 2; factor_lags=1, obs_lags=2)
# yₜ = Λ₀ fₜ + Λ₁ fₜ₋₁ + Λ₂ fₜ₋₂ + εₜ
# State: [f₁ₜ, f₂ₜ, f₁ₜ₋₁, f₂ₜ₋₁, f₁ₜ₋₂, f₂ₜ₋₂]
```

# Parameters Created
- `λ0_i_j`: Contemporaneous loadings (i > j for identification in first k rows)
- `λl_i_j`: Loadings for lag l (l = 1, ..., obs_lags), all free
- `φ_i_l`: AR coefficient for factor i at lag l (l = 1, ..., factor_lags)
- `var_obs_i`: Observation error variances (if correlated_errors=false)
- `H_σ_i`, `H_corr_i`: Covariance parameters (if correlated_errors=true, σ for D*Corr*D)
- `var_factor_i`: Factor innovation variances
"""
function dynamic_factor(n_obs::Int, n_factors::Int;
                        factor_lags::Int=1,
                        obs_lags::Int=0,
                        correlated_errors::Bool=false,
                        loadings_init::Real=0.5,
                        ar_init::Real=0.5,
                        var_obs_init::Real=1.0,
                        var_factor_init::Real=1.0,
                        diffuse::Bool=true)
    # Validation
    n_factors < n_obs || throw(ArgumentError("n_factors must be < n_obs for identification"))
    n_factors >= 1 || throw(ArgumentError("n_factors must be >= 1"))
    factor_lags >= 1 || throw(ArgumentError("factor_lags must be >= 1"))
    obs_lags >= 0 || throw(ArgumentError("obs_lags must be >= 0"))

    # State dimension = n_factors * max(factor_lags, obs_lags + 1)
    # We need enough lags in the state to support both factor dynamics and observation loadings
    n_state_lags = max(factor_lags, obs_lags + 1)
    n_states = n_factors * n_state_lags
    # Number of shocks = n_factors (only current factors have innovations)
    n_shocks = n_factors

    # Build using custom_ssm approach for flexibility
    params = SSMParameter{Float64}[]
    param_set = Set{Symbol}()
    matrix_exprs = Dict{Symbol,Any}()

    # ==========================================
    # Z matrix (loadings): n_obs × n_states
    # ==========================================
    # Z = [Λ₀  Λ₁  ...  Λₛ  0  ...  0]
    # where each Λₗ is n_obs × n_factors
    #
    # For identification: upper-left block of Λ₀ is identity
    Z = SSMMatrixSpec((n_obs, n_states))

    # Contemporaneous loadings (Λ₀) - columns 1:n_factors
    for i in 1:n_obs
        for j in 1:n_factors
            if i == j && i <= n_factors
                # Diagonal of identity block: fixed at 1
                Z.elements[(i, j)] = FixedValue(1.0)
            elseif i < j && i <= n_factors
                # Upper triangle of identity block: fixed at 0
                Z.elements[(i, j)] = FixedValue(0.0)
            else
                # Free loading parameter (contemporaneous)
                pname = Symbol("λ0_$(i)_$(j)")
                push!(params, SSMParameter(pname; lower=-Inf, upper=Inf, init=loadings_init))
                push!(param_set, pname)
                Z.elements[(i, j)] = ParameterRef(pname)
            end
        end
    end

    # Lagged loadings (Λ₁, Λ₂, ..., Λₛ) - all free parameters
    for lag in 1:obs_lags
        col_offset = lag * n_factors
        for i in 1:n_obs
            for j in 1:n_factors
                col = col_offset + j
                # All lagged loadings are free (no identification restrictions)
                pname = Symbol("λ$(lag)_$(i)_$(j)")
                # Use smaller initial values for lagged loadings
                init_val = loadings_init / (lag + 1)
                push!(params, SSMParameter(pname; lower=-Inf, upper=Inf, init=init_val))
                push!(param_set, pname)
                Z.elements[(i, col)] = ParameterRef(pname)
            end
        end
    end
    # Remaining columns (beyond obs_lags) are zero - use default

    # ==========================================
    # H matrix (observation errors): n_obs × n_obs
    # ==========================================
    if correlated_errors
        # Full covariance matrix using CovMatrixExpr (D*Corr*D parameterization)
        # Note: CovMatrixExpr uses std dev σ parameters (kept as-is per design decision)
        prefix = :H
        n = n_obs

        # Create σ parameters (positive std devs for D*Corr*D)
        σ_param_names = Symbol[]
        for i in 1:n
            pname = Symbol("$(prefix)_σ_$i")
            if !(pname in param_set)
                push!(params, SSMParameter(pname; lower=0.0, upper=Inf, init=sqrt(var_obs_init)))
                push!(param_set, pname)
            end
            push!(σ_param_names, pname)
        end

        # Create correlation parameters (unconstrained)
        n_corr = n * (n - 1) ÷ 2
        corr_param_names = Symbol[]
        for i in 1:n_corr
            pname = Symbol("$(prefix)_corr_$i")
            if !(pname in param_set)
                push!(params, SSMParameter(pname; lower=-Inf, upper=Inf, init=0.0))
                push!(param_set, pname)
            end
            push!(corr_param_names, pname)
        end

        # Store CovMatrixExpr
        matrix_exprs[:H] = CovMatrixExpr(n, σ_param_names, corr_param_names)
        H = SSMMatrixSpec((n_obs, n_obs))  # Placeholder
    else
        # Diagonal covariance (variance parameters)
        H = SSMMatrixSpec((n_obs, n_obs))
        for i in 1:n_obs
            pname = Symbol("var_obs_$i")
            push!(params, SSMParameter(pname; lower=0.0, upper=Inf, init=var_obs_init))
            push!(param_set, pname)
            H.elements[(i, i)] = ParameterRef(pname)
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

    # First n_factors rows: AR coefficient blocks (only up to factor_lags)
    for i in 1:n_factors
        for lag in 1:factor_lags
            # Column block for lag l starts at (lag-1)*n_factors + 1
            col = (lag - 1) * n_factors + i
            pname = Symbol("φ_$(i)_$(lag)")
            # Initialize with decreasing values for higher lags
            init_val = ar_init / lag
            push!(params, SSMParameter(pname; lower=-0.9999, upper=0.9999, init=init_val))
            push!(param_set, pname)
            T.elements[(i, col)] = ParameterRef(pname)
        end
        # Columns beyond factor_lags are zero (default)
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
    for i in 1:n_factors
        pname = Symbol("var_factor_$i")
        push!(params, SSMParameter(pname; lower=0.0, upper=Inf, init=var_factor_init))
        push!(param_set, pname)
        Q.elements[(i, i)] = ParameterRef(pname)
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

    SSMSpec(:DynamicFactor, n_states, n_obs, n_shocks, params,
            Z, H, T, R, Q, a1, P1, matrix_exprs)
end

# ============================================
# Dynamic Nelson-Siegel Model
# ============================================

"""
    dns_model(maturities; T_structure=:diagonal, H_structure=:diagonal,
              Q_structure=:full, λ_init=0.0609, ...)

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
  - `:diagonal` - Diagonal (p parameters: `H_1`, ..., `H_p`)
  - `:scalar` - Single variance for all (1 parameter: `H`)
- `Q_structure::Symbol=:full`: State covariance structure
  - `:full` - Full 3×3 covariance via Cholesky (6 parameters)
  - `:diagonal` - Diagonal (3 parameters: `Q_L`, `Q_S`, `Q_C`)
- `λ_init`, `λ_lower`, `λ_upper`: Decay parameter settings
- `T_init`: Initial value for T diagonal elements (scalar or 3-vector)
- `Q_init`: Initial value for Q variances (scalar or 3-vector)
- `H_init`: Initial value for H variances (scalar or p-vector)
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
function dns_model(maturities::AbstractVector{<:Real};
                   T_structure::Symbol=:diagonal,
                   H_structure::Symbol=:diagonal,
                   Q_structure::Symbol=:full,
                   λ_init::Real=0.0609,
                   λ_lower::Real=0.001,
                   λ_upper::Real=1.0,
                   T_init=0.95,
                   Q_init=0.01,
                   H_init=0.01,
                   diffuse::Bool=true)
    p = length(maturities)
    n_states = 3
    n_shocks = 3

    params = SSMParameter{Float64}[]
    param_set = Set{Symbol}()
    matrix_exprs = Dict{Symbol,Any}()

    # ==========================================
    # Z matrix: DNS loadings via MatrixExpr
    # ==========================================
    Z_expr = build_dns_loadings(maturities; λ_init=λ_init, λ_lower=λ_lower, λ_upper=λ_upper)
    matrix_exprs[:Z] = Z_expr

    # Add λ parameter from MatrixExpr
    for z_param in Z_expr.params
        push!(params, z_param)
        push!(param_set, z_param.name)
    end

    # Placeholder Z spec (actual matrix built from MatrixExpr)
    Z = SSMMatrixSpec((p, n_states))

    # ==========================================
    # T matrix: Factor dynamics
    # ==========================================
    T = _build_dns_T_spec(T_structure, T_init, params, param_set)

    # ==========================================
    # H matrix: Observation covariance
    # ==========================================
    H = _build_dns_H_spec(H_structure, p, H_init, params, param_set, matrix_exprs)

    # ==========================================
    # Q matrix: State covariance
    # ==========================================
    Q = _build_dns_Q_spec(Q_structure, Q_init, params, param_set, matrix_exprs)

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

    SSMSpec(:DNS, n_states, p, n_shocks, params, Z, H, T, R, Q, a1, P1, matrix_exprs)
end

"""
Build T matrix specification for DNS model based on structure.
"""
function _build_dns_T_spec(structure::Symbol, T_init, params, param_set)
    T = SSMMatrixSpec((3, 3))

    if structure == :full
        # Full 3×3 VAR matrix
        for i in 1:3
            for j in 1:3
                pname = Symbol("T_$(i)_$(j)")
                # Diagonal elements get T_init, off-diagonal get small values
                if i == j
                    init_val = T_init isa Real ? T_init : T_init[i]
                else
                    init_val = 0.0
                end
                push!(params, SSMParameter(pname; init=init_val, lower=-0.9999, upper=0.9999))
                push!(param_set, pname)
                T.elements[(i, j)] = ParameterRef(pname)
            end
        end
    elseif structure == :diagonal
        # Diagonal AR(1) for each factor
        factor_names = [:T_L, :T_S, :T_C]
        for (i, fname) in enumerate(factor_names)
            init_val = T_init isa Real ? T_init : T_init[i]
            push!(params, SSMParameter(fname; init=init_val, lower=-0.9999, upper=0.9999))
            push!(param_set, fname)
            T.elements[(i, i)] = ParameterRef(fname)
        end
    elseif structure == :ar1
        # Single AR coefficient for all factors
        init_val = T_init isa Real ? T_init : T_init[1]
        push!(params, SSMParameter(:φ; init=init_val, lower=-0.9999, upper=0.9999))
        push!(param_set, :φ)
        for i in 1:3
            T.elements[(i, i)] = ParameterRef(:φ)
        end
    else
        throw(ArgumentError("T_structure must be :full, :diagonal, or :ar1, got :$structure"))
    end

    T
end

"""
Build H matrix specification for DNS model based on structure.
"""
function _build_dns_H_spec(structure::Symbol, p::Int, H_init, params, param_set, matrix_exprs)
    H = SSMMatrixSpec((p, p))

    if structure == :diagonal
        # Diagonal observation variances
        for i in 1:p
            pname = Symbol("H_$i")
            init_val = H_init isa Real ? H_init : H_init[i]
            push!(params, SSMParameter(pname; init=init_val, lower=0.0, upper=Inf))
            push!(param_set, pname)
            H.elements[(i, i)] = ParameterRef(pname)
        end
    elseif structure == :scalar
        # Single variance for all observations
        push!(params, SSMParameter(:H; init=H_init isa Real ? H_init : H_init[1], lower=0.0, upper=Inf))
        push!(param_set, :H)
        for i in 1:p
            H.elements[(i, i)] = ParameterRef(:H)
        end
    else
        throw(ArgumentError("H_structure must be :diagonal or :scalar, got :$structure"))
    end

    H
end

"""
Build Q matrix specification for DNS model based on structure.
"""
function _build_dns_Q_spec(structure::Symbol, Q_init, params, param_set, matrix_exprs)
    Q = SSMMatrixSpec((3, 3))

    if structure == :full
        # Full covariance via CovMatrixExpr (D*Corr*D parameterization)
        prefix = :Q
        n = 3

        # Create σ parameters (std devs) and store original variances
        σ_param_names = Symbol[]
        var_init = Float64[]
        factor_labels = [:L, :S, :C]
        for (i, fl) in enumerate(factor_labels)
            pname = Symbol("Q_σ_$fl")
            var_val = Q_init isa Real ? Float64(Q_init) : Float64(Q_init[i])
            init_val = sqrt(var_val)
            push!(params, SSMParameter(pname; init=init_val, lower=0.0, upper=Inf))
            push!(param_set, pname)
            push!(σ_param_names, pname)
            push!(var_init, var_val)  # Store original variance
        end

        # Create correlation parameters (unconstrained for Cholesky)
        n_corr = n * (n - 1) ÷ 2  # 3 correlations
        corr_param_names = Symbol[]
        for i in 1:n_corr
            pname = Symbol("Q_corr_$i")
            push!(params, SSMParameter(pname; init=0.0, lower=-Inf, upper=Inf))
            push!(param_set, pname)
            push!(corr_param_names, pname)
        end

        # Store CovMatrixExpr with original variances
        matrix_exprs[:Q] = CovMatrixExpr(n, σ_param_names, corr_param_names, var_init)

    elseif structure == :diagonal
        # Diagonal variances
        factor_names = [:Q_L, :Q_S, :Q_C]
        for (i, fname) in enumerate(factor_names)
            init_val = Q_init isa Real ? Q_init : Q_init[i]
            push!(params, SSMParameter(fname; init=init_val, lower=0.0, upper=Inf))
            push!(param_set, fname)
            Q.elements[(i, i)] = ParameterRef(fname)
        end
    else
        throw(ArgumentError("Q_structure must be :full or :diagonal, got :$structure"))
    end

    Q
end
