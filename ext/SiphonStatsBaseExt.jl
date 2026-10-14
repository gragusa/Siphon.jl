module SiphonStatsBaseExt

using Siphon: Siphon, StateSpaceModel, _quantile_normal
using LinearAlgebra: diag
using SpecialFunctions: erfc
using StatsBase: StatsBase, CoefTable

"""
    coeftable(model::StateSpaceModel; level = 0.95, static = :auto) -> CoefTable

Table of the fitted parameters with standard errors from [`vcov`](@ref), z
statistics, two-sided normal p-values for a zero value, and normal confidence
intervals at `level`. Requires StatsBase to be loaded.
"""
function StatsBase.coeftable(model::StateSpaceModel; level::Real = 0.95,
        static::Symbol = :auto)
    0 < level < 1 || throw(DomainError(level, "level must be in (0, 1)"))
    θ = StatsBase.coef(model)
    se = sqrt.(diag(StatsBase.vcov(model; static)))
    z = θ ./ se
    p = erfc.(abs.(z) ./ sqrt(2))
    q = _quantile_normal((1 + level) / 2)
    pct = round(100 * level; digits = 2)
    pct_str = isinteger(pct) ? string(Int(pct)) : string(pct)
    return CoefTable(hcat(θ, se, z, p, θ .- q .* se, θ .+ q .* se),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $pct_str%", "Upper $pct_str%"],
        StatsBase.coefnames(model), 4, 3)
end

end # module
