# ============================================
# Missing data
# ============================================
#
# Inside the package a missing observation is NaN. The public entry points also
# accept arrays with `missing` entries and convert them with `_nan_obs`.

"""
    missing_to_nan(y::AbstractArray)

Return `y` with every `missing` replaced by `NaN`.

The element type is the floating-point type of the non-missing values, so
`Union{Missing, Float32}` becomes `Float32` and `Union{Missing, Int}` becomes
`Float64`. An array without `missing` entries is returned as `float(y)`, which is `y`
itself when its elements are already floating point.

The filters and estimators accept `missing` entries directly, so this is only needed
to obtain the NaN-coded array yourself.
"""
function missing_to_nan(y::AbstractArray{<:Union{Missing, Real}})
    T = float(nonmissingtype(eltype(y)))
    result = similar(y, T)
    for i in eachindex(y, result)
        val = y[i]
        result[i] = ismissing(val) ? T(NaN) : T(val)
    end
    return result
end

missing_to_nan(y::AbstractArray{<:Real}) = float(y)

# An array of only `missing` values has no element type to keep.
missing_to_nan(y::AbstractArray{Missing}) = fill!(similar(y, Float64), NaN)

"""
    nan_to_missing(y::AbstractArray{<:Real})

Return a copy of `y` with every `NaN` replaced by `missing`. The element type is
`Union{Missing, eltype(y)}`.
"""
function nan_to_missing(y::AbstractArray{<:Real})
    result = similar(y, Union{Missing, eltype(y)})
    for i in eachindex(y, result)
        result[i] = isnan(y[i]) ? missing : y[i]
    end
    return result
end

# Observations as the filters expect them: NaN-coded, with `missing` converted.
_nan_obs(y::AbstractArray{<:Real}) = y
_nan_obs(y::AbstractArray{<:Union{Missing, Real}}) = missing_to_nan(y)

_isabsent(x) = ismissing(x) || isnan(x)

"""
    ismissing_period(y_t::AbstractVector) -> Bool

Return whether no value of the observation vector `y_t` is present, that is, every
entry is `NaN` or `missing`. These are the periods marked by `missing_mask`.

See also [`isincomplete_period`](@ref).
"""
ismissing_period(y_t::AbstractVector) = all(_isabsent, y_t)

"""
    isincomplete_period(y_t::AbstractVector) -> Bool

Return whether at least one value of the observation vector `y_t` is `NaN` or
`missing`. This includes periods where every value is absent.

See also [`ismissing_period`](@ref).
"""
isincomplete_period(y_t::AbstractVector) = any(_isabsent, y_t)

"""
    count_missing_periods(y::AbstractMatrix) -> Int

Count the periods (columns of the `p × n` matrix `y`) in which no value is present.
This is `count(missing_mask(result))` for a filter run on `y`.
"""
count_missing_periods(y::AbstractMatrix) =
    count(t -> ismissing_period(view(y, :, t)), axes(y, 2))

"""
    count_incomplete_periods(y::AbstractMatrix) -> Int

Count the periods (columns of the `p × n` matrix `y`) in which at least one value is
absent, including those in which every value is absent.
"""
count_incomplete_periods(y::AbstractMatrix) =
    count(t -> isincomplete_period(view(y, :, t)), axes(y, 2))
