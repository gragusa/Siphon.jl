# Dynamic Factor Models

`DynamicFactorModel` is a large-N dynamic factor model with VAR factors and
AR idiosyncratic errors, estimated by EM. For small models with custom
restrictions, the [`dynamic_factor`](@ref) template builds an `SSMSpec`
instead; see the Dynamic Factor Models tutorial for when to use each.

`isconverged`, `niterations`, `loglikelihood`, `parameters` and
`system_matrices` also apply to a `DynamicFactorModel`.

## Model

```@docs
DynamicFactorModel
Siphon.DynamicFactorModelSpec
```

## Factors

```@docs
factors
factors_cov
filtered_factors
filtered_factors_cov
```

## Parameters

```@docs
loadings
var_coefficients
factor_innovation_cov
ar_coefficients
idiosyncratic_variances
```
