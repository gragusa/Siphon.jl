# DSL & Templates

This page documents the domain-specific language for model specification and pre-built templates.

## Core Types

```@docs
Siphon.AbstractSSMSpec
Siphon.SSMParameter
Siphon.SSMSpec
Siphon.FixedValue
Siphon.ParameterRef
```

## Model Introspection

```@docs
Siphon.param_names
Siphon.n_params
Siphon.initial_values
Siphon.param_bounds
```

## Model Building

```@docs
Siphon.build_linear_state_space
Siphon.ssm_loglik
Siphon.objective_function
```

## Pre-Built Templates

### Local Level Model

```@docs
Siphon.local_level
```

### Local Linear Trend

```@docs
Siphon.local_linear_trend
```

### AR(1) Model

```@docs
Siphon.ar1
```

### ARMA Model

```@docs
Siphon.arma
```

### Dynamic Factor Model

```@docs
Siphon.dynamic_factor
```

## Custom Model Specification

```@docs
Siphon.custom_ssm
Siphon.FreeParam
```

## Parameter Expressions

```@docs
Siphon.MatrixExpr
```

## DNS/Svensson Yield Curve Helpers

```@docs
Siphon.build_dns_loadings
Siphon.build_svensson_loadings
Siphon.dns_loading1
Siphon.dns_loading2
```
