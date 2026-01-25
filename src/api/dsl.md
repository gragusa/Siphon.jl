# DSL & Templates

This page documents the domain-specific language for model specification and pre-built templates.

## Core Types

```@docs
Siphon.DSL.SSMParameter
Siphon.DSL.SSMSpec
Siphon.DSL.FixedValue
Siphon.DSL.ParameterRef
```

## Model Introspection

```@docs
Siphon.DSL.param_names
Siphon.DSL.n_params
Siphon.DSL.initial_values
Siphon.DSL.param_bounds
```

## Model Building

```@docs
Siphon.DSL.build_linear_state_space
Siphon.DSL.ssm_loglik
Siphon.DSL.objective_function
```

## Pre-Built Templates

### Local Level Model

```@docs
Siphon.DSL.local_level
```

### Local Linear Trend

```@docs
Siphon.DSL.local_linear_trend
```

### AR(1) Model

```@docs
Siphon.DSL.ar1
```

### ARMA Model

```@docs
Siphon.DSL.arma
```

### Dynamic Factor Model

```@docs
Siphon.DSL.dynamic_factor
```

## Custom Model Specification

```@docs
Siphon.DSL.custom_ssm
Siphon.DSL.FreeParam
```

## Parameter Expressions

```@docs
Siphon.DSL.ParamExpr
Siphon.DSL.MatrixExpr
```

## DNS/Svensson Yield Curve Helpers

```@docs
Siphon.DSL.build_dns_loadings
Siphon.DSL.build_svensson_loadings
Siphon.DSL.dns_loading1
Siphon.DSL.dns_loading2
```
