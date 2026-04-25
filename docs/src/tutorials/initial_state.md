# [Initial State Conventions and MARSS Compatibility](@id initial_state)

This page explains how Siphon.jl handles the initial state distribution
`(a₁, P₁)`: where it comes from, what each `fit!` method does with it,
and how to translate from MARSS's conventions when porting models.

## Model

Siphon.jl uses the standard linear-Gaussian state space form

```
Observation:  yₜ = Z αₜ + εₜ,    εₜ ~ N(0, H)
State:        αₜ₊₁ = T αₜ + R ηₜ,  ηₜ ~ N(0, Q)
Initial:      α₁ ~ N(a₁, P₁)
```

The Kalman filter starts at `t = 1` with the prior `α₁ ~ N(a₁, P₁)` and
applies the first measurement update `v₁ = y₁ - Z·a₁`. So `(a₁, P₁)` is the
prior over the state *at the time of the first observation* — the
convention MARSS calls `tinitx = 1`.

## Where `(a₁, P₁)` come from

You never pass `a1` or `P1` to `fit!`. They are properties of the spec.

### Built-in templates

Each template sets a sensible default. The `diffuse` keyword controls the
covariance:

| Template | `a₁` | `P₁` (default `diffuse=true`) | `diffuse=false` | `diffuse=:exact` |
|---|---|---|---|---|
| `local_level()` | `[0]` | `[1e7]` | `[1e4]` | `P₁_star = [0]`, `P₁_inf = [1]` |
| `local_linear_trend()` | `[0, 0]` | `1e7 · I₂` | `1e4 · I₂` | exact-diffuse split |
| `ar1()` | `[0]` | `[1e7]` | `[1e4]` | exact-diffuse split |
| `arma(p, q)` | zeros | `1e7 · I` | `1e4 · I` | exact-diffuse split |

`diffuse = true` is essentially "ignore the prior, let the data dominate"
and is the right default for stationary models with enough data.
`diffuse = :exact` triggers Durbin–Koopman exact diffuse initialization,
which is the principled choice when one or more states are
non-stationary (e.g. a random walk level).

### `custom_ssm`

`a1` and `P1` are required keyword arguments. You pass them as plain
matrices/vectors of `Float64`s (fixed prior), or you can mark elements
as `FreeParam(...)` to make the initial state itself estimable:

```julia
spec = custom_ssm(
    Z = [1.0],
    H = [FreeParam(:var_obs, init=100.0, lower=0.0)],
    T = [1.0], R = [1.0],
    Q = [FreeParam(:var_state, init=1.0, lower=0.0)],
    a1 = [FreeParam(:μ₀, init=0.0)],         # estimate the initial mean
    P1 = [FreeParam(:p₀, init=1.0, lower=0.0)],  # and variance
)
```

## How `fit!` treats `(a₁, P₁)`

Both estimation paths consume whatever `(a₁, P₁)` the spec encodes.
**Neither `fit!(EM, ...)` nor `fit!(MLE, ...)` accepts `tinitx`, `V0`,
`x0`, or `update_initial_state` keyword arguments.** The spec is the
single source of truth.

### `fit!(MLE(), model, y)`

`(a₁, P₁)` are rebuilt from the parameter vector at every objective call.

- If `a1` / `P1` cells are `FixedValue`s (the template default), they
  stay constant across the optimization.
- If any cell is a `FreeParam`, it is treated as a model parameter and
  is **optimized jointly with `Z, H, T, R, Q`**.

So MLE *can* fit the initial state — you just have to declare it as
free at spec time. There's no separate switch.

### `fit!(EM(), model, y)`

`(a₁, P₁)` are copied into the in-place workspace once, before the EM
loop, and held **fixed for every iteration**. Even when `a1` / `P1`
contain `FreeParam`s, the EM M-step does *not* currently update them.

This is the simplest correct behaviour: with a diffuse prior the
initial state has negligible effect, and with a spec-specified informative
prior you presumably want it respected. If you need MARSS's
"update `(a₁, P₁)` from the smoother every iteration" behaviour, use
`fit!(MLE())` with `a1`/`P1` declared free, or fall back to the lower-level
`em_estimate!` and update the workspace's `a1`/`P1` between calls.

| Method | Initial state behaviour |
|---|---|
| `fit!(MLE(), m, y)` | Re-evaluates spec each call. `FreeParam` cells in `a1`/`P1` → optimized. `FixedValue` cells → constant. |
| `fit!(EM(), m, y)` | Loaded from spec once. **Held fixed** through all EM iterations regardless of `FreeParam`/`FixedValue`. |

## Migrating from MARSS

MARSS's default is `tinitx = 0`: you pass `(x₀, V₀)` at *t = 0*, and MARSS
propagates one step before the filter starts. Siphon expects the prior
already at `t = 1`, so do the propagation yourself when constructing the
spec:

```julia
using Siphon, LinearAlgebra

# What you'd write in MARSS with tinitx=0:
x0 = zeros(m)
V0 = 100.0 * Matrix(I, m, m)
T_init = ...   # transition matrix
R       = Matrix(I, m, m)
Q_init = ...   # state covariance

# Translate to Siphon's t=1 convention:
a1 = T_init * x0                              # = zeros(m) when x0 is zero
P1 = T_init * V0 * T_init' + R * Q_init * R'   # one step of dynamics

spec = custom_ssm(Z=..., H=..., T=T_init, R=R, Q=Q_init, a1=a1, P1=P1)
```

For MARSS's `tinitx = 1` mode, `a1 = x0` and `P1 = V0` directly.

| MARSS setting | Siphon `(a₁, P₁)` |
|---|---|
| `tinitx = 0`, `x0`, `V0` | `a1 = T * x0`, `P1 = T * V0 * T' + R * Q * R'` |
| `tinitx = 1`, `x0`, `V0` | `a1 = x0`, `P1 = V0` |

At identical parameter values Siphon and MARSS produce the same
log-likelihood to machine precision (DNS model, MARSS converged
parameters: MARSS = 430.032178464672, Siphon = 430.032178461962, Δ ≈
2.7e-9).

## Recipes

**Diffuse prior on a stationary model:**

```julia
spec = local_level()  # diffuse=true → P1 = 1e7
fit!(MLE(), StateSpaceModel(spec, n), y)
```

**Exact diffuse initialization** (correct treatment of non-stationary states):

```julia
spec = local_level(diffuse=:exact)
fit!(MLE(), StateSpaceModel(spec, n), y)
```

**Tight informative prior** (e.g. you know `α₁ ≈ 100 ± 5`):

```julia
spec = custom_ssm(...; a1 = [100.0], P1 = [25.0])
fit!(MLE(), StateSpaceModel(spec, n), y)
```

**Estimate the initial state by MLE** (treats `α₁` as a model parameter):

```julia
spec = custom_ssm(...;
    a1 = [FreeParam(:μ₀, init=0.0)],
    P1 = [FreeParam(:p₀, init=1.0, lower=0.0)])
fit!(MLE(), StateSpaceModel(spec, n), y)
parameters(model).μ₀   # estimated initial mean
parameters(model).p₀   # estimated initial variance
```

(EM cannot estimate the initial state today — see the table above.)
