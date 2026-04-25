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
single source of truth: cells declared `FixedValue` stay fixed,
cells declared `FreeParam` are estimated.

### `fit!(MLE(), model, y)`

`(a₁, P₁)` are rebuilt from the parameter vector at every objective call.

- `FixedValue` cells stay constant across the optimization.
- `FreeParam` cells are treated as model parameters and **optimized
  jointly with `Z, H, T, R, Q`** by LBFGS.

### `fit!(EM(), model, y)`

`(a₁, P₁)` are loaded from the spec once. After each E-step the M-step
updates `FreeParam` cells in `a1` / `P1` to the smoother's
`E[α₁ | y₁:n]` and `Var[α₁ | y₁:n]` respectively (the standard
Shumway–Stoffer / MARSS closed forms). `FixedValue` cells stay constant
throughout, just like in MLE.

Templates (`local_level`, `ar1`, …) declare `a1` and `P1` as
`FixedValue`s, so they behave identically to the prior fixed-prior
implementation. Only specs that explicitly use `FreeParam` in `a1`/`P1`
see EM update those cells — and in that case the user has asked for it.

| Method | Initial state behaviour |
|---|---|
| `fit!(MLE(), m, y)` | Re-evaluates spec each call. `FreeParam` cells in `a1`/`P1` → optimized jointly. `FixedValue` cells → constant. |
| `fit!(EM(), m, y)` | Loaded from spec once. `FreeParam` cells in `a1`/`P1` → updated to the smoothed `E[α₁ y]` / `Var[α₁ y]` each iteration. `FixedValue` cells → constant. |

For a partial-free `P1` (some entries free, others fixed) the EM update
writes the smoothed covariance into the free cells and leaves the rest
alone, then symmetrises off-diagonals defensively. This is valid when
the free pattern is consistent with positive-definiteness; pathological
patterns (e.g., a single off-diagonal free with the corresponding
diagonal fixed at zero) are the user's responsibility.

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

**Estimate the initial state** (treats `α₁` as a model parameter, works
with both `MLE()` and `EM()`):

```julia
spec = custom_ssm(...;
    a1 = [FreeParam(:μ₀, init=0.0)],
    P1 = [FreeParam(:p₀, init=1.0, lower=0.0)])

# Either path estimates μ₀ and p₀ jointly with the rest of the parameters.
model = StateSpaceModel(spec, n)
fit!(EM(), model, y)          # closed-form M-step: μ₀ → E[α₁|y], p₀ → Var[α₁|y]
# fit!(MLE(), model, y)       # LBFGS: μ₀, p₀ optimised jointly with Z/H/T/Q
parameters(model).μ₀          # estimated initial mean
parameters(model).p₀          # estimated initial variance
```

EM and MLE can give numerically different point estimates for `(μ₀, p₀)`
because EM's closed form puts `p₀` at the smoothed posterior variance
`Var[α₁ | y₁:n]`, which is generally smaller than the MLE-optimised
prior variance. Both are valid stationary points of the joint
likelihood; they're not the same problem unless you also profile `p₀`
analytically out of the MLE objective.
