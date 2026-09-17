# Extended Kalman Filter (EKF) and Nonlinear MLE

Siphon.jl provides Extended-Kalman-Filter machinery for state-space models with
**linear transitions and nonlinear measurements**:

```math
\begin{aligned}
y_t      &= h(\alpha_t, t) + \varepsilon_t,        & \varepsilon_t &\sim N(0, H) \\
\alpha_{t+1} &= T \alpha_t + R \eta_t,             & \eta_t        &\sim N(0, Q)
\end{aligned}
```

The EKF linearises the measurement equation around the predicted state at each
time step:
```math
Z_t = \left.\frac{\partial h}{\partial \alpha}\right|_{\alpha = a_t}
```
and runs a Kalman filter with the time-varying ``Z_t`` in place of a constant
observation matrix.

This tutorial covers:

1. The measurement protocol (`AbstractEKFMeasurement`, `measurement` /
   `measurement!` / `measurement_jacobian!`).
2. Two Jacobian modes: `ADJacobian()` (ForwardDiff, default) and
   `AnalyticJacobian()` (user-supplied).
3. Three layers of the API: the functional `ekf_loglik` / `ekf_filter`, the
   in-place `EKFWorkspace` + `ekf_filter!` + `ekf_smoother!`, and the DSL
   `EKFSpec` + `optimize_ekf` for MLE.
4. A worked example: a shadow-rate dynamic Nelson-Siegel ("SB-DNS") yield-curve
   model fit by EKF MLE.

## The Measurement Protocol

A measurement object subtypes `AbstractEKFMeasurement` and implements at least
one of:

| Method | When required |
|---|---|
| `measurement(model, a, t) -> AbstractVector` | `ADJacobian()` mode (default), and for the pure functional path |
| `measurement!(out, model, a, t)` | `EKFWorkspace` / `ekf_filter!` (in-place) |
| `measurement_jacobian(model, a, t) -> AbstractMatrix` | optional pure analytic |
| `measurement_jacobian!(Z, model, a, t)` | `AnalyticJacobian()` mode |

The `t` argument is the time index, allowing time-varying nonlinearities
without forced allocation.

### Linear measurement (sanity-check example)

A "linear measurement" wrapped as an EKF measurement should reproduce the linear
Kalman filter exactly:

```julia
using Siphon, LinearAlgebra

struct LinearMeasurement{ZT<:AbstractMatrix} <: Siphon.AbstractEKFMeasurement
    Z::ZT
end

# Pure functional (used by AD path)
Siphon.measurement(m::LinearMeasurement, a, t) = m.Z * a

# In-place (used by EKFWorkspace)
Siphon.measurement!(out, m::LinearMeasurement, a, t) = (mul!(out, m.Z, a); out)

# Analytic Jacobian (constant, equal to Z)
Siphon.measurement_jacobian(m::LinearMeasurement, a, t)         = m.Z
Siphon.measurement_jacobian!(Z, m::LinearMeasurement, a, t)     = (copyto!(Z, m.Z); Z)
```

With this in place, `ekf_loglik` matches `kalman_loglik` to machine precision.

## Three API Layers

### 1. Functional API — `ekf_loglik`, `ekf_filter`

```julia
p = EKFParms(measurement, AnalyticJacobian(), H, T, R, Q)
ll  = ekf_loglik(p, y, a1, P1)        # log-likelihood only
res = ekf_filter(p, y, a1, P1)        # full result with at, Pt, att, Ptt, ...
```

`ekf_filter` returns an `EKFFilterResult` with the same accessors as
`KalmanFilterResult`, plus two EKF-specific fields:
* `predicted_observations(res)` — the per-step ``\hat{y}_t = h(a_t, t)``
* `measurement_jacobians(res)` — the per-step ``Z_t``

The functional path is AD-compatible: ForwardDiff propagates through
`ekf_loglik` for parameter gradients.

### 2. In-place API — `EKFWorkspace`, `ekf_filter!`, `ekf_smoother!`

For repeated calls (MLE inner loop, EM, large panels):

```julia
ws = EKFWorkspace(measurement, AnalyticJacobian(), H, T, R, Q, a1, P1, n)
ll = ekf_filter!(ws, y)
ekf_smoother!(ws; crosscov=true)
αs = smoothed_states(ws)
Vs = variances_smoothed_states(ws)
```

Allocation budget: after a warm-up call, both `ekf_filter!` and the combined
`ekf_filter_and_smooth!` allocate **O(1) bytes** in the time loop in both
Analytic and AD modes (the AD path uses a workspace-owned
`ForwardDiff.JacobianConfig` cached at construction time).

To swap parameters between calls without rebuilding the workspace:

```julia
update_params!(ws; H=H_new, Tmat=T_new, Q=Q_new)   # selective in-place update
set_initial!(ws, a1_new, P1_new)
ll_new = ekf_filter!(ws, y)
```

### 3. DSL API — `EKFSpec`, `custom_ekf`, `optimize_ekf`

`EKFSpec` is the nonlinear-measurement counterpart to `SSMSpec`. System
matrices use the same DSL helpers (`diag_free`, `cov_free`, `identity_mat`,
`MatrixExpr`); only the measurement is new.

```julia
spec = custom_ekf(
    measurement = MyMeasurement(...),               # fixed measurement object
    jacobian_mode = AnalyticJacobian(),
    H = diag_free(p, :σh; init=0.04, lower=1e-8),
    T = diag_free(m, :ψ; init=0.5, lower=-0.99, upper=0.99),
    R = identity_mat(m),
    Q = diag_free(m, :σq; init=0.01, lower=1e-8),
    a1 = zeros(m),
    P1 = 10.0 .* Matrix(I, m, m),
)
res = optimize_ekf(spec, y; method = Optim.LBFGS())
```

When the measurement itself depends on free parameters (e.g. a smoothing
scale, a kink location, or a state mean stored inside the measurement object),
use a `MeasurementExpr`:

```julia
mexpr = MeasurementExpr(
    params  = [SSMParameter(:γ; init=1.0, lower=0.05)],
    builder = (θ, data) -> MyMeasurement(data.Λ, θ[:γ]),
    data    = (Λ = Λ_loadings,),
)
spec = custom_ekf(measurement = mexpr, ...)
```

`optimize_ekf` runs in unconstrained ℝⁿ via TransformVariables (variances
through `asℝ₊`, bounded reals through `as(Real, lo, hi)`, etc.), so you only
declare bounds on the parameters and the optimizer handles the rest.

## Worked Example — Shadow-Rate DNS

Opschoor & van der Wel (2024) propose four "smooth shadow-rate" dynamic
Nelson-Siegel variants whose measurement equations have the common form

```math
y_t(\tau_i) = m\bigl( \Lambda_i^{\top} \beta_t \, ;\, \gamma, r_{LB}\bigr) + \varepsilon_{t,i},
```

with state law ``\beta_t = \mu + \Psi (\beta_{t-1} - \mu) + \eta_t`` and the
softplus variant (SB-S) using
``m(s; r_{LB}) = r_{LB} + \log(1 + e^{s - r_{LB}})``.
The state ``\beta_t \in \mathbb{R}^3`` is the level/slope/curvature triple; the
loadings ``\Lambda \in \mathbb{R}^{N \times 3}`` are the fixed Nelson-Siegel
loadings.

To match Siphon's intercept-free transition ``\alpha_{t+1} = T \alpha_t + R \eta_t``,
we **store the state in deviations** ``\xi_t = \beta_t - \mu`` and bake ``\mu``
into the measurement object. Then ``\xi_{t+1} = \Psi \xi_t + \eta_t``.

```julia
using Siphon, LinearAlgebra

# Stable softplus / logistic without LogExpFunctions
@inline _softplus(x) = x >= 0 ? x + log1p(exp(-x)) : log1p(exp(x))
@inline _logistic(x) = x >= 0 ? inv(one(x) + exp(-x)) : exp(x) / (one(x) + exp(x))

# Two type parameters so AD can promote μ without dragging r_LB along
struct SBSCenteredMeasurement{LT<:AbstractMatrix, T1<:Real, T2<:Real} <:
        Siphon.AbstractEKFMeasurement
    Λ::LT
    r_LB::T1
    μ::Vector{T2}
end

# Pure functional
function Siphon.measurement(m::SBSCenteredMeasurement, ξ, t)
    sh = m.Λ * (ξ .+ m.μ)
    return m.r_LB .+ _softplus.(sh .- m.r_LB)
end

# In-place: hand-rolled inner product avoids allocating sh
function Siphon.measurement!(out, m::SBSCenteredMeasurement, ξ, t)
    @inbounds for i in axes(m.Λ, 1)
        s = zero(eltype(out))
        for k in axes(m.Λ, 2); s += m.Λ[i, k] * (ξ[k] + m.μ[k]); end
        out[i] = m.r_LB + _softplus(s - m.r_LB)
    end
    return out
end

# Analytic Jacobian: ∂y_i/∂ξ_j = logistic(s_i - r_LB) * Λ_{i,j}
function Siphon.measurement_jacobian!(Z, m::SBSCenteredMeasurement, ξ, t)
    @inbounds for i in axes(m.Λ, 1)
        s = zero(eltype(Z))
        for k in axes(m.Λ, 2); s += m.Λ[i, k] * (ξ[k] + m.μ[k]); end
        d = _logistic(s - m.r_LB)
        for j in axes(m.Λ, 2); Z[i, j] = d * m.Λ[i, j]; end
    end
    return Z
end
```

### Simulating data and fitting via `optimize_ekf`

```julia
using Random
Random.seed!(0)

# Nelson-Siegel loadings at λ = 0.069
λ = 0.069
maturities = [3.0, 6.0, 12.0, 24.0, 36.0, 60.0, 84.0, 120.0]
Λ = hcat(
    ones(length(maturities)),
    (1 .- exp.(-λ .* maturities)) ./ (λ .* maturities),
    (1 .- exp.(-λ .* maturities)) ./ (λ .* maturities) .- exp.(-λ .* maturities),
)

m_state, p_obs, n = 3, length(maturities), 250
μ_true = [3.5, -1.0, 0.5]
Ψ_true = Matrix(Diagonal([0.97, 0.95, 0.90]))
Q_true = Matrix(Diagonal([0.10, 0.15, 0.20] .^ 2))
H_true = (0.05^2) .* Matrix(I, p_obs, p_obs)

α_true = zeros(m_state, n);   y = zeros(p_obs, n)
α_true[:, 1] = μ_true .+ randn(m_state)
for t in 1:n
    if t > 1
        α_true[:, t] = μ_true + Ψ_true * (α_true[:, t-1] - μ_true) +
                       cholesky(Q_true).L * randn(m_state)
    end
    ξ = α_true[:, t] .- μ_true
    yhat = Siphon.measurement(SBSCenteredMeasurement(Λ, 0.0, μ_true), ξ, t)
    y[:, t] = yhat .+ sqrt.(diag(H_true)) .* randn(p_obs)
end

# MeasurementExpr: μ as free parameters, builder reconstructs the measurement
mexpr = MeasurementExpr(
    params  = [SSMParameter(:μ_L; init=3.5),
               SSMParameter(:μ_S; init=-1.0),
               SSMParameter(:μ_C; init=0.5)],
    builder = (θ, data) -> SBSCenteredMeasurement(data.Λ, data.r_LB,
                                                  [θ[:μ_L], θ[:μ_S], θ[:μ_C]]),
    data    = (Λ = Λ, r_LB = 0.0),
)

spec = custom_ekf(
    measurement   = mexpr,
    jacobian_mode = AnalyticJacobian(),
    H  = diag_free(p_obs,  :σh2; init=0.05^2, lower=1e-10),
    T  = diag_free(m_state, :ψ;  init=0.9,    lower=-0.999, upper=0.999),
    R  = identity_mat(m_state),
    Q  = diag_free(m_state, :σq2; init=0.04,  lower=1e-10),
    a1 = zeros(m_state),
    P1 = 1.0 .* Matrix(I, m_state, m_state),
)

result = optimize_ekf(spec, y; maxiters = 400)
result.converged    # → true
result.loglik       # → ≈ 2312
result.θ.μ_L        # ≈ 3.5
result.θ.ψ_1        # ≈ 0.97
```

The fitted spec has 17 free parameters (8 ``\sigma_h^2`` + 3 ``\psi`` +
3 ``\sigma_q^2`` + 3 ``\mu``). One-shot AD-driven L-BFGS converges to the
neighbourhood of the true parameter values from the warm-start initial values.

### Filtered and smoothed state paths

After fitting, build a workspace at the optimum to obtain ``\beta_t`` paths and
their smoothed counterparts:

```julia
# Build EKFParms at the optimum
θ̂ = result.θ
p̂ = build_ekfparms(spec, θ̂)
a1, P1 = build_initial_state(spec, θ̂)

ws = EKFWorkspace(p̂.measurement, p̂.jacobian_mode, p̂.H, p̂.T, p̂.R, p̂.Q,
                  a1, P1, n)
ekf_filter_and_smooth!(ws, y; crosscov = true)

αs = smoothed_states(ws)              # smoothed state in deviation form
μ̂ = [θ̂.μ_L, θ̂.μ_S, θ̂.μ_C]
β_smoothed = αs .+ μ̂                  # smoothed state in original β-coordinates
```

Smoothed RMSE on each component is typically 10-20% lower than filtered RMSE,
and `Vs = variances_smoothed_states(ws)` is positive semidefinite throughout.

## Choosing Between Jacobian Modes

* **`ADJacobian()`** is the default. It only requires the pure functional
  `measurement(model, a, t)` and uses ForwardDiff for the per-step Jacobian.
  In `EKFWorkspace`, the AD config is cached at construction so the time loop
  remains allocation-free. Use this whenever the measurement is smooth and
  AD-friendly.

* **`AnalyticJacobian()`** is for nonsmooth measurements (e.g. B-DNS kinks,
  truncated observations) and for cases where the analytic Jacobian is very
  cheap relative to AD. Implement `measurement!` and `measurement_jacobian!`.

* **`FiniteDiffJacobian()`** is diagnostic only.

## Performance Notes

* The functional `ekf_loglik` is intended for AD parameter gradients; it
  allocates per call.
* For repeated filtering (MLE inner loop, EM, simulation), use
  `EKFWorkspace` + `ekf_filter!`. With allocation-free user methods this is
  the same wall-time as the linear `kalman_filter!` for the same dimensions.
* The state Jacobian is recomputed at every time step. Storage of
  ``Z_t`` per step is unavoidable when using `ekf_smoother!`; for filter-only
  workloads, the workspace stores ``Z_t`` regardless (used by the cross-lag
  smoother step).
* Always pre-build the `EKFWorkspace` and the measurement object outside any
  optimizer closure; mutate `μ`-like fields in place when needed.
