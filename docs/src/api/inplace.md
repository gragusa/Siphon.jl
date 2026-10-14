# In-Place Workspaces

The workspace types hold preallocated storage for the filter and smoother, so
repeated runs (inside an optimizer, an EM loop or a Gibbs sampler) do not
allocate. `StateSpaceModel` uses them internally; this page documents them for
direct use.

## Workspaces

A `KalmanWorkspace` built with `covariances = false` stores no per-period state
covariances. Use it for a large state when only the smoothed means, the
disturbance smoother or simulation-smoother draws are needed: at `m = 255` and
`n = 300` it takes about 18 MB instead of about 610 MB.

```@docs
KalmanWorkspace
DiffuseKalmanWorkspace
```

## Filtering and Smoothing

```@docs
kalman_filter!
kalman_smoother!
kalman_filter_and_smooth!
```

## Updating Parameters and the Initial State

```@docs
set_params!
update_params!
set_initial!
set_initial_diffuse!
Siphon.set_Z!
Siphon.set_H!
Siphon.set_T!
Siphon.set_R!
Siphon.set_Q!
```

## Dimensions

```@docs
Siphon.obs_dim
Siphon.state_dim
Siphon.shock_dim
Siphon.n_times
```

## Missing Observations

```@docs
Siphon.missing_mask
Siphon.observed_mask
Siphon.observed_rows
Siphon.n_observed
Siphon.n_valid_obs
```

## Companion Form and Singular Innovation Covariances

```@docs
set_companion_structure!
companion_block
set_rank_revealing!
rank_revealing
innovation_ranks
InnovationSupportError
Siphon.cholesky_factors
```

## Transition Intercept and Period Scales

```@docs
Siphon.set_transition_intercept!
Siphon.transition_intercept
Siphon.set_period_scales!
Siphon.period_scales
```

## Simulation Smoother

```@docs
simulation_smoother
simulation_smoother!
SimulationSmootherWorkspace
refresh_factors!
Siphon.PSDFactor
Siphon.psd_factor
Siphon.psd_rand!
```
