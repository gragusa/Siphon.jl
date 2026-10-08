# ============================================
# Abstract types and the shared accessor vocabulary
# ============================================

"""
    AbstractFilterResult

Supertype of the values returned by the functional filters: `KalmanFilterResult`,
`KalmanFilterResultScalar`, `DiffuseFilterResult` and `EKFFilterResult`.

Every subtype supports the filter accessors `predicted_states`, `predicted_states_cov`,
`filtered_states`, `filtered_states_cov`, `prediction_errors`, `prediction_errors_cov`,
`kalman_gains`, `loglikelihood`, `missing_mask`, `system_matrices`, `initial_state` and
`initial_state_cov`. The matrix accessors `obs_matrix`, `obs_cov`, `transition_matrix`,
`selection_matrix` and `state_cov` are available where the result stores the system.
"""
abstract type AbstractFilterResult end

"""
    AbstractFilterWorkspace

Supertype of the pre-allocated workspaces of the in-place filters: `KalmanWorkspace`,
`DiffuseKalmanWorkspace` and `EKFWorkspace`.

Workspaces support the same accessors as [`AbstractFilterResult`](@ref), plus the
smoother accessors `smoothed_states`, `smoothed_states_cov` and `crosslag_covs`, which
hold values after `kalman_smoother!` (or `ekf_smoother!`) has run.

The accessors return the workspace's own storage, not copies: the next run overwrites them.
"""
abstract type AbstractFilterWorkspace end

const _FilterOutput = Union{AbstractFilterResult, AbstractFilterWorkspace}

# The object whose fields `at`, `Pt`, `att`, `Ptt`, `vt`, `Ft`, `Kt`, `loglik`,
# `missing_mask` hold the filter output. Workspaces that wrap another one override it.
_filter_store(x::_FilterOutput) = x

"""
    predicted_states(x) -> Matrix

Return the predicted state means E[αₜ | y₁:ₜ₋₁] for t = 1:n (m × n).

`x` is a filter result, a filter workspace or a fitted model.
"""
predicted_states(x::_FilterOutput) = _filter_store(x).at

"""
    predicted_states_cov(x) -> Array{T,3}

Return the predicted state covariances Var[αₜ | y₁:ₜ₋₁] for t = 1:n (m × m × n).

`x` is a filter result, a filter workspace or a fitted model.
"""
predicted_states_cov(x::_FilterOutput) = _filter_store(x).Pt

"""
    filtered_states(x) -> Matrix

Return the filtered state means E[αₜ | y₁:ₜ] for t = 1:n (m × n).

`x` is a filter result, a filter workspace or a fitted model.
"""
filtered_states(x::_FilterOutput) = _filter_store(x).att

"""
    filtered_states_cov(x) -> Array{T,3}

Return the filtered state covariances Var[αₜ | y₁:ₜ] for t = 1:n (m × m × n).

`x` is a filter result, a filter workspace or a fitted model.
"""
filtered_states_cov(x::_FilterOutput) = _filter_store(x).Ptt

"""
    prediction_errors(x) -> Matrix

Return the prediction errors (innovations) vₜ = yₜ − E[yₜ | y₁:ₜ₋₁] for t = 1:n (p × n).
Periods with no observed row hold NaN. In a partially observed period the innovations of
the observed rows are packed into the leading rows; see `observed_rows`.

`x` is a filter result, a filter workspace or a fitted model.
"""
prediction_errors(x::_FilterOutput) = _filter_store(x).vt

"""
    prediction_errors_cov(x) -> Array{T,3}

Return the innovation covariances Var[yₜ | y₁:ₜ₋₁] for t = 1:n (p × p × n). Partially
observed periods are packed into the leading block, as in [`prediction_errors`](@ref).

`x` is a filter result, a filter workspace or a fitted model.
"""
prediction_errors_cov(x::_FilterOutput) = _filter_store(x).Ft

"""
    kalman_gains(x) -> Array{T,3}

Return the Kalman gains Kₜ for t = 1:n (m × p × n). Zero for periods with no observed row.

`x` is a filter result, a filter workspace or a fitted model.
"""
kalman_gains(x::_FilterOutput) = _filter_store(x).Kt

"""
    loglikelihood(x) -> Real

Return the log-likelihood of the observed data.

`x` is a filter result, a filter workspace or a fitted model.
"""
loglikelihood(x::_FilterOutput) = _filter_store(x).loglik

"""
    missing_mask(x) -> BitVector

Return the mask that is true for each period in which no row was observed. For
partially observed periods, see `observed_mask`.

`x` is a filter result, a filter workspace or a fitted model.
"""
missing_mask(x::_FilterOutput) = _filter_store(x).missing_mask

"""
    smoothed_states(x) -> Matrix

Return the smoothed state means E[αₜ | y₁:ₙ] for t = 1:n (m × n).

On a workspace this needs a prior `kalman_smoother!` run. On a fitted model the smoother
runs on first use and its output is cached.
"""
smoothed_states(ws::AbstractFilterWorkspace) = _filter_store(ws).αs

"""
    smoothed_states_cov(x) -> Array{T,3}

Return the smoothed state covariances Var[αₜ | y₁:ₙ] for t = 1:n (m × m × n).

On a workspace this needs a prior `kalman_smoother!` run. On a fitted model the smoother
runs on first use and its output is cached.
"""
smoothed_states_cov(ws::AbstractFilterWorkspace) = _filter_store(ws).Vs

"""
    crosslag_covs(ws) -> Array{T,3}

Return the smoothed cross-lag covariances Cov[αₜ₊₁, αₜ | y₁:ₙ] for t = 1:n−1
(m × m × (n−1)). Needs a prior `kalman_smoother!(ws; crosscov = true)` run.
"""
crosslag_covs(ws::AbstractFilterWorkspace) = _filter_store(ws).Pcross

"""
    system_matrices(x) -> KFParms

Return the system matrices of `x` as a `KFParms` with fields `Z`, `H`, `T`, `R` and `Q`,
which can be passed straight to `kalman_filter` and the other functional entry points.
For the extended Kalman filter it is the `EKFParms`, which holds the measurement model in
place of `Z`.

`x` is a filter result, a filter workspace or a fitted model. On a model the matrices are
built at the fitted parameters.
"""
system_matrices(r::AbstractFilterResult) = r.p

"""
    obs_matrix(x)

Return the observation matrix Z (p × m) of a filter result, workspace or fitted model.
"""
obs_matrix(x) = system_matrices(x).Z

"""
    obs_cov(x)

Return the observation noise covariance H (p × p) of a filter result, workspace or
fitted model.
"""
obs_cov(x) = system_matrices(x).H

"""
    transition_matrix(x)

Return the state transition matrix T (m × m) of a filter result, workspace or fitted
model.
"""
transition_matrix(x) = system_matrices(x).T

"""
    selection_matrix(x)

Return the state noise selection matrix R (m × r) of a filter result, workspace or
fitted model.
"""
selection_matrix(x) = system_matrices(x).R

"""
    state_cov(x)

Return the state noise covariance Q (r × r) of a filter result, workspace or fitted
model.
"""
state_cov(x) = system_matrices(x).Q

"""
    initial_state(x) -> Vector

Return the initial state mean a₁ used by the filter.

`x` is a filter result, a filter workspace or a fitted model. A filter result reads it
from the predicted state of period 1.
"""
initial_state(r::AbstractFilterResult) = r.at[:, 1]
initial_state(ws::AbstractFilterWorkspace) = _filter_store(ws).a1

"""
    initial_state_cov(x) -> Matrix

Return the initial state covariance P₁ used by the filter. Under exact diffuse
initialisation this is the finite part P₁*; the diffuse part is returned by
`diffuse_covariances` (results) or held by the workspace.

`x` is a filter result, a filter workspace or a fitted model. A filter result reads it
from the predicted covariance of period 1.
"""
initial_state_cov(r::AbstractFilterResult) = r.Pt[:, :, 1]
initial_state_cov(ws::AbstractFilterWorkspace) = _filter_store(ws).P1
