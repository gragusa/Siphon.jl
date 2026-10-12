# ============================================
# Abstract types and the shared accessor vocabulary
# ============================================

"""
    AbstractFilterResult

Supertype of the values returned by the functional filters: `KalmanFilterResult`,
`DiffuseFilterResult` and `EKFFilterResult`.

Every subtype supports the filter accessors `predicted_states`, `predicted_states_cov`,
`filtered_states`, `filtered_states_cov`, `prediction_errors`, `prediction_errors_cov`,
`kalman_gains`, `loglikelihood`, `missing_mask`, `system_matrices`, `initial_state` and
`initial_state_cov`. The matrix accessors `obs_matrix`, `obs_cov`, `transition_matrix`,
`selection_matrix` and `state_cov` are available where the result stores the system.

A filter stops at the first period whose innovation covariance is not positive definite.
It then returns a result whose log-likelihood is `-Inf`. The state and innovation
accessors of such a result throw an `ArgumentError` that names the period.
"""
abstract type AbstractFilterResult end

"""
    AbstractFilterWorkspace

Supertype of the pre-allocated workspaces of the in-place filters: `KalmanWorkspace`,
`DiffuseKalmanWorkspace` and `EKFWorkspace`.

Workspaces support the same accessors as [`AbstractFilterResult`](@ref), plus the
smoother accessors `smoothed_states`, `smoothed_states_cov` and `crosslag_covs`, which
hold values after `kalman_smoother!` has run.

The accessors return the workspace's own storage, not copies: the next run overwrites them.
The exceptions are `prediction_errors`, `prediction_errors_cov` and `kalman_gains`, which
return new arrays in series positions.

Before the first filter run, and after a run that stopped on an innovation covariance
that was not positive definite, the accessors throw an `ArgumentError`. After a failed
run `loglikelihood` returns `-Inf`. The smoother accessors also throw until the smoother
has run on the current filter output.
"""
abstract type AbstractFilterWorkspace end

const _FilterOutput = Union{AbstractFilterResult, AbstractFilterWorkspace}

# The object whose fields `at`, `Pt`, `att`, `Ptt`, `vt`, `Ft`, `Kt`, `loglik`,
# `missing_mask` hold the filter output. Workspaces that wrap another one override it.
_filter_store(x::_FilterOutput) = x

# Run status of a filter output: 0 after a successful run, t > 0 when the run stopped
# in period t because an innovation covariance was not positive definite, and
# `_NOT_RUN` for a workspace that no filter has written to yet.
const _NOT_RUN = -1
_run_status(r::AbstractFilterResult) = r.failed_at
_run_status(ws::AbstractFilterWorkspace) = _filter_store(ws).status

function _failure_message(status::Integer)
    return "the filter stopped in period $status because the innovation covariance " *
           "was not positive definite, so the log-likelihood is -Inf and the filter " *
           "output is not available"
end

# Throw unless `x` holds the output of a successful filter run.
function _require_output(x::_FilterOutput, f::Symbol)
    status = _run_status(x)
    status == 0 && return x
    status == _NOT_RUN && throw(ArgumentError(
        "$f: the workspace holds no filter output. Call kalman_filter!(ws, y) first."))
    throw(ArgumentError("$f: $(_failure_message(status))."))
end

# The `p × n` mask of observed values. The EKF types override it.
_observed_mask(x::_FilterOutput) = _filter_store(x).observed_mask

# The filters store the innovations, their covariances and the gains of a partially
# observed period in the leading rows and columns of the period's slice. These spread
# them back to series positions: NaN (zero for the gains) where a value was not observed.
function _unpack_innovations(vt::AbstractMatrix, mask::AbstractMatrix{Bool})
    out = fill!(similar(vt), NaN)
    for t in axes(vt, 2)
        k = 0
        for i in axes(vt, 1)
            mask[i, t] || continue
            k += 1
            out[i, t] = vt[k, t]
        end
    end
    return out
end

function _unpack_innovation_covs(Ft::AbstractArray{<:Any, 3}, mask::AbstractMatrix{Bool})
    out = fill!(similar(Ft), NaN)
    for t in axes(Ft, 3)
        rows = findall(view(mask, :, t))
        for (l, j) in pairs(rows), (k, i) in pairs(rows)
            out[i, j, t] = Ft[k, l, t]
        end
    end
    return out
end

function _unpack_gains(Kt::AbstractArray{<:Any, 3}, mask::AbstractMatrix{Bool})
    out = fill!(similar(Kt), 0)
    for t in axes(Kt, 3)
        k = 0
        for j in axes(Kt, 2)
            mask[j, t] || continue
            k += 1
            out[:, j, t] .= view(Kt, :, k, t)
        end
    end
    return out
end

# The storage of `x`, after checking that it holds the output of a successful run.
_output(x::_FilterOutput, f::Symbol) = _filter_store(_require_output(x, f))

# Throw unless the smoother has run on the current filter output of `ws`.
function _require_smoothed(ws::AbstractFilterWorkspace, f::Symbol)
    _require_output(ws, f)
    _filter_store(ws).smoothed || throw(ArgumentError(
        "$f: the smoother has not run on this filter output. " *
        "Call kalman_smoother!(ws) first."))
    return ws
end

"""
    predicted_states(x) -> Matrix

Return the predicted state means E[αₜ | y₁:ₜ₋₁] for t = 1:n (m × n).

`x` is a filter result, a filter workspace or a fitted model.
"""
predicted_states(x::_FilterOutput) = _output(x, :predicted_states).at

"""
    predicted_states_cov(x) -> Array{T,3}

Return the predicted state covariances Var[αₜ | y₁:ₜ₋₁] for t = 1:n (m × m × n).

`x` is a filter result, a filter workspace or a fitted model.
"""
predicted_states_cov(x::_FilterOutput) = _output(x, :predicted_states_cov).Pt

"""
    filtered_states(x) -> Matrix

Return the filtered state means E[αₜ | y₁:ₜ] for t = 1:n (m × n).

`x` is a filter result, a filter workspace or a fitted model.
"""
filtered_states(x::_FilterOutput) = _output(x, :filtered_states).att

"""
    filtered_states_cov(x) -> Array{T,3}

Return the filtered state covariances Var[αₜ | y₁:ₜ] for t = 1:n (m × m × n).

`x` is a filter result, a filter workspace or a fitted model.
"""
filtered_states_cov(x::_FilterOutput) = _output(x, :filtered_states_cov).Ptt

"""
    prediction_errors(x) -> Matrix

Return the prediction errors (innovations) vₜ = yₜ − E[yₜ | y₁:ₜ₋₁] for t = 1:n (p × n).
Row `i` of column `t` belongs to series `i`; it is NaN when that value was not observed.

The result is a new array, not the workspace's storage.

`x` is a filter result, a filter workspace or a fitted model.
"""
function prediction_errors(x::_FilterOutput)
    return _unpack_innovations(_output(x, :prediction_errors).vt, _observed_mask(x))
end

"""
    prediction_errors_cov(x) -> Array{T,3}

Return the innovation covariances Var[yₜ | y₁:ₜ₋₁] for t = 1:n (p × p × n), indexed by
series as in [`prediction_errors`](@ref). Rows and columns of series not observed in
period `t` are NaN.

The result is a new array, not the workspace's storage.

`x` is a filter result, a filter workspace or a fitted model.
"""
function prediction_errors_cov(x::_FilterOutput)
    return _unpack_innovation_covs(_output(x, :prediction_errors_cov).Ft, _observed_mask(x))
end

"""
    kalman_gains(x) -> Array{T,3}

Return the Kalman gains Kₜ for t = 1:n (m × p × n). Column `i` belongs to series `i`; it is
zero when that value was not observed.

The result is a new array, not the workspace's storage.

`x` is a filter result, a filter workspace or a fitted model.
"""
kalman_gains(x::_FilterOutput) = _unpack_gains(_output(x, :kalman_gains).Kt, _observed_mask(x))

"""
    loglikelihood(x) -> Real

Return the log-likelihood of the observed data. It is `-Inf` when the filter stopped on
an innovation covariance that was not positive definite.

`x` is a filter result, a filter workspace or a fitted model.
"""
function loglikelihood(x::_FilterOutput)
    _run_status(x) == _NOT_RUN && _require_output(x, :loglikelihood)
    return _filter_store(x).loglik
end

"""
    missing_mask(x) -> BitVector

Return the mask that is true for each period in which no row was observed. For
partially observed periods, see `observed_mask`.

`x` is a filter result, a filter workspace or a fitted model.
"""
missing_mask(x::_FilterOutput) = _output(x, :missing_mask).missing_mask

"""
    smoothed_states(x) -> Matrix

Return the smoothed state means E[αₜ | y₁:ₙ] for t = 1:n (m × n).

On a workspace this needs a prior `kalman_smoother!` run. On a fitted model the smoother
runs on first use and its output is cached.
"""
smoothed_states(ws::AbstractFilterWorkspace) =
    _filter_store(_require_smoothed(ws, :smoothed_states)).αs

"""
    smoothed_states_cov(x) -> Array{T,3}

Return the smoothed state covariances Var[αₜ | y₁:ₙ] for t = 1:n (m × m × n).

On a workspace this needs a prior `kalman_smoother!` run. On a fitted model the smoother
runs on first use and its output is cached.
"""
smoothed_states_cov(ws::AbstractFilterWorkspace) =
    _filter_store(_require_smoothed(ws, :smoothed_states_cov)).Vs

"""
    crosslag_covs(ws) -> Array{T,3}

Return the smoothed cross-lag covariances Cov[αₜ₊₁, αₜ | y₁:ₙ] for t = 1:n−1
(m × m × (n−1)). Needs a prior `kalman_smoother!(ws; crosscov = true)` run.
"""
function crosslag_covs(ws::AbstractFilterWorkspace)
    store = _filter_store(_require_smoothed(ws, :crosslag_covs))
    store.crosscov || throw(ArgumentError(
        "crosslag_covs: the last smoother run skipped the cross-lag covariances. " *
        "Call kalman_smoother!(ws; crosscov = true)."))
    return store.Pcross
end

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
