module Siphon

using StaticArrays
using ForwardDiff
using LinearAlgebra
using Random: Random, AbstractRNG, randn!
using RecipesBase
using SpecialFunctions: erfcinv

include("accessors.jl")
include("types.jl")
include("missing.jl")
include("filter_ad.jl")
include("smoother_ad.jl")

# EKF extension — types + functional (AD-compatible) likelihood/filter
include("ekf_types.jl")
include("ekf_ad.jl")

# Core Kalman filter/smoother exports
export KFParms, KFParms_static
export KalmanFilterResult, SmootherResult, SmootherWorkspace
export DiffuseFilterResult  # Exact diffuse initialization
export STATIC_THRESHOLD, to_static_if_small, select_backend
export kalman_loglik
export kalman_filter
export kalman_smoother, kalman_filter_and_smooth
# Exact diffuse initialization (Durbin-Koopman method)
# Note: Use 5-arg kalman_loglik/kalman_filter for diffuse (P1_inf triggers diffuse)
export diffuse_period, diffuse_covariances, diffuse_flags

# Abstract types and the accessors shared by results, workspaces and models
export AbstractFilterResult, AbstractFilterWorkspace, AbstractStateSpaceModel
export parameters, system_matrices
export obs_matrix, obs_cov, transition_matrix, selection_matrix, state_cov
export initial_state, initial_state_cov
export predicted_states, predicted_states_cov, filtered_states, filtered_states_cov
export prediction_errors, prediction_errors_cov
export kalman_gains, loglikelihood
export smoothed_states, smoothed_states_cov, crosslag_covs

# Include DSL submodule (before predict.jl which depends on SSMSpec)
include("dsl/dsl.jl")
using .DSL

# Prediction and forecasting (depends on DSL types)
include("predict.jl")

# In-place filter/smoother for large models (depends on DSL types)
include("inplace.jl")

# In-place EKF workspace (depends on inplace.jl loading first so its
# `set_params!` / `update_params!` exist; we add EKF-typed methods on top.)
include("ekf_inplace.jl")

# Simulation smoother (depends on the in-place workspace)
include("simsmooth.jl")

# Plotting recipes (depends on filter/smoother types and inplace types)
include("recipes.jl")

# Prediction and forecasting exports
export predict, forecast, forecast_paths
export ForecastResult, forecast_periods, forecast_interval
export forecast_observations, forecast_states
export forecast_states_cov, forecast_observations_cov

# Missing data utilities
export missing_to_nan, nan_to_missing
export ismissing_period, isincomplete_period, count_missing_periods,
       count_incomplete_periods

# EKF extension exports
export AbstractEKFMeasurement, EKFParms, EKFFilterResult
export AbstractEKFJacobianMode, ADJacobian, AnalyticJacobian, FiniteDiffJacobian
export measurement, measurement!, measurement_jacobian, measurement_jacobian!
export EKFWorkspace
export predicted_observations, measurement_jacobians

# Re-export DSL components
export SSMParameter, SSMSpec, FixedValue, ParameterRef, SSMMatrixSpec
export param_names, n_params, initial_values, param_bounds
export uses_exact_diffuse
export build_linear_state_space, objective_function, ssm_loglik
export local_level, local_linear_trend, ar1, arma, dynamic_factor, dns_model
export custom_ssm, FreeParam, @P

# Matrix helpers
export diag_free, scalar_free, diag_fixed
export identity_mat, zeros_mat, ones_mat
export lower_triangular_free, symmetric_free
export selection_mat, companion_mat
export cov_free, CovFree, CovMatrixExpr
export block_diag

# Functional expressions (for DNS, Svensson, etc.)
export ParamExpr, MatrixExpr
export build_dns_loadings, build_svensson_loadings
export dns_loading1, dns_loading2

# Transformation (TransformVariables.jl integration)
export build_transformation
export transform_to_constrained, transform_to_unconstrained

# Log-density for optimization/sampling (LogDensityProblems.jl integration)
export SSMLogDensity, logdensity
export FlatPrior, NormalPrior, NormalPriorVec, InverseGammaPrior, CompositePrior

# Optimization (Optimization.jl integration)
export optimize_ssm, optimize_ssm_with_stderr

# EKF DSL surface
export EKFSpec, MeasurementExpr, custom_ekf
export build_measurement, build_ekfparms, build_nonlinear_state_space
export EKFLogDensity

# EM algorithm (DSL-based) - main API is fit!(EM(), model, y)
export EMResult, profile_em_ssm, ProfileEMResult

# In-place Kalman filter/smoother (zero-allocation for large models)
export KalmanWorkspace
export kalman_filter!, kalman_smoother!, kalman_filter_and_smooth!
export set_params!, set_initial!, update_params!
export set_companion_structure!, companion_block
export set_rank_revealing!, rank_revealing, innovation_ranks
export InnovationSupportError
# In-place diffuse filter (exact diffuse initialization)
# Note: Use kalman_filter!(ws::DiffuseKalmanWorkspace, y) - workspace type determines diffuse
export DiffuseKalmanWorkspace
export set_initial_diffuse!

# Simulation smoother (Durbin & Koopman 2002)
export SimulationSmootherWorkspace, simulation_smoother, simulation_smoother!
export refresh_factors!

# In-place EM algorithm
export EMWorkspace, compute_sufficient_stats!, em_estimate!
export MLE, EM  # Estimation method markers

# High-level StateSpaceModel API
export StateSpaceModel, fit!
export isfitted, backend

# Dynamic Factor Model
export DynamicFactorModel, DynamicFactorModelSpec
export factors, loadings, var_coefficients
export isconverged, niterations, innovation_cov, ar_coefficients, idiosyncratic_variances

# Plotting recipes (RecipesBase)
export ObservablePlot
export confidence_bands, select_vars, quantile_normal

end # module
