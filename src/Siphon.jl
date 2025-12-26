module Siphon

using StaticArrays
using ForwardDiff
using LinearAlgebra
using RecipesBase

include("types.jl")
include("filter_ad.jl")
include("smoother_ad.jl")

# Core Kalman filter/smoother exports
export KFParms, KFParms_static
export KalmanFilterResult, KalmanFilterResultScalar, SmootherWorkspace
export DiffuseFilterResult  # Exact diffuse initialization
export STATIC_THRESHOLD, to_static_if_small, select_backend
export kalman_loglik, kalman_loglik_scalar, kalman_loglik_static
export kalman_filter, kalman_filter_scalar, kalman_filter_static
export kalman_smoother, kalman_smoother_scalar, kalman_filter_and_smooth
# Exact diffuse initialization (Durbin-Koopman method)
# Note: Use 5-arg kalman_loglik/kalman_filter for diffuse (P1_inf triggers diffuse)
export diffuse_period, diffuse_covariances, diffuse_flags

# Accessor methods for KalmanFilterResult
export parameters
export obs_matrix, obs_cov, transition_matrix, selection_matrix, state_cov
export predicted_states, filtered_states
export variances_predicted_states, variances_filtered_states
export prediction_errors, variances_prediction_errors
export kalman_gains, loglikelihood
export smoothed_states, variances_smoothed_states

# Include DSL submodule (before predict.jl which depends on SSMSpec)
include("dsl/dsl.jl")
using .DSL

# Prediction and forecasting (depends on DSL types)
include("predict.jl")

# In-place filter/smoother for large models (depends on DSL types)
include("inplace.jl")

# Plotting recipes (depends on filter/smoother types and inplace types)
include("recipes.jl")

# Prediction and forecasting exports
export predict, forecast, forecast_paths

# Missing data utilities
export missing_to_nan, nan_to_missing, count_missing, ismissing_obs

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

# EM algorithm (DSL-based) - main API is fit!(EM(), model, y)
export EMResult, profile_em_ssm, ProfileEMResult

# In-place Kalman filter/smoother (zero-allocation for large models)
export KalmanWorkspace
export kalman_filter!, kalman_smoother!, filter_and_smooth!
export set_params!, set_initial!, update_params!
# In-place diffuse filter (exact diffuse initialization)
# Note: Use kalman_filter!(ws::DiffuseKalmanWorkspace, y) - workspace type determines diffuse
export DiffuseKalmanWorkspace
export set_initial_diffuse!

# In-place EM algorithm
export EMWorkspace, compute_sufficient_stats!, em_estimate!
export MLE, EM  # Estimation method markers

# High-level StateSpaceModel API
export StateSpaceModel, fit!, system_matrices

# Dynamic Factor Model
export DynamicFactorModel, DynamicFactorModelSpec
export factors, loadings, var_coefficients
export isconverged, niterations, innovation_cov, ar_coefficients, idiosyncratic_variances

# Plotting recipes (RecipesBase)
export SmootherResult, ForecastResult, ObservablePlot
export confidence_bands, select_vars, quantile_normal
export forecast_observations, forecast_states
export variances_forecast_states, variances_forecast_observations

end # module
