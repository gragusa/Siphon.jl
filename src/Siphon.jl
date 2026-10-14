module Siphon

using StaticArrays
using ForwardDiff
using LinearAlgebra
using LogDensityProblems: LogDensityProblems, logdensity
using Optimization: Optimization, OptimizationFunction, OptimizationProblem
using OptimizationOptimJL: Optim
using Random: Random, AbstractRNG, randn!
using RecipesBase
using SpecialFunctions: erfcinv, loggamma
using Statistics: mean, var
using StatsAPI: StatsAPI, StatisticalModel, stderror, aic, aicc, bic, coeftable
import StatsAPI: fit!, loglikelihood, predict, isfitted, nobs, coef, coefnames, vcov, dof
using TransformVariables
using TransformVariables: corr_cholesky_factor

include("accessors.jl")
include("types.jl")
include("missing.jl")
include("filter_ad.jl")
include("smoother_ad.jl")
include("diffuse_smoother.jl")

# EKF extension — types + functional (AD-compatible) likelihood/filter
include("ekf_types.jl")
include("ekf_ad.jl")

# Core Kalman filter/smoother exports
export KFParms
export KalmanFilterResult, SmootherResult
export DiffuseFilterResult  # Exact diffuse initialization
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
export DisturbanceSmootherResult, disturbance_smoother, disturbance_smoother!
export smoothed_obs_disturbances, smoothed_obs_disturbances_cov
export smoothed_state_disturbances, smoothed_state_disturbances_cov, auxiliary_residuals

# Model specification: parameters, matrix specs, templates, estimation
include("dsl/types.jl")
include("dsl/expressions.jl")
include("dsl/codegen.jl")
include("dsl/templates.jl")
include("dsl/builder.jl")
include("dsl/matrix_helpers.jl")
include("dsl/bayesian.jl")
include("dsl/optimization.jl")

# EKF model specification (builds on the specification layer above)
include("ekf_spec.jl")
include("ekf_codegen.jl")
include("ekf_optim.jl")

# Prediction and forecasting
include("predict.jl")

# In-place filter/smoother for large models
include("inplace.jl")
include("em_spec.jl")

# Disturbance smoother (functional, in-place and model methods)
include("disturbance_smoother.jl")

# In-place EKF workspace (depends on inplace.jl loading first so its
# `set_params!` / `update_params!` exist; we add EKF-typed methods on top.)
include("ekf_inplace.jl")

# Simulation smoother (depends on the in-place workspace)
include("simsmooth.jl")

# Plotting recipes (depends on filter/smoother types and inplace types)
include("recipes.jl")
include("show.jl")

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
export AbstractEKFMeasurement, LinearMeasurement, EKFParms, EKFFilterResult
export AbstractEKFJacobianMode, ADJacobian, AnalyticJacobian, FiniteDiffJacobian
export measurement, measurement!, measurement_jacobian, measurement_jacobian!
export EKFWorkspace
export predicted_observations, measurement_jacobians

# Model specification
export AbstractSSMSpec, SSMParameter, SSMSpec, FixedValue, ParameterRef, SSMMatrixSpec
export param_names, n_params, initial_values, param_bounds
export uses_exact_diffuse
export build_linear_state_space, objective_function, ssm_loglik
export local_level, local_linear_trend, ar1, arma, dynamic_factor, dns_model
export custom_ssm, FreeParam

# Matrix helpers
export diag_free, scalar_free
export lower_triangular_free, symmetric_free
export selection_mat, companion_free
export cov_free, CovFree, CovMatrixExpr
export block_diag

# Functional expressions (for DNS, Svensson, etc.)
export MatrixExpr

# Transformation (TransformVariables.jl integration)
export build_transformation
export transform_to_constrained, transform_to_unconstrained

# Log-density for optimization/sampling (LogDensityProblems.jl integration)
export SSMLogDensity, logdensity
export AbstractPrior, FlatPrior, ParameterPriors, NormalPrior, InverseGammaPrior,
       CompositePrior

# Optimization (Optimization.jl integration)
export optimize_ssm, optimize_ssm_with_stderr

# EKF model specification
export EKFSpec, MeasurementExpr, custom_ekf
export build_measurement, build_ekfparms, build_nonlinear_state_space

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

# EM algorithm: fit!(EM(), model, y) and profile_em_ssm. The workspace layer
# (EMWorkspace, em_estimate!, compute_sufficient_stats!, set_em_masks!) is
# reached as Siphon.<name>.
export EMResult, profile_em_ssm, ProfileEMResult
export MLE, EM  # Estimation method markers

# High-level StateSpaceModel API
export StateSpaceModel, fit!
export isfitted

# StatsAPI generics with methods for fitted models (the same bindings StatsBase exports)
export nobs, coef, coefnames, vcov, stderror, dof, aic, aicc, bic, coeftable

# Dynamic Factor Model
export DynamicFactorModel
export factors, factors_cov, filtered_factors, filtered_factors_cov
export loadings, var_coefficients
export isconverged, niterations, factor_innovation_cov, ar_coefficients,
       idiosyncratic_variances

# Plotting recipes (RecipesBase)
export ObservablePlot

# Public but not exported: reached as `Siphon.name`
# Missing-data masks and the dimensions of in-place workspaces
public missing_mask, observed_mask, observed_rows, n_observed, n_valid_obs
public cholesky_factors
public obs_dim, state_dim, shock_dim, n_times
public set_Z!, set_H!, set_T!, set_R!, set_Q!
public set_transition_intercept!, transition_intercept, set_period_scales!, period_scales
# StaticArrays threshold and confidence bands
public STATIC_THRESHOLD, confidence_bands
# Factorization and draws from a positive semi-definite covariance
public PSDFactor, psd_factor, psd_rand!
# Specification internals: system matrices and parameter positions
public build_kfparms, build_initial_state, param_index, AbstractMatrixExpr
# EM workspace layer behind fit!(EM(), …)
public EMWorkspace, em_estimate!, compute_sufficient_stats!, set_em_masks!
# Dynamic factor model specification (`model.spec`)
public DynamicFactorModelSpec
# Dynamic Nelson-Siegel and Svensson loadings
public build_dns_loadings, build_svensson_loadings, dns_loading1, dns_loading2

end # module
