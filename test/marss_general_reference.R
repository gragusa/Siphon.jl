# MARSS General Reference Script for EM Algorithm Validation
#
# This script creates a reference state-space model where ALL parameters
# (Z, T, R, Q) are estimated using MARSS EM.
#
# Model: 2 states (VAR(1)), 3 observations
#
# State equation:  x_t = B x_{t-1} + w_t,  w_t ~ N(0, Q)
# Observation:     y_t = Z x_t + v_t,      v_t ~ N(0, R)
#
# where:
#   x_t is 2x1 state vector
#   y_t is 3x1 observation vector
#   B is 2x2 transition matrix (estimated)
#   Z is 3x2 observation matrix (estimated, with identification constraints)
#   Q is 2x2 diagonal (estimated)
#   R is 3x3 diagonal (estimated)

library(MARSS)

set.seed(54321)

# ============================================
# Model Parameters (True Values)
# ============================================

# State dimensions
m <- 2  # number of states
p <- 3  # number of observations

# True transition matrix B (VAR(1) coefficients)
B_true <- matrix(c(
  0.8, 0.1,   # state1 depends on both states
  0.2, 0.7    # state2 depends on both states
), nrow = 2, ncol = 2, byrow = TRUE)

# True observation matrix Z
# For identification: Z[1,1]=1, Z[1,2]=0, Z[2,2]=1 (upper triangular identity in first 2 rows)
Z_true <- matrix(c(
  1.0, 0.0,     # obs1 = state1 (fixed for identification)
  0.0, 1.0,     # obs2 = state2 (fixed for identification)
  0.6, 0.4      # obs3 = 0.6*state1 + 0.4*state2 (free)
), nrow = 3, ncol = 2, byrow = TRUE)

# True state covariance Q (diagonal)
true_var_state1 <- 0.5
true_var_state2 <- 0.8
Q_true <- diag(c(true_var_state1, true_var_state2))

# True observation covariance R (diagonal)
true_var_obs1 <- 0.3
true_var_obs2 <- 0.4
true_var_obs3 <- 0.5
R_true <- diag(c(true_var_obs1, true_var_obs2, true_var_obs3))

# ============================================
# Generate Data
# ============================================

n <- 200  # time series length

# Generate states (VAR(1) process)
states <- matrix(0, nrow = m, ncol = n)
states[, 1] <- c(0, 0)

for (t in 2:n) {
  states[, t] <- B_true %*% states[, t-1] +
                 rnorm(m, 0, sqrt(c(true_var_state1, true_var_state2)))
}

# Generate observations
y <- matrix(0, nrow = p, ncol = n)
for (t in 1:n) {
  y[, t] <- Z_true %*% states[, t] +
            rnorm(p, 0, sqrt(c(true_var_obs1, true_var_obs2, true_var_obs3)))
}

# ============================================
# MARSS Model Specification
# ============================================

# B matrix: all elements estimated
# Use "unconstrained" for full estimation
B_model <- matrix(list(
  "b11", "b12",
  "b21", "b22"
), nrow = 2, ncol = 2, byrow = TRUE)

# Z matrix: identification constraints
# Z[1,1]=1 (fixed), Z[1,2]=0 (fixed), Z[2,1]=0 (fixed), Z[2,2]=1 (fixed)
# Z[3,1] and Z[3,2] are free
Z_model <- matrix(list(
  1, 0,           # obs1 = state1 (fixed)
  0, 1,           # obs2 = state2 (fixed)
  "z31", "z32"    # obs3 = z31*state1 + z32*state2 (free)
), nrow = 3, ncol = 2, byrow = TRUE)

# Define the model structure
model_list <- list(
  B = B_model,
  U = matrix(0, m, 1),   # No drift
  Q = "diagonal and unequal",
  Z = Z_model,
  A = matrix(0, p, 1),   # No offset
  R = "diagonal and unequal",
  x0 = matrix(0, m, 1),
  V0 = diag(10, m),      # Initial covariance (smaller than before)
  tinitx = 0
)

# ============================================
# Run MARSS EM
# ============================================

cat("Running MARSS EM estimation (general model)...\n")
cat("\nTrue parameters:\n")
cat("B matrix:\n")
print(B_true)
cat("\nZ matrix:\n")
print(Z_true)
cat(sprintf("\nvar_state1 = %.4f, var_state2 = %.4f\n", true_var_state1, true_var_state2))
cat(sprintf("var_obs1 = %.4f, var_obs2 = %.4f, var_obs3 = %.4f\n",
            true_var_obs1, true_var_obs2, true_var_obs3))

# Fit the model using EM, then refine with BFGS
fit_em <- MARSS(y, model = model_list, method = "kem",
                control = list(maxit = 2000, conv.test.slope.tol = 1e-8, abstol = 1e-8),
                silent = TRUE)

# Refine with BFGS starting from EM estimates
fit <- MARSS(y, model = model_list, method = "BFGS",
             inits = fit_em,
             control = list(maxit = 2000),
             silent = TRUE)

# ============================================
# Extract Results
# ============================================

cat("\n\nMARSS EM Results:\n")
cat("=================\n")

# Extract B estimates
B_est <- coef(fit, type = "matrix")$B
cat("\nEstimated B (transition matrix):\n")
print(B_est)
cat(sprintf("  B[1,1] = %.6f (true: %.4f)\n", B_est[1,1], B_true[1,1]))
cat(sprintf("  B[1,2] = %.6f (true: %.4f)\n", B_est[1,2], B_true[1,2]))
cat(sprintf("  B[2,1] = %.6f (true: %.4f)\n", B_est[2,1], B_true[2,1]))
cat(sprintf("  B[2,2] = %.6f (true: %.4f)\n", B_est[2,2], B_true[2,2]))

# Extract Z estimates
Z_est <- coef(fit, type = "matrix")$Z
cat("\nEstimated Z (observation matrix):\n")
print(Z_est)
cat(sprintf("  Z[3,1] = %.6f (true: %.4f)\n", Z_est[3,1], Z_true[3,1]))
cat(sprintf("  Z[3,2] = %.6f (true: %.4f)\n", Z_est[3,2], Z_true[3,2]))

# Extract Q estimates
Q_est <- coef(fit, type = "matrix")$Q
cat("\nEstimated Q (state covariance):\n")
print(Q_est)
cat(sprintf("  Q[1,1] = %.6f (true: %.4f)\n", Q_est[1,1], true_var_state1))
cat(sprintf("  Q[2,2] = %.6f (true: %.4f)\n", Q_est[2,2], true_var_state2))

# Extract R estimates
R_est <- coef(fit, type = "matrix")$R
cat("\nEstimated R (observation covariance):\n")
print(R_est)
cat(sprintf("  R[1,1] = %.6f (true: %.4f)\n", R_est[1,1], true_var_obs1))
cat(sprintf("  R[2,2] = %.6f (true: %.4f)\n", R_est[2,2], true_var_obs2))
cat(sprintf("  R[3,3] = %.6f (true: %.4f)\n", R_est[3,3], true_var_obs3))

# Log-likelihood
cat(sprintf("\nFinal log-likelihood: %.6f\n", fit$logLik))
cat(sprintf("Number of iterations: %d\n", fit$numIter))
cat(sprintf("Converged: %s\n", ifelse(fit$convergence == 0, "Yes", "No")))

# ============================================
# Save Results for Julia Comparison
# ============================================

# Save data
write.csv(t(y), "test/marss_general_data.csv", row.names = FALSE)

# Save results as CSV for easy reading
results <- data.frame(
  parameter = c("B11", "B12", "B21", "B22", "Z31", "Z32",
                "Q11", "Q22", "R11", "R22", "R33", "loglik"),
  true_value = c(B_true[1,1], B_true[1,2], B_true[2,1], B_true[2,2],
                 Z_true[3,1], Z_true[3,2],
                 true_var_state1, true_var_state2,
                 true_var_obs1, true_var_obs2, true_var_obs3, NA),
  estimated = c(B_est[1,1], B_est[1,2], B_est[2,1], B_est[2,2],
                Z_est[3,1], Z_est[3,2],
                Q_est[1,1], Q_est[2,2],
                R_est[1,1], R_est[2,2], R_est[3,3], fit$logLik)
)
write.csv(results, "test/marss_general_results.csv", row.names = FALSE)

cat("\nData saved to: test/marss_general_data.csv\n")
cat("Results saved to: test/marss_general_results.csv\n")
