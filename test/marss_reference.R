# MARSS Reference Script for EM Algorithm Validation
#
# This script creates a reference state-space model and runs MARSS EM estimation
# to validate the Siphon.jl EM implementation.
#
# Model: 2 states (random walks), 3 observations
#
# State equation:  x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)
# Observation:     y_t = Z * x_t + v_t,  v_t ~ N(0, R)
#
# where:
#   x_t is 2x1 state vector
#   y_t is 3x1 observation vector
#   Z is 3x2 fixed observation matrix (identity-like)
#   Q is 2x2 diagonal (var_state1, var_state2)
#   R is 3x3 diagonal (var_obs1, var_obs2, var_obs3)

library(MARSS)

set.seed(12345)

# ============================================
# Model Parameters (True Values)
# ============================================

# State dimensions
m <- 2  # number of states
p <- 3  # number of observations

# True parameters
true_var_state1 <- 1.0
true_var_state2 <- 2.0
true_var_obs1 <- 0.5
true_var_obs2 <- 1.0
true_var_obs3 <- 1.5

# Observation matrix Z (fixed, 3x2)
# Each observation loads on one state for simplicity
# obs1 = state1, obs2 = state2, obs3 = 0.7*state1 + 0.3*state2
Z_true <- matrix(c(
  1.0, 0.0,     # obs1 = state1
  0.0, 1.0,     # obs2 = state2
  0.7, 0.3      # obs3 = 0.7*state1 + 0.3*state2
), nrow = 3, ncol = 2, byrow = TRUE)

# State covariance Q (diagonal)
Q_true <- diag(c(true_var_state1, true_var_state2))

# Observation covariance R (diagonal)
R_true <- diag(c(true_var_obs1, true_var_obs2, true_var_obs3))

# ============================================
# Generate Data
# ============================================

n <- 100  # time series length

# Generate states (random walks)
states <- matrix(0, nrow = m, ncol = n)
states[, 1] <- c(0, 0)

for (t in 2:n) {
  states[, t] <- states[, t-1] + rnorm(m, 0, sqrt(c(true_var_state1, true_var_state2)))
}

# Generate observations
y <- matrix(0, nrow = p, ncol = n)
for (t in 1:n) {
  y[, t] <- Z_true %*% states[, t] + rnorm(p, 0, sqrt(c(true_var_obs1, true_var_obs2, true_var_obs3)))
}

# ============================================
# MARSS Model Specification
# ============================================

# MARSS uses a specific form:
#   x_t = B x_{t-1} + u + w_t,  w_t ~ MVN(0, Q)
#   y_t = Z x_t + a + v_t,     v_t ~ MVN(0, R)

# B = identity (random walk)
# u = 0 (no drift)
# a = 0 (no offset)

# Define the model structure
model_list <- list(
  B = diag(m),           # Identity for random walk
  U = matrix(0, m, 1),   # No drift
  Q = "diagonal and unequal",  # Diagonal Q, estimate variances
  Z = Z_true,            # Fixed Z matrix
  A = matrix(0, p, 1),   # No offset
  R = "diagonal and unequal",  # Diagonal R, estimate variances
  x0 = matrix(0, m, 1),  # Initial state
  V0 = diag(1e7, m),     # Diffuse initial covariance
  tinitx = 0             # x0 is at t=0
)

# ============================================
# Run MARSS EM
# ============================================

cat("Running MARSS EM estimation...\n")
cat("True parameters:\n")
cat(sprintf("  var_state1 = %.4f\n", true_var_state1))
cat(sprintf("  var_state2 = %.4f\n", true_var_state2))
cat(sprintf("  var_obs1 = %.4f\n", true_var_obs1))
cat(sprintf("  var_obs2 = %.4f\n", true_var_obs2))
cat(sprintf("  var_obs3 = %.4f\n", true_var_obs3))
cat("\n")

# Fit the model using EM
fit <- MARSS(y, model = model_list, method = "kem",
             control = list(maxit = 500, conv.test.slope.tol = 1e-9, abstol = 1e-9))

# ============================================
# Extract Results
# ============================================

cat("\nMARSS EM Results:\n")
cat("================\n")

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
write.csv(t(y), "test/marss_reference_data.csv", row.names = FALSE)

# Save results as CSV for easy reading
results <- data.frame(
  parameter = c("Q11", "Q22", "R11", "R22", "R33", "loglik"),
  true_value = c(true_var_state1, true_var_state2,
                 true_var_obs1, true_var_obs2, true_var_obs3, NA),
  estimated = c(Q_est[1,1], Q_est[2,2],
                R_est[1,1], R_est[2,2], R_est[3,3], fit$logLik)
)
write.csv(results, "test/marss_reference_results.csv", row.names = FALSE)

cat("\nData saved to: test/marss_reference_data.csv\n")
cat("Results saved to: test/marss_reference_results.csv\n")

# ============================================
# Also print Z matrix used
# ============================================
cat("\nZ matrix (fixed):\n")
print(Z_true)
