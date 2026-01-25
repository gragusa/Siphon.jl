# MARSS Reference Script for DNS Model Validation
#
# This script generates reference values for validating Siphon.jl's DNS implementation
# against the MARSS R package.
#
# Dynamic Nelson-Siegel Model:
#   y_t = Z(lambda) * f_t + eps_t,  eps_t ~ N(0, R)
#   f_t = B * f_{t-1} + eta_t,      eta_t ~ N(0, Q)
#
# where Z(lambda) is the p x 3 loading matrix:
#   Z[i,1] = 1                               (Level)
#   Z[i,2] = (1 - exp(-lambda*tau_i))/(lambda*tau_i)  (Slope)
#   Z[i,3] = Z[i,2] - exp(-lambda*tau_i)     (Curvature)
#
# Notation mapping (Siphon.jl -> MARSS):
#   T -> B (transition matrix)
#   H -> R (observation covariance)
#   Q -> Q (state covariance)
#   Z -> Z (loading matrix)

library(MARSS)
set.seed(42)

# ============================================
# DNS Loading Functions
# ============================================

dns_loadings <- function(lambda, maturities) {
  p <- length(maturities)
  Z <- matrix(1, nrow = p, ncol = 3)

  for (i in 1:p) {
    tau <- maturities[i]
    x <- lambda * tau
    if (x < 1e-10) {
      Z[i, 2] <- 1.0 - x/2
    } else {
      Z[i, 2] <- (1 - exp(-x)) / x
    }
    Z[i, 3] <- Z[i, 2] - exp(-x)
  }
  return(Z)
}

# ============================================
# Load Real Yield Data
# ============================================

cat("Loading yield data from monthlyyields.csv...\n")

# Get script directory for relative paths
script_dir <- dirname(sys.frame(1)$ofile)
if (is.null(script_dir) || script_dir == "") {
  # Fallback for interactive use
  script_dir <- getwd()
}

# Load data
yields_path <- file.path(dirname(script_dir), "examples", "monthlyyields.csv")
if (!file.exists(yields_path)) {
  # Try alternative path
  yields_path <- "examples/monthlyyields.csv"
}
yields <- read.csv(yields_path)

# Select maturities: 3, 12, 24, 60, 120 months
# Column names are Y3, Y12, Y24, etc.
maturities <- c(3, 12, 24, 60, 120)
p <- length(maturities)
m <- 3  # states (Level, Slope, Curvature)

# Extract columns (note: Y3 means 3-month maturity)
col_names <- paste0("Y", maturities)
y_data <- as.matrix(yields[, col_names])

# Use first 200 observations for tractable estimation
n_obs <- 200
y_data <- y_data[1:n_obs, ]

# MARSS expects p x n (variables x time)
y <- t(y_data)

cat("Data dimensions: ", nrow(y), "maturities x", ncol(y), "observations\n")
cat("Maturities:", maturities, "\n")

# ============================================
# Configuration
# ============================================

# Fixed lambda (Diebold-Li canonical value for monthly data)
lambda_fixed <- 0.0609

# Build Z matrix at fixed lambda
Z_fixed <- dns_loadings(lambda_fixed, maturities)

cat("\nZ matrix at lambda =", lambda_fixed, ":\n")
print(round(Z_fixed, 4))

# ============================================
# Model 1: Diagonal T, Diagonal Q, Diagonal H
# ============================================

cat("\n=== Model 1: Diagonal T, Diagonal Q, Diagonal H ===\n")

# B (transition) diagonal structure
B_diag <- matrix(list(
  "B_L", 0,     0,
  0,     "B_S", 0,
  0,     0,     "B_C"
), nrow = 3, ncol = 3, byrow = TRUE)

model_diag <- list(
  B = B_diag,
  U = matrix(0, m, 1),
  Q = "diagonal and unequal",
  Z = Z_fixed,
  A = matrix(0, p, 1),
  R = "diagonal and unequal",
  x0 = matrix(0, m, 1),
  V0 = diag(100, m),
  tinitx = 0
)

cat("Running MARSS EM...\n")
fit_diag <- MARSS(y, model = model_diag, method = "kem",
                  control = list(maxit = 5000, conv.test.slope.tol = 1e-9, abstol = 1e-9),
                  silent = TRUE)

cat("Convergence:", fit_diag$convergence, "\n")
cat("Iterations:", fit_diag$numIter, "\n")
cat("Log-likelihood:", fit_diag$logLik, "\n")

# Extract parameters
B_diag_est <- coef(fit_diag, type = "matrix")$B
Q_diag_est <- coef(fit_diag, type = "matrix")$Q
R_diag_est <- coef(fit_diag, type = "matrix")$R

cat("\nEstimated B (diagonal):\n")
print(round(B_diag_est, 6))
cat("\nEstimated Q (diagonal):\n")
print(round(Q_diag_est, 6))
cat("\nEstimated R (diagonal):\n")
print(round(R_diag_est, 6))

# ============================================
# Model 2: Full T, Full Q, Diagonal H
# ============================================

cat("\n=== Model 2: Full T, Full Q, Diagonal H ===\n")

B_full <- matrix(list(
  "B_1_1", "B_1_2", "B_1_3",
  "B_2_1", "B_2_2", "B_2_3",
  "B_3_1", "B_3_2", "B_3_3"
), nrow = 3, ncol = 3, byrow = TRUE)

model_full <- list(
  B = B_full,
  U = matrix(0, m, 1),
  Q = "unconstrained",
  Z = Z_fixed,
  A = matrix(0, p, 1),
  R = "diagonal and unequal",
  x0 = matrix(0, m, 1),
  V0 = diag(100, m),
  tinitx = 0
)

cat("Running MARSS EM...\n")
fit_full <- MARSS(y, model = model_full, method = "kem",
                  control = list(maxit = 5000, conv.test.slope.tol = 1e-9, abstol = 1e-9),
                  silent = TRUE)

cat("Convergence:", fit_full$convergence, "\n")
cat("Iterations:", fit_full$numIter, "\n")
cat("Log-likelihood:", fit_full$logLik, "\n")

# Extract parameters
B_full_est <- coef(fit_full, type = "matrix")$B
Q_full_est <- coef(fit_full, type = "matrix")$Q
R_full_est <- coef(fit_full, type = "matrix")$R

cat("\nEstimated B (full):\n")
print(round(B_full_est, 6))
cat("\nEstimated Q (full):\n")
print(round(Q_full_est, 6))
cat("\nEstimated R (diagonal):\n")
print(round(R_full_est, 6))

# ============================================
# Model 3: Diagonal T, Diagonal Q, Scalar H
# ============================================

cat("\n=== Model 3: Diagonal T, Diagonal Q, Scalar H ===\n")

model_scalar <- list(
  B = B_diag,
  U = matrix(0, m, 1),
  Q = "diagonal and unequal",
  Z = Z_fixed,
  A = matrix(0, p, 1),
  R = "diagonal and equal",  # Scalar: R = sigma^2 * I
  x0 = matrix(0, m, 1),
  V0 = diag(100, m),
  tinitx = 0
)

cat("Running MARSS EM...\n")
fit_scalar <- MARSS(y, model = model_scalar, method = "kem",
                    control = list(maxit = 5000, conv.test.slope.tol = 1e-9, abstol = 1e-9),
                    silent = TRUE)

cat("Convergence:", fit_scalar$convergence, "\n")
cat("Iterations:", fit_scalar$numIter, "\n")
cat("Log-likelihood:", fit_scalar$logLik, "\n")

# Extract parameters
B_scalar_est <- coef(fit_scalar, type = "matrix")$B
Q_scalar_est <- coef(fit_scalar, type = "matrix")$Q
R_scalar_est <- coef(fit_scalar, type = "matrix")$R

cat("\nEstimated B (diagonal):\n")
print(round(B_scalar_est, 6))
cat("\nEstimated Q (diagonal):\n")
print(round(Q_scalar_est, 6))
cat("\nEstimated R (scalar):\n")
print(round(R_scalar_est[1,1], 6))

# ============================================
# Save Results to CSV
# ============================================

output_dir <- ifelse(grepl("test$", script_dir), script_dir, file.path(script_dir, "test"))
if (!dir.exists(output_dir)) {
  output_dir <- "test"
}

# Save data (transposed back to n x p for Julia's readdlm with header=true)
cat("\nSaving data to marss_dns_data.csv...\n")
write.csv(t(y), file.path(output_dir, "marss_dns_data.csv"), row.names = FALSE)

# Save diagonal model results
results_diag <- data.frame(
  parameter = c("B_L", "B_S", "B_C",
                "Q_L", "Q_S", "Q_C",
                paste0("R_", 1:p),
                "loglik", "lambda", "n_obs"),
  value = c(diag(B_diag_est),
            diag(Q_diag_est),
            diag(R_diag_est),
            fit_diag$logLik,
            lambda_fixed,
            n_obs)
)
write.csv(results_diag, file.path(output_dir, "marss_dns_diag_results.csv"), row.names = FALSE)
cat("Saved diagonal model results to marss_dns_diag_results.csv\n")

# Save full model results
# Flatten B matrix row-major (row 1, row 2, row 3)
B_flat <- as.vector(t(B_full_est))
B_names <- c("B_1_1", "B_1_2", "B_1_3", "B_2_1", "B_2_2", "B_2_3", "B_3_1", "B_3_2", "B_3_3")

# Flatten Q matrix (symmetric, save all elements row-major)
Q_flat <- as.vector(t(Q_full_est))
Q_names <- c("Q_1_1", "Q_1_2", "Q_1_3", "Q_2_1", "Q_2_2", "Q_2_3", "Q_3_1", "Q_3_2", "Q_3_3")

results_full <- data.frame(
  parameter = c(B_names, Q_names, paste0("R_", 1:p), "loglik", "lambda", "n_obs"),
  value = c(B_flat, Q_flat, diag(R_full_est), fit_full$logLik, lambda_fixed, n_obs)
)
write.csv(results_full, file.path(output_dir, "marss_dns_full_results.csv"), row.names = FALSE)
cat("Saved full model results to marss_dns_full_results.csv\n")

# Save scalar H model results
results_scalar <- data.frame(
  parameter = c("B_L", "B_S", "B_C",
                "Q_L", "Q_S", "Q_C",
                "R_scalar",
                "loglik", "lambda", "n_obs"),
  value = c(diag(B_scalar_est),
            diag(Q_scalar_est),
            R_scalar_est[1,1],
            fit_scalar$logLik,
            lambda_fixed,
            n_obs)
)
write.csv(results_scalar, file.path(output_dir, "marss_dns_scalar_results.csv"), row.names = FALSE)
cat("Saved scalar H model results to marss_dns_scalar_results.csv\n")

# Save Z matrix info
z_info <- data.frame(
  maturity = maturities,
  Z_level = Z_fixed[,1],
  Z_slope = Z_fixed[,2],
  Z_curvature = Z_fixed[,3]
)
write.csv(z_info, file.path(output_dir, "marss_dns_Z.csv"), row.names = FALSE)
cat("Saved Z matrix to marss_dns_Z.csv\n")

# ============================================
# Summary
# ============================================

cat("\n=== Summary ===\n")
cat("Lambda:", lambda_fixed, "\n")
cat("Maturities:", maturities, "\n")
cat("Observations:", n_obs, "\n")
cat("\nLog-likelihoods:\n")
cat("  Diagonal model:", fit_diag$logLik, "\n")
cat("  Full model:", fit_full$logLik, "\n")
cat("  Scalar H model:", fit_scalar$logLik, "\n")
cat("\nFiles saved to:", output_dir, "\n")
