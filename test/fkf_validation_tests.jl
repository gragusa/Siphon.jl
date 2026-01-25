"""
    fkf_validation_tests.jl

Tests comparing Siphon.jl Kalman filter against R's FKF package.
Reference values generated from FKF on Nile data with MLE-estimated parameters.

FKF model for Nile data:
    y_t = a_t + eps_t,   Var(eps) = GGt  (observation noise)
    a_t = a_{t-1} + eta_t, Var(eta) = HHt  (state noise)

NOTE: FKF's naming convention is confusing:
    - HHt (they call "observation covariance") is actually STATE variance (our Q)
    - GGt (they call "state covariance") is actually OBSERVATION variance (our H)

Reference R code:
```R
library(FKF)
data(Nile)
y <- Nile
dt <- ct <- matrix(0)
Zt <- Tt <- matrix(1)
a0 <- y[1]            # = 1120
P0 <- matrix(100)

fit.fkf <- optim(c(HHt = var(y, na.rm = TRUE) * .5,
                   GGt = var(y, na.rm = TRUE) * .5),
                 fn = function(par, ...)
                   -fkf(HHt = matrix(par[1]), GGt = matrix(par[2]), ...)\$logLik,
                 yt = rbind(y), a0 = a0, P0 = P0, dt = dt, ct = ct,
                 Zt = Zt, Tt = Tt)

fkf.obj <- fkf(a0, P0, dt, ct, Tt, Zt, HHt = matrix(fit.fkf\$par[1]),
               GGt = matrix(fit.fkf\$par[2]), yt = rbind(y))
```
"""

using Test
using Siphon
using LinearAlgebra

# ============================================
# FKF Reference Values (from R FKF package)
# ============================================

# Estimated parameters
# NOTE: FKF naming is backwards - HHt is STATE variance, GGt is OBSERVATION variance
const fkf_HHt = 1300.7770281326016   # state variance (Q in our notation)
const fkf_GGt = 15247.7728343637     # observation variance (H in our notation)
const fkf_a0 = 1120.0
const fkf_P0 = 100.0
const fkf_logLik = -637.6260115846168

# Predicted states at[,t] = E[α_t | y_{1:t-1}]
const fkf_at_first10 = [
    1120.0,
    1120.0,
    1123.3640894301147,
    1100.1322949354128,
    1120.689614058128,
    1129.105786767722,
    1136.2243333621434,
    1058.6623178789882,
    1100.715153023627,
    1167.6439427329178,
]
const fkf_at_last5 = [
    907.2004994420702,
    910.1801224150503,
    861.6505846700672,
    824.365695204029,
    803.0615762434667,
]

# Predicted variances Pt[,,t] = Var[α_t | y_{1:t-1}]
const fkf_Pt_first10 = [
    100.0,
    1400.125467798218,
    2583.1488137544575,
    3509.7075078221296,
    4153.784032169073,
    4565.25501622511,
    4814.120467776086,
    4959.684698730231,
    5043.164980325884,
    5090.500040863827,
]

# Filtered states att[,t] = E[α_t | y_{1:t}]
const fkf_att_first10 = [
    1120.0,
    1123.3640894301147,
    1100.1322949354128,
    1120.689614058128,
    1129.105786767722,
    1136.2243333621434,
    1058.6623178789882,
    1100.715153023627,
    1167.6439427329178,
    1160.7248944698297,
]
const fkf_att_last5 = [
    907.2004994420702,
    910.1801224150503,
    861.6505846700672,
    824.365695204029,
    803.0615762434667,
]

# Filtered variances Ptt[,,t] = Var[α_t | y_{1:t}]
const fkf_Ptt_first10 = [
    99.3484396656165,
    1282.3717856218557,
    2208.930479689528,
    2853.007004036471,
    3264.4779880925084,
    3513.3434396434845,
    3658.9076705976295,
    3742.387952193283,
    3789.7230127312255,
    3816.390345069674,
]
const fkf_Ptt_100 = 3850.3845142427253

# Innovations vt[,t] = y_t - E[y_t | y_{1:t-1}]
const fkf_vt_first10 = [
    0.0,
    40.0,
    -160.36408943011475,
    109.8677050645872,
    39.31038594187203,
    30.894213232277934,
    -323.2243333621434,
    171.33768212101177,
    269.28484697637305,
    -27.64394273291782,
]

# Innovation variances Ft[,,t] = Var[y_t | y_{1:t-1}]
const fkf_Ft_first10 = [
    15347.7728343637,
    16647.89830216192,
    17830.921648118157,
    18757.48034218583,
    19401.556866532774,
    19813.027850588813,
    20061.893302139786,
    20207.457533093933,
    20290.937814689583,
    20338.27287522753,
]

# Kalman gains Kt[,,t]
const fkf_Kt_first10 = [
    0.0065156033438349935,
    0.08410223575286951,
    0.14486905751319246,
    0.18710975268510607,
    0.21409539763967356,
    0.23041682728414667,
    0.2399634169753368,
    0.2454383333780467,
    0.2485427251506776,
    0.25029165810161635,
]

# Nile data
const nile_Y = [
    1120.0,
    1160.0,
    963.0,
    1210.0,
    1160.0,
    1160.0,
    813.0,
    1230.0,
    1370.0,
    1140.0,
    995.0,
    935.0,
    1110.0,
    994.0,
    1020.0,
    960.0,
    1180.0,
    799.0,
    958.0,
    1140.0,
    1100.0,
    1210.0,
    1150.0,
    1250.0,
    1260.0,
    1220.0,
    1030.0,
    1100.0,
    774.0,
    840.0,
    874.0,
    694.0,
    940.0,
    833.0,
    701.0,
    916.0,
    692.0,
    1020.0,
    1050.0,
    969.0,
    831.0,
    726.0,
    456.0,
    824.0,
    702.0,
    1120.0,
    1100.0,
    832.0,
    764.0,
    821.0,
    768.0,
    845.0,
    864.0,
    862.0,
    698.0,
    845.0,
    744.0,
    796.0,
    1040.0,
    759.0,
    781.0,
    865.0,
    845.0,
    944.0,
    984.0,
    897.0,
    822.0,
    1010.0,
    771.0,
    676.0,
    649.0,
    846.0,
    812.0,
    742.0,
    801.0,
    1040.0,
    860.0,
    874.0,
    848.0,
    890.0,
    744.0,
    749.0,
    838.0,
    1050.0,
    918.0,
    986.0,
    797.0,
    923.0,
    975.0,
    815.0,
    1020.0,
    906.0,
    901.0,
    1170.0,
    912.0,
    746.0,
    919.0,
    718.0,
    714.0,
    740.0,
]

# ============================================
# Tests
# ============================================

@testset "FKF Validation - Kalman Filter" begin
    # Setup: convert FKF parameters to our convention
    # FKF: HHt = state variance, GGt = observation variance (confusing names!)
    # Ours: H = observation variance, Q = state variance
    H_obs = fkf_GGt      # observation variance = 15247.77
    Q_state = fkf_HHt    # state variance = 1300.77

    # Build model matrices
    Z = [1.0;;]
    H = [H_obs;;]
    T = [1.0;;]
    R = [1.0;;]
    Q = [Q_state;;]

    # Initial conditions (FKF uses a0, P0 directly without propagation)
    a1 = [fkf_a0]
    P1 = [fkf_P0;;]

    # Data
    y = reshape(nile_Y, 1, :)
    n = size(y, 2)

    # Run our filter
    p = KFParms(Z, H, T, R, Q)
    result = kalman_filter(p, y, a1, P1)

    @testset "Log-likelihood" begin
        @test isapprox(result.loglik, fkf_logLik, rtol = 1e-10)
    end

    @testset "Predicted states (at)" begin
        # FKF at[,t] corresponds to our at[:,t]
        for (i, t) in enumerate(1:10)
            @test isapprox(result.at[1, t], fkf_at_first10[i], rtol = 1e-10)
        end
        # For local level model with T=1: at[:,t+1] = att[:,t]
        # This is a consistency check (fkf_at_last5 reference values were incorrect)
        for t = 2:n
            @test isapprox(result.at[1, t], result.att[1, t-1], rtol = 1e-10)
        end
    end

    @testset "Predicted variances (Pt)" begin
        for (i, t) in enumerate(1:10)
            @test isapprox(result.Pt[1, 1, t], fkf_Pt_first10[i], rtol = 1e-10)
        end
    end

    @testset "Filtered states (att)" begin
        # FKF att[,t] = E[α_t | y_{1:t}] now directly stored in att[:,t]
        for (i, t) in enumerate(1:10)
            @test isapprox(result.att[1, t], fkf_att_first10[i], rtol = 1e-10)
        end
        # Last 5 filtered states
        for (i, t) in enumerate(96:100)
            @test isapprox(result.att[1, t], fkf_att_last5[i], rtol = 1e-10)
        end
    end

    @testset "Filtered variances (Ptt)" begin
        # FKF Ptt[,,t] now directly stored in Ptt[:,:,t]
        for (i, t) in enumerate(1:10)
            @test isapprox(result.Ptt[1, 1, t], fkf_Ptt_first10[i], rtol = 1e-10)
        end
        # Last filtered variance
        @test isapprox(result.Ptt[1, 1, 100], fkf_Ptt_100, rtol = 1e-10)
    end

    @testset "Innovations (vt)" begin
        for (i, t) in enumerate(1:10)
            @test isapprox(result.vt[1, t], fkf_vt_first10[i], rtol = 1e-10)
        end
    end

    @testset "Innovation variances (Ft)" begin
        for (i, t) in enumerate(1:10)
            @test isapprox(result.Ft[1, 1, t], fkf_Ft_first10[i], rtol = 1e-10)
        end
    end

    @testset "Kalman gains (Kt)" begin
        for (i, t) in enumerate(1:10)
            @test isapprox(result.Kt[1, 1, t], fkf_Kt_first10[i], rtol = 1e-10)
        end
    end
end

@testset "FKF Validation - kalman_loglik consistency" begin
    # Verify kalman_loglik matches kalman_filter
    H_obs = fkf_GGt
    Q_state = fkf_HHt

    Z = [1.0;;]
    H = [H_obs;;]
    T = [1.0;;]
    R = [1.0;;]
    Q = [Q_state;;]

    a1 = [fkf_a0]
    P1 = [fkf_P0;;]
    y = reshape(nile_Y, 1, :)

    p = KFParms(Z, H, T, R, Q)

    ll_full = kalman_filter(p, y, a1, P1).loglik
    ll_direct = kalman_loglik(p, y, a1, P1)

    @test isapprox(ll_full, ll_direct, rtol = 1e-12)
    @test isapprox(ll_direct, fkf_logLik, rtol = 1e-10)
end

# ============================================
# Missing Data Tests
# ============================================

# FKF Reference Values with Missing Data (y[3] and y[10] are NA)
# NOTE: FKF's naming convention is backwards - see above
const fkf_na_HHt = 1385.0660439648536   # state variance (Q in our notation)
const fkf_na_GGt = 15124.1312944227     # observation variance (H in our notation)
const fkf_na_logLik = -627.0054683261661

# Predicted states with NA at t=3 and t=10
const fkf_na_at_first15 = [
    1120.0,
    1120.0,
    1123.5750503019735,
    1123.5750503019735,
    1142.0844759831643,
    1146.279488101204,
    1149.6506377958817,
    1064.742569925566,
    1107.0217591553153,
    1174.8280519639382,
    1174.8280519639382,
    1119.793956485756,
    1067.2207231992086,
    1078.9121347544572,
    1056.2123426414194,
]

const fkf_na_at_last5 = [
    906.3743167946249,
    909.6604839598359,
    859.7757863151869,
    821.8337918361999,
    800.5343884405607,
]

# Predicted variances
const fkf_na_Pt_first15 = [
    100.0,
    1484.409192036385,
    2736.804297745189,
    4121.870341710042,
    4624.165229614517,
    4926.459195251042,
    5101.087062097229,
    5199.586565399375,
    5254.398930507148,
    5284.671090115226,
    6669.73713408008,
    6013.614685801527,
    5687.828417361774,
    5518.432865524594,
    5428.24123048511,
]

# Filtered states
const fkf_na_att_first15 = [
    1120.0,
    1123.5750503019735,
    1123.5750503019735,
    1142.0844759831643,
    1146.279488101204,
    1149.6506377958817,
    1064.742569925566,
    1107.0217591553153,
    1174.8280519639382,
    1174.8280519639382,
    1119.793956485756,
    1067.2207231992086,
    1078.9121347544572,
    1056.2123426414194,
    1046.6480292688207,
]

const fkf_na_att_last5 = [
    906.3743167946249,
    909.6604839598359,
    859.7757863151869,
    821.8337918361999,
    800.5343884405607,
]

# Filtered variances
const fkf_na_Ptt_first15 = [
    99.34314807153144,
    1351.7382537803355,
    2736.804297745189,
    3239.0991856496626,
    3541.393151286188,
    3716.021018132376,
    3814.5205214345215,
    3869.3328865422945,
    3899.605046150373,
    5284.671090115226,
    4628.548641836674,
    4302.76237339692,
    4133.36682155974,
    4043.175186520256,
    3994.547732538421,
]

const fkf_na_Ptt_100 = 3936.454198447892

# Nile data with NA at positions 3 and 10
const nile_Y_na = [
    1120.0,
    1160.0,
    NaN,
    1210.0,
    1160.0,
    1160.0,
    813.0,
    1230.0,
    1370.0,
    NaN,
    995.0,
    935.0,
    1110.0,
    994.0,
    1020.0,
    960.0,
    1180.0,
    799.0,
    958.0,
    1140.0,
    1100.0,
    1210.0,
    1150.0,
    1250.0,
    1260.0,
    1220.0,
    1030.0,
    1100.0,
    774.0,
    840.0,
    874.0,
    694.0,
    940.0,
    833.0,
    701.0,
    916.0,
    692.0,
    1020.0,
    1050.0,
    969.0,
    831.0,
    726.0,
    456.0,
    824.0,
    702.0,
    1120.0,
    1100.0,
    832.0,
    764.0,
    821.0,
    768.0,
    845.0,
    864.0,
    862.0,
    698.0,
    845.0,
    744.0,
    796.0,
    1040.0,
    759.0,
    781.0,
    865.0,
    845.0,
    944.0,
    984.0,
    897.0,
    822.0,
    1010.0,
    771.0,
    676.0,
    649.0,
    846.0,
    812.0,
    742.0,
    801.0,
    1040.0,
    860.0,
    874.0,
    848.0,
    890.0,
    744.0,
    749.0,
    838.0,
    1050.0,
    918.0,
    986.0,
    797.0,
    923.0,
    975.0,
    815.0,
    1020.0,
    906.0,
    901.0,
    1170.0,
    912.0,
    746.0,
    919.0,
    718.0,
    714.0,
    740.0,
]

@testset "FKF Validation - Missing Data" begin
    # Setup with FKF parameters (corrected naming)
    H_obs = fkf_na_GGt      # observation variance
    Q_state = fkf_na_HHt    # state variance

    Z = [1.0;;]
    H = [H_obs;;]
    T = [1.0;;]
    R = [1.0;;]
    Q = [Q_state;;]

    a1 = [1120.0]
    P1 = [100.0;;]
    y = reshape(nile_Y_na, 1, :)

    p = KFParms(Z, H, T, R, Q)
    result = kalman_filter(p, y, a1, P1)

    @testset "Predicted states (at) with missing data" begin
        for (i, t) in enumerate(1:15)
            @test isapprox(result.at[1, t], fkf_na_at_first15[i], rtol = 1e-10)
        end
        # For local level model with T=1: at[:,t+1] = att[:,t] (when no missing)
        # When t is missing: at[:,t+1] = at[:,t]
        # This is a consistency check (fkf_na_at_last5 reference values were incorrect)
        n = size(y, 2)
        for t = 2:n
            if result.missing_mask[t-1]
                # Missing at t-1: predicted state propagates unchanged
                @test isapprox(result.at[1, t], result.at[1, t-1], rtol = 1e-10)
            else
                # Not missing: predicted state = T * filtered state
                @test isapprox(result.at[1, t], result.att[1, t-1], rtol = 1e-10)
            end
        end
    end

    @testset "Predicted variances (Pt) with missing data" begin
        for (i, t) in enumerate(1:15)
            @test isapprox(result.Pt[1, 1, t], fkf_na_Pt_first15[i], rtol = 1e-10)
        end
    end

    @testset "Filtered states (att) with missing data" begin
        # Filtered states now directly stored in att
        for (i, t) in enumerate(1:15)
            @test isapprox(result.att[1, t], fkf_na_att_first15[i], rtol = 1e-10)
        end
        for (i, t) in enumerate(96:100)
            @test isapprox(result.att[1, t], fkf_na_att_last5[i], rtol = 1e-10)
        end
    end

    @testset "Filtered variances (Ptt) with missing data" begin
        # Filtered variances now directly stored in Ptt
        for (i, t) in enumerate(1:15)
            @test isapprox(result.Ptt[1, 1, t], fkf_na_Ptt_first15[i], rtol = 1e-10)
        end
        @test isapprox(result.Ptt[1, 1, 100], fkf_na_Ptt_100, rtol = 1e-10)
    end

    @testset "Missing data behavior" begin
        # At NA positions, filtered state should equal predicted state
        @test isapprox(result.att[1, 3], result.at[1, 3], rtol = 1e-15)
        @test isapprox(result.att[1, 10], result.at[1, 10], rtol = 1e-15)

        # Innovations at NA positions should be NaN
        @test isnan(result.vt[1, 3])
        @test isnan(result.vt[1, 10])
    end

    @testset "Log-likelihood with missing data" begin
        # NOTE: FKF includes constant term -n*log(2π)/2 for ALL n=100 observations,
        # even when some are missing. We correctly count only n_obs=98 non-missing.
        # The difference is exactly 2 * log(2π)/2 = log(2π) ≈ 1.8379
        #
        # Our approach is statistically correct: the likelihood should only
        # include terms for actual observations.
        n_missing = 2
        const_term_diff = -n_missing * log(2π) / 2

        # Our log-likelihood + correction should match FKF
        @test isapprox(result.loglik + const_term_diff, fkf_na_logLik, rtol = 1e-10)

        # Verify the correction is as expected
        @test isapprox(const_term_diff, -1.8378770664093453, rtol = 1e-10)
    end
end
