# Dense Gaussian reference for a DynamicFactorModel with AR(r) errors. It shares no
# code with the state-space implementation: the exact log-likelihood and the
# posterior of the latents come from conditioning the joint Gaussian of all
# periods, and the error autocovariances from the MA(∞) weights.

using LinearAlgebra

# Dense Gaussian representation of a DFM with AR(r) errors. Latents z = (u, e):
# u = (f₁, f₀, …, f₂₋ₛ, η₂, …, ηₙ) with (f₁, …, f₂₋ₛ) ~ N(0, Pf I), and
# e = (e₁, …, eₙ) stacked by period with eᵢ stationary AR(r). Data x = Mx u + e.
function _dfm_dense(Λ, Φ, Ση, δ, σ², N, n; s, Pf = 10.0)
    k = size(Ση, 1)
    p, q, r = length(Λ) - 1, length(Φ), length(δ)
    Tt = promote_type(eltype(Λ[1]), eltype(Φ[1]), eltype(Ση), eltype(δ), eltype(σ²))
    nu = k * s + k * (n - 1)
    F = Dict{Int, Matrix{Tt}}()
    for j in 0:(s - 1)
        M = zeros(Tt, k, nu)
        M[:, (j * k + 1):((j + 1) * k)] .= Matrix{Tt}(I, k, k)
        F[1 - j] = M
    end
    for t in 2:n
        M = zeros(Tt, k, nu)
        for l in 1:q
            M .+= Φ[l] * F[t - l]
        end
        M[:, (k * s + (t - 2) * k + 1):(k * s + (t - 1) * k)] .+= Matrix{Tt}(I, k, k)
        F[t] = M
    end
    Σu = zeros(Tt, nu, nu)
    for i in 1:(k * s)
        Σu[i, i] = Pf
    end
    for t in 2:n
        rows = (k * s + (t - 2) * k + 1):(k * s + (t - 1) * k)
        Σu[rows, rows] .= Ση
    end
    Mx = zeros(Tt, N * n, nu)
    for t in 1:n, a in 0:p

        Mx[((t - 1) * N + 1):(t * N), :] .+= Λ[a + 1] * F[t - a]
    end
    Σe = zeros(Tt, N * n, N * n)
    γ = _ar_acf(δ, n)
    for t1 in 1:n, t2 in 1:n, i in 1:N

        Σe[(t1 - 1) * N + i, (t2 - 1) * N + i] = σ²[i] * γ[abs(t1 - t2) + 1]
    end
    return (; F, Σu, Mx, Σe, nu, k, p)
end

# Autocovariances γ₀, …, γₙ₋₁ of a unit-innovation AR(r), from its MA(∞) weights.
function _ar_acf(δ, n; J = 4000)
    r = length(δ)
    ψ = zeros(eltype(δ), J)
    ψ[1] = 1
    for j in 2:J, l in 1:min(r, j - 1)

        ψ[j] += δ[l] * ψ[j - l]
    end
    return [sum(ψ[1:(J - h)] .* ψ[(1 + h):J]) for h in 0:(n - 1)]
end

function dfm_loglik_dense(Λ, Φ, Ση, δ, σ², y; s, Pf = 10.0)
    N, n = size(y)
    D = _dfm_dense(Λ, Φ, Ση, δ, σ², N, n; s, Pf)
    Σ = D.Mx * D.Σu * D.Mx' + D.Σe
    x = vec(y)
    o = findall(!isnan, x)
    c = cholesky(Symmetric(Σ[o, o]))
    return -(length(o) * log(2π) + logdet(c) + dot(x[o], c \ x[o])) / 2
end

# Posterior mean and covariance of the latents z = (u, e) given the observed y,
# at the parameters θ0.
function dfm_posterior_dense(θ0, y; s, Pf = 10.0)
    N, n = size(y)
    D = _dfm_dense(θ0..., N, n; s, Pf)
    nz = D.nu + N * n
    Σz = zeros(nz, nz)
    Σz[1:(D.nu), 1:(D.nu)] .= D.Σu
    Σz[(D.nu + 1):end, (D.nu + 1):end] .= D.Σe
    L = hcat(D.Mx, Matrix(1.0I, N * n, N * n))      # x = L z
    x = vec(y)
    o = findall(!isnan, x)
    Lo = L[o, :]
    K = Σz * Lo' / Symmetric(Lo * Σz * Lo')
    μ = K * x[o]
    S = Σz - K * Lo * Σz
    return (; μ, S, L, D)
end

# Expected complete-data log-likelihood of the errors, Σᵢ E[log N(eᵢ'; 0, σ²ᵢ Γₙ(δ))],
# with eᵢₜ' = xᵢₜ - Σₐ Λₐ[i, :] fₜ₋ₐ and the expectation under the posterior `post`.
function dfm_Q_errors(Λ, δ, σ², post; N, n)
    D = post.D
    T = promote_type(eltype(Λ[1]), eltype(δ), eltype(σ²))
    γ = _ar_acf(δ, n)
    Γn = [γ[abs(t1 - t2) + 1] for t1 in 1:n, t2 in 1:n]
    cΓ = cholesky(Symmetric(Γn))
    nz = length(post.μ)
    Q = zero(T)
    for i in 1:N
        # eᵢ' = E z, with x rows of series i minus loadings times the factor map
        E = zeros(T, n, nz)
        for t in 1:n
            E[t, :] .= post.L[(t - 1) * N + i, :]
            for a in 0:(D.p)
                E[t, 1:(D.nu)] .-= vec(Λ[a + 1][i:i, :] * D.F[t - a])
            end
        end
        Mom = E * post.S * E' + (E * post.μ) * (E * post.μ)'
        Q += -(n * log(2π) + n * log(σ²[i]) + logdet(cΓ) + tr(cΓ \ Mom) / σ²[i]) / 2
    end
    return Q
end
