module SiphonDistributionsExt

using Siphon: Siphon
using Distributions: UnivariateDistribution, logpdf

Siphon._is_univariate_prior(::UnivariateDistribution) = true
Siphon._univariate_logpdf(d::UnivariateDistribution, x) = logpdf(d, x)

end
