using Documenter
using Siphon

makedocs(
    sitename = "Siphon",
    format = Documenter.HTML(prettyurls = false),
    modules = [Siphon],
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Getting Started" => "tutorials/getting_started.md",
            "Custom Models" => "tutorials/custom_models.md",
            "Dynamic Factor Models" => "tutorials/dynamic_factor.md",
            "ARMA Models" => "tutorials/arma_models.md",
            "Estimation Methods" => "tutorials/estimation_methods.md",
            "Parameter Transformations" => "tutorials/transformations.md",
            "Initial State Conventions" => "tutorials/initial_state.md",
            "Visualization" => "tutorials/visualization.md"
        ],
        "API Reference" => [
            "Core Functions" => "api/core.md",
            "DSL & Templates" => "api/dsl.md",
            "Matrix Helpers" => "api/matrix_helpers.md",
            "Estimation & Bayesian" => "api/optimization.md"
        ]
    ],
    warnonly = [:missing_docs]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
