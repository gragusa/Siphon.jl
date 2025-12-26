using Documenter
using Siphon

makedocs(
    modules = [Siphon],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://gragusa.github.io/Siphon.jl/stable/",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Getting Started" => "tutorials/getting_started.md",
            "Custom Models" => "tutorials/custom_models.md",
            "ARMA Models" => "tutorials/arma_models.md",
            "Dynamic Factor Models" => "tutorials/dynamic_factor.md",
            "Estimation Methods" => "tutorials/estimation_methods.md",
            "Parameter Transformations" => "tutorials/transformations.md",
            "Visualization" => "tutorials/visualization.md",
        ],
        "API Reference" => [
            "Core Functions" => "api/core.md",
            "DSL & Templates" => "api/dsl.md",
            "Matrix Helpers" => "api/matrix_helpers.md",
            "Optimization & Bayesian" => "api/optimization.md",
        ],
    ],
    sitename = "Siphon.jl",
    authors = "Giuseppe Ragusa",
    warnonly = [:missing_docs, :cross_references],
)

deploydocs(
    repo = "github.com/gragusa/Siphon.jl.git",
    devbranch = "master",
    push_preview = true,
)
