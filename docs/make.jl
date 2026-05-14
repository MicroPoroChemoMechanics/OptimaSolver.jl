using OptimaSolver
using Documenter

DocMeta.setdocmeta!(
    OptimaSolver,
    :DocTestSetup,
    :(using OptimaSolver);
    recursive = true,
)

makedocs(;
    clean    = false,
    modules  = [OptimaSolver],
    authors  = "Jean-François Barthélémy",
    sitename = "OptimaSolver.jl",
    remotes  = nothing,
    format   = Documenter.HTML(;
        prettyurls    = get(ENV, "CI", "false") == "true",
        canonical     = "https://MicroPoroChemoMechanics.codeberg.page/OptimaSolver.jl",
        repolink      = "https://codeberg.org/MicroPoroChemoMechanics/OptimaSolver.jl",
        edit_link     = "main",
        collapselevel = 1,
    ),
    pages = [
        "Home"            => "index.md",
        "Getting Started" => "getting_started.md",
        "Theory"          => "theory.md",
        "Examples"        => [
            "Basic Usage"     => "examples/basic_usage.md",
            "Warm Start"      => "examples/warm_start.md",
            "Sensitivity"     => "examples/sensitivity.md",
            "SciML Interface" => "examples/sciml_interface.md",
        ],
        "API Reference"   => "api.md",
    ],
    warnonly = [:missing_docs, :docs_block],
)

deploydocs(;
    repo         = "git@codeberg-docs:MicroPoroChemoMechanics/OptimaSolver.jl.git",
    devbranch    = "main",
    push_preview = false,
)
