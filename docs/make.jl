using FastAI2Julia
using Documenter

makedocs(;
    modules=[FastAI2Julia],
    authors="Peter Wolf, Julia Community",
    repo="https://github.com/opus111/FastAI2Julia.jl/blob/{commit}{path}#L{line}",
    sitename="FastAI2Julia.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://opus111.github.io/FastAI2Julia.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/opus111/FastAI2Julia.jl",
)
