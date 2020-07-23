using FastAI
using Documenter

makedocs(;
    modules=[FastAI],
    authors="Peter Wolf, Julia Community",
    repo="https://github.com/opus111/FastAI.jl/blob/{commit}{path}#L{line}",
    sitename="FastAI.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://opus111.github.io/FastAI.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/opus111/FastAI.jl",
)
