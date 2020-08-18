using FastAI
using Documenter

makedocs(;
    modules=[FastAI],
    authors="Peter Wolf, Julia Community",
    repo="https://github.com/FluxML/FastAI.jl/blob/{commit}{path}#L{line}",
    sitename="FastAI.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://FluxML.github.io/FastAI.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "Training" => [
            "Overview" => "training/training.md",
            "Basic Train" => "training/basic-train.md",
        ],
        "Core" => [
            "Overview" => "core/core.md",
            "Torch Core" => "core/torch-core.md",
            "Basic Data" => "core/basic-data.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/FluxML/FastAI.jl",
)
