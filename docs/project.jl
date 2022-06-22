using Pollen
using Pkg
using ImageShow

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using FastAI, FastVision, FastMakie, FastTabular, Flux, FluxTraining
import DataAugmentation, MLUtils
m = FastAI
ms = [
    DataAugmentation,
    Flux,
    FluxTraining,
    MLUtils,
    m,
    FastVision,
    FastTabular,
    FastMakie,
]

project = Project(
    Pollen.Rewriter[
        DocumentFolder(Pkg.pkgdir(m), prefix = "documents"),
        ParseCode(),
        ExecuteCode(),
        PackageDocumentation(ms),
        StaticResources(),
        DocumentGraph(),
        SearchIndex(),
        SaveAttributes((:title,), useoutputs=false),
        LoadFrontendConfig(Pkg.pkgdir(m))
    ],
)
