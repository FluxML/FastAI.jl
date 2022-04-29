using Pollen
using Pkg
using Crayons
Crayons.COLORS[:nothing] = 67
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using FastAI, Flux, FluxTraining
import DataAugmentation
m = FastAI
ms = [
    DataAugmentation,
    Flux,
    FluxTraining,
    m,
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
        SaveAttributes((:title,)),
        LoadFrontendConfig(Pkg.pkgdir(m))
    ],
)
