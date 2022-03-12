using Pollen
using Pkg

using FastAI, Flux, FluxTraining
import DataAugmentation
m = FastAI
ms = [
    DataAugmentation,
    Flux,
    FluxTraining,
    FastAI.Vision,
    FastAI.Models,
    FastAI.Tabular,
    FastAI.Datasets,
    m,
]


project = Project(
    Pollen.Rewriter[
        Pollen.DocumentFolder(pkgdir(m), prefix = "documents"),
        Pollen.ParseCode(),
        Pollen.ExecuteCode(),
        Pollen.PackageDocumentation(ms),
        Pollen.DocumentGraph(),
        Pollen.SearchIndex(),
        Pollen.SaveAttributes((:title,)),
        Pollen.LoadFrontendConfig(pkgdir(m))
    ],
)
