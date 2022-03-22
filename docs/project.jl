using Pollen
using Pkg
using Crayons
Crayons.COLORS[:nothing] = 67

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
