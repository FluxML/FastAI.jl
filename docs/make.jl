using Publish
using FastAI
using Pkg.Artifacts


Publish.Themes.default() = artifact"flux-theme"

p = Publish.Project(FastAI)

rm("dev", recursive = true, force = true)
rm(p.env["version"], recursive = true, force = true)

deploy(FastAI; root = "/FastAI.jl", force = true, label = "dev")
