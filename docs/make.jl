using Publish
using FastAI
using Pkg.Artifacts


# Download theme
download_artifact(
    Base.SHA1("6e4be8bec8da9323c18d777d7855ef79dddcf524"),
    "https://github.com/darsnack/flux-theme/releases/download/v0.2.1/flux-theme-0.2.1.tar.gz",
    "b00941248ecdb643a72960bbfda19a3128591eca3d7cbcb3bfb80a6ab1c7f99e",
)

Publish.Themes.default() = artifact"flux-theme"
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

p = Publish.Project(FastAI)

rm("dev", recursive = true, force = true)
rm(p.env["version"], recursive = true, force = true)

deploy(FastAI; root = "/FastAI.jl", force = true, label = "dev")
