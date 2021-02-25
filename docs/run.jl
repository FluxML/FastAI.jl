using Pkg

Pkg.activate(@__DIR__)

using Artifacts
using FastAI
using Publish

cd(@__DIR__)
Publish.Themes.default() = artifact"flux-theme"

serve(FastAI, port = 8002)
