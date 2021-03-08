using Pkg

Pkg.activate(@__DIR__)

using Artifacts
using FastAI
using Publish

cd(@__DIR__) do
    @eval Publish.Themes.default() = artifact"flux-theme"
end

serve(FastAI, port = 8001)
