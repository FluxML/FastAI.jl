module FastMakie

import FastAI: FastAI, createhandle, showblock!, AbstractBlock, OneHotTensor, OneHotTensorMulti,
               Label, LabelMulti, ShowBackend, ShowMakie, axiskwargs, showblock,
               showblocks!, showblocks
import Makie


"""
    cleanaxis(f)

Create a `Makie.Axis` with no interactivity, decorations and aspect distortion.
"""
function makeaxis(f; kwargs...)
    ax = Makie.Axis(f; kwargs...)
    ax.aspect = Makie.DataAspect()
    ax.xzoomlock = true
    ax.yzoomlock = true
    ax.xrectzoom = false
    ax.yrectzoom = false
    ax.xpanlock = true
    ax.ypanlock = true
    ax.bottomspinevisible = false
    ax.leftspinevisible = false
    ax.rightspinevisible = false
    ax.topspinevisible = false
    Makie.hidedecorations!(ax)
    Makie.tightlimits!(ax)

    return ax
end

blockaxis(f, block::AbstractBlock) = makeaxis(f; axiskwargs(block)...)

include("showmakie.jl")
include("blocks.jl")
include("lrfind.jl")

FastAI.SHOW_BACKEND[] = ShowMakie()


end
