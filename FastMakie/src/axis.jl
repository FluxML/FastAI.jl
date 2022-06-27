
"""
    makeaxis(f[; interactive, clean, dataaspect])

Create a `Makie.Axis` that block data can be plotted to. When plotting block data,
(using `showblock!`), keyword arguments from `axiskwargs(block)` are passed to this
function.
"""
function makeaxis(f; interactive = false, clean = true, dataaspect = true, kwargs...)
    ax = Makie.Axis(f; kwargs...)
    if dataaspect
        ax.aspect = Makie.DataAspect()
    end
    if !interactive
        ax.xzoomlock = true
        ax.yzoomlock = true
        ax.xrectzoom = false
        ax.yrectzoom = false
        ax.xpanlock = true
        ax.ypanlock = true
    end
    if clean
        ax.bottomspinevisible = false
        ax.leftspinevisible = false
        ax.rightspinevisible = false
        ax.topspinevisible = false
        Makie.hidedecorations!(ax)
        Makie.tightlimits!(ax)
    end
    return ax
end

function blockaxis(f, backend::ShowMakie, block::AbstractBlock)
    ax = makeaxis(f; axiskwargs(block)...)
    if get(backend.kwargs, :showblock, true)
        ax.subtitle = FastAI.blockname(block)
        ax.subtitlefont = "mono"
    end
    return ax
end

@testset "makeaxis" begin
    @test_nowarn makeaxis(Makie.Figure()[1, 1])
    @test_nowarn makeaxis(Makie.Figure()[1, 1], interactive = true)
    @test_nowarn makeaxis(Makie.Figure()[1, 1], interactive = true, dataaspect = false)
end
