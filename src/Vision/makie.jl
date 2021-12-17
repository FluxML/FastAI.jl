



function showblock!(grid, ::ShowMakie, block::Image{2}, obs)
    ax = cleanaxis(grid[1, 1])
    plotimage!(ax, obs)
end
function showblock!(grid, ::ShowMakie, block::Mask{2}, obs)
    ax = cleanaxis(grid[1, 1])
    plotmask!(ax, obs, block.classes)
end


function showblock!(grid, ::ShowMakie, block::Keypoints{2}, obs)
    ax = cleanaxis(grid[1, 1])
    h = maximum(first.(obs))
    ks = [SVector(x, h-y) for (y, x) in obs]
    Makie.scatter!(ax, ks)
end

function showblock!(grid, ::ShowMakie, block::Bounded{2, <:Keypoints{2}}, obs)
    ax = cleanaxis(grid[1, 1])
    h, w = block.size
    ks = [SVector(x, h-y) for (y, x) in obs]
    Makie.xlims!(ax, 0, w)
    Makie.ylims!(ax, 0, h)
    Makie.scatter!(ax, ks)
end

# ## Helpers


@recipe(PlotImage, image) do scene
    Makie.Attributes(
        alpha = 1,
        interpolate = false,
    )
end

function Makie.plot!(plot::PlotImage)
    im = plot[:image]
    rim = @lift alphacolor.(copy(rotr90($im)), $(plot.attributes[:alpha]))
    Makie.image!(plot, rim; plot.attributes...)
    return plot
end


@recipe(PlotMask, mask, classes) do scene
    Makie.Attributes()
end

function Makie.plot!(plot::PlotMask; kwargs...)
    mask = plot[:mask]
    classes = try
        classes = plot[:classes]
    catch
        classes = @lift unique($mask)
    end
    im = @lift _maskimage($mask, $classes)
    plotimage!(plot, im; alpha = 1, plot.attributes...)
    return plot
end

"""
    cleanaxis(f)

Create a `Makie.Axis` with no interactivity, decorations and aspect distortion.
"""
function cleanaxis(f; kwargs...)
    ax = Makie.Axis(f; kwargs...)
    ax.aspect = Makie.DataAspect()
    ax.xzoomlock = true
    ax.yzoomlock = true
    ax.xrectzoom = false
    ax.yrectzoom = false
    ax.panbutton = nothing
    ax.xpanlock = true
    ax.ypanlock = true
    ax.bottomspinevisible = false
    ax.leftspinevisible = false
    ax.rightspinevisible = false
    ax.topspinevisible = false
    Makie.hidedecorations!(ax)

    return ax
end
