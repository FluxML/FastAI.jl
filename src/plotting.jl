
defaultfigure(;kwargs...) = Figure(
    ;resolution = (800, 800),
    background = RGBA(0, 0, 0, 0),
    kwargs...)

# ## Plotting interface definition

"""
    plotsample(method, sample)
"""
function plotsample(method, sample)
    f = defaultfigure(resolution = (300, 150))
    plotsample!(f, method, sample)
    return f
end

"""
    plotsample!(f, method, sample)
"""
function plotsample! end

"""
    plotxy(method, (x, y))
"""
function plotxy(method, xy)
    f = defaultfigure(resolution = (300, 150))
    plotxy!(f, method, xy)
    return f
end
"""
    plotxy!(f, method, (x, y))
"""
function plotxy! end


function plotbatch(method, (xs, ys))
    n = size(xs)[end]
    nrows = Int(ceil(sqrt(n)))
    f = Figure()
    is = Iterators.product(1:nrows, 1:nrows)
    for (i, (x, y)) in zip(is, DataLoaders.obsslices((xs, ys)))
        plotxy!(f[i...], method, (x, y))
    end
    return f
end


# ## Utilities

function imageaxis(f; kwargs...)
    ax = AbstractPlotting.Axis(f; kwargs...)
    ax.aspect = DataAspect()
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
    MakieLayout.tightlimits!(ax)
    hidedecorations!(ax)

    return ax
end


imageaxis(f::AbstractPlotting.FigurePosition; kwargs...) = imageaxis(f.fig; kwargs...)

# ## Plot recipes


@recipe(PlotImage, image) do scene
    Attributes()
end

function AbstractPlotting.plot!(plot::PlotImage)
    im = plot[:image]
    rim = @lift rotr90($im)
    image!(plot, rim; plot.attributes...)
    return plot
end


@recipe(PlotMask, mask, classes) do scene
    Attributes()
end

function AbstractPlotting.plot!(plot::PlotMask; kwargs...)
    mask = plot[:mask]
    classes = try
        classes = plot[:classes]
    catch
        classes = @lift unique($mask)
    end
    im = @lift maskimage($mask, $classes)
    plotimage!(plot, im; plot.attributes...)
    return plot
end


function maskimage(mask, classes)
    colors = distinguishable_colors(length(classes), transform = deuteranopic)
    im = map(c -> colors[c], mask)
end

maskimage(mask::AbstractArray{<:Gray{T}}, args...) where T =
    maskimage(reinterpret(T, mask), args...)
maskimage(mask::AbstractArray{<:Normed{T}}, args...) where T =
    maskimage(reinterpret(T, mask), args...)
