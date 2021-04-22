
defaultfigure(;kwargs...) = Figure(
    ;resolution=(800, 800),
    background=RGBA(0, 0, 0, 0),
    kwargs...)

# ## Plotting interface definition

"""
    plotsample(method, sample)

Plot a `sample` of a `method` or `Task` type.
See also [`plotsample!`](#)

## Examples

```julia
sample = (rand(Gray, 28, 28), 1)
plotsample(ImageClassificationTask, sample)

method = ImageClassification(1:10, (16, 16))
plotsample(method, sample)
```
"""
function plotsample(method, sample)
    f = defaultfigure(resolution=(300, 150))
    plotsample!(f, method, sample)
    return f
end

"""
    plotsample!(f, Task, sample)
    plotsample!(f, method, sample)

Plot a `sample` of a `method` or `Task` type on figure or axis `f`.
See also [`plotsample`](#)

"""
function plotsample! end

"""
    plotxy(method, (x, y))
"""
function plotxy(method, xy)
    f = defaultfigure(resolution=(300, 150))
    plotxy!(f, method, xy)
    return f
end
"""
    plotxy!(f, method, (x, y))
"""
function plotxy! end


"""
    plotbatch(method, xs, ys)
    plotbatch(method, dataloader)

Plot an encoded batch of data in a grid.
"""
plotbatch(method, xs, ys) = plotbatch!(defaultfigure(), method, xs, ys)

function plotbatch!(f, method, xs, ys)
    n = size(xs)[end]
    nrows = Int(ceil(sqrt(n)))
    is = Iterators.product(1:nrows, 1:nrows)
    for (i, (x, y)) in zip(is, DataLoaders.obsslices((xs, ys)))
        plotxy!(f[i...], method, (x, y))
    end
    return f
end
# TODO: implement `plotbatch(method, dataloader; n)`
"""
    plotsamples(Task, samples)
    plotsamples(method, samples)

Plot samples for a `LearningTask`/`LearningMethod` in a grid.
"""
function plotsamples(method::LearningMethod, samples)
    n = length(samples)
    nrows = Int(ceil(sqrt(n)))
    f = Figure()
    is = Iterators.product(1:nrows, 1:nrows)
    for (i, sample) in zip(is, samples)
        plotsample!(f[i...], method, sample)
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
    rim = @lift copy(rotr90($im))
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
    colors = distinguishable_colors(length(classes), transform=deuteranopic)
    im = map(c -> colors[c], mask)
end

maskimage(mask::AbstractArray{<:Gray{T}}, args...) where T =
    maskimage(reinterpret(T, mask), args...)
maskimage(mask::AbstractArray{<:Normed{T}}, args...) where T =
    maskimage(reinterpret(T, mask), args...)
