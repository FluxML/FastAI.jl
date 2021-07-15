
defaultfigure(;kwargs...) = Figure(
    ;resolution=(800, 800),
    background=RGBA(0, 0, 0, 0),
    kwargs...)

# ## Plotting interface definition

"""
    plotsample(method, sample)

Plot a `sample` of a `method`.

Learning methods should implement [`plotsample!`](#) to make this work.

## Examples

```julia
using FastAI, Colors

sample = (rand(Gray, 28, 28), 1)
method = ImageClassification(1:10, (16, 16))
plotsample(method, sample)
```
"""
function plotsample(method, sample; figkwargs...)
    f = defaultfigure(; figkwargs...)
    plotsample!(f, method, sample)
    return f
end

"""
    plotsample!(f, method, sample)

Plot a `sample` of a `method` type on a Makie.jl figure or
axis `f`. See also [`plotsample`](#).

Part of the plotting interface for learning methods.
"""
function plotsample! end

"""
    plotxy(method, (x, y))
"""
function plotxy(method, x, y; figkwargs...)
    f = defaultfigure(; figkwargs...)
    plotxy!(f, method, x, y)
    return f
end
"""
    plotxy!(f, method, x, y)
"""
function plotxy! end

function plotprediction(method, x, ŷ, y; figkwargs...)
    f = defaultfigure(; figkwargs...)
    plotprediction!(f, method, x, ŷ, y)
    return f
end

"""
    plotprediction(method, x, ŷ, y)
    plotprediction!(f, method, x, ŷ, y)

Plot a comparison of model output `ŷ` with the ground truth `y` on
Makie.jl figure or axis `f`. `x` is the model input.
"""
function plotprediction! end

"""
    plotpredictions(method, xs, ŷs, ys)

Plot a comparison of batches of model outputs `ŷs` with the ground truths
`ys` on Makie.jl figure or axis `f`. `xs` is a batch of model inputs.

[`plotprediction!`](#) needs to be implemented for `method`.
"""
plotpredictions(method, xs, ŷs, ys) = plotpredictions!(defaultfigure(), method, xs, ŷs, ys)

function plotpredictions!(f, method, xs, ŷs, ys)
    n = size(xs)[end]
    nrows = Int(ceil(sqrt(n)))
    is = Iterators.product(1:nrows, 1:nrows)
    for (i, (x, ŷ, y)) in zip(is, DataLoaders.obsslices((xs, ŷs, ys)))
        plotprediction!(f[i...], method, x, ŷ, y)
    end
    return f
end

"""
    plotbatch(method, xs, ys)

Plot an encoded batch of data in a grid.

[`plotxy!`](#) needs to be implemented for `method`.
"""
plotbatch(method, xs, ys) = plotbatch!(defaultfigure(), method, xs, ys)

function plotbatch!(f, method, xs, ys)
    n = DataLoaders._batchsize(xs, DataLoaders.BatchDimLast())
    nrows = Int(ceil(sqrt(n)))
    is = Iterators.product(1:nrows, 1:nrows)
    for (i, (x, y)) in zip(is, DataLoaders.obsslices((xs, ys)))
        plotxy!(f[i...], method, x, y)
    end
    return f
end
# TODO: implement `plotbatch(method, dataloader; n)`
"""
    plotsamples(method::LearningMethod, samples)

Plot a vector of samples for a `LearningMethod` in a grid. Use
[`plotbatch`](#) instead if you want to plot an encoded batch of data.

[`plotsample!`](#) needs to be implemented for `method`.
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

"""
    imageaxis(f)

Create a `Makie.Axis` with no interactivity, decorations and aspect distortion
suitable for showing images on.
"""
function imageaxis(f; kwargs...)
    ax = Makie.Axis(f; kwargs...)
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


#imageaxis(f::Makie.GridPosition; kwargs...) = imageaxis(f.fig; kwargs...)

# ## Plot recipes


@recipe(PlotImage, image) do scene
    Attributes(
        alpha = 1
    )
end

function Makie.plot!(plot::PlotImage)
    im = plot[:image]
    rim = @lift alphacolor.(copy(rotr90($im)), $(plot.attributes[:alpha]))
    image!(plot, rim; plot.attributes...)
    return plot
end


@recipe(PlotMask, mask, classes) do scene
    Attributes()
end

function Makie.plot!(plot::PlotMask; kwargs...)
    mask = plot[:mask]
    classes = try
        classes = plot[:classes]
    catch
        classes = @lift unique($mask)
    end
    im = @lift maskimage($mask, $classes)
    plotimage!(plot, im; alpha = 1, plot.attributes...)
    return plot
end


function maskimage(mask, classes)
    classtoidx = Dict(class => i for (i, class) in enumerate(classes))
    colors = distinguishable_colors(length(classes), transform=deuteranopic)
    return map(x -> colors[classtoidx[x]], mask)
end

maskimage(mask::AbstractArray{<:Gray{T}}, args...) where T =
    maskimage(reinterpret(T, mask), args...)
maskimage(mask::AbstractArray{<:Normed{T}}, args...) where T =
    maskimage(reinterpret(T, mask), args...)


_toindex(v) = CartesianIndex(Tuple(round.(Int, v)))
_boxIs(I, r) = I-(r*CartesianIndex(1, 1)):I+(r*CartesianIndex(1, 1))
function _drawkeypoint!(img, v; r = 2, c = RGB(0, 0, 1))
    Is = _boxIs(_toindex(v), r)
    for I in Is
        checkbounds(Bool, img, I) && (img[I] = c)
    end
    return img
end
