
function plotblock! end

abstract type PlotContext end
struct TextContext <: PlotContext end
struct NDContext{N} <: PlotContext end
struct NDOverlayContext{N} <: PlotContext end


struct Plot2D <: PlotContext end

plotcontext(::Image{N}) where N = NDContext{N}()
plotcontext(::Mask{N}) where N = NDOverlayContext{N}()
plotcontext(::Label) = TextContext()
plotcontext(::LabelMulti) = TextContext()
plotcontext(blocks::Tuple) = map(plotcontext, blocks)

function plotblock!(ax, ::Image{2}, data; kwargs...)
    plotimage!(ax, data; kwargs...)
end

function plotblock!(ax, ::Label, data)
    ax.title[] = ax.title[] * string(data)
end

function plotblock!(ax, block::Mask{2}, data; kwargs...)
    plotmask!(ax, data, block.classes; kwargs...)
end


# plotsample!

function plotsample!(f, method::BlockMethod, sample)
    contexts = plotcontext(method.blocks)
    plotsample!(f, contexts, method.blocks, sample)
end

function plotsample!(f, ctxs::Tuple{NDContext{2}, TextContext}, blocks, datas)
    f[1, 1] = ax = imageaxis(f)
    plotblock!(ax, blocks[1], datas[1])  # plots an N-D image
    plotblock!(ax, blocks[2], datas[2])  # appends to axis title
end

function plotsample!(f, ctxs::Tuple{NDContext{2}, NDOverlayContext{2}}, blocks, datas)
    f[1, 1] = ax = imageaxis(f)
    plotblock!(ax, blocks[2], datas[2], alpha = 1.)  # overlays transparently over image
    plotblock!(ax, blocks[1], datas[1], alpha = 0.6)  # plots an N-D image
end

function plotsample!(f, ctxs::Tuple{NDContext{2}, NDContext{2}}, blocks, datas)
    f[1, 1] = ax1 = imageaxis(f)
    plotblock!(ax1, blocks[1], datas[1])  # plots an image
    f[1, 2] = ax2 = imageaxis(f)
    plotblock!(ax, blocks[2], datas[2])  # plots an image
end

# plotxy! decodes and falls back to `plotsample`

function plotxy!(f, method::BlockMethod, x, y)
    xyblocks = encodedblock(method.encodings, method.blocks)
    input, target = decode(method.encodings, Validation(), xyblocks, (x, y))
    plotsample!(f, method, (input, target))
end


function plotprediction!(f, method::BlockMethod, x, ŷ, y)
    xblock, yblock = encodedblock(method.encodings, method.blocks)
    ŷblock = method.outputblock
    blocks = (xblock, ŷblock, yblock)
    input, target_pred, target = decode(method.encodings, Validation(), blocks, (x, ŷ, y))
    inblocks = decodedblock(method.encodings, blocks)
    contexts = plotcontext(inblocks)
    plotprediction!(f, contexts, inblocks, (input, target_pred, target))
end

function plotprediction!(
        f, ::Tuple{NDContext{2}, TextContext, TextContext},
        blocks,
        datas)

    ax = f[1, 1] = imageaxis(f)
    plotblock!(ax, blocks[1], datas[1])
    ax.title[] = ax.title[] * "Prediction: "
    plotblock!(ax, blocks[2], datas[2])
    ax.title[] = ax.title[] * " | True: "
    plotblock!(ax, blocks[3], datas[3])
end


function plotprediction!(
        f,
        ::Tuple{NDContext{2}, NDOverlayContext{2}, NDOverlayContext{2}},
        blocks,
        datas)

    # image with title showing ground truth and prediction
    ax1 = f[1, 1] = imageaxis(f)
    ax1.title[] = "True"
    plotblock!(ax1, blocks[3], datas[3])
    plotblock!(ax1, blocks[1], datas[1], alpha=0.4)


    ax2 = f[1, 2] = imageaxis(f)
    ax2.title[] = "Prediction"
    plotblock!(ax2, blocks[2], datas[2])
    plotblock!(ax2, blocks[1], datas[1], alpha=0.4)
end
