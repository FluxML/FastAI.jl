
function plotblock! end

abstract type PlotContext end

struct Plot2D <: PlotContext end

plotcontext(::Image{N}) = PlotND{N}()
plotcontext(::Mask{N}) = PlotNDOverlay{N}()
plotcontext(::Label{N}) = PlotText{N}()
plotcontext(::LabelMulti{N}) = PlotText{N}()
plotcontext(blocks::Tuple) = plotcontext(blocks)


# plotsample!

function plotsample!(f, method::BlockMethod, input, target)
    contexts = plotcontext(method.blocks)
    plotsample!(f, contexts, method.blocks, (input, target))
end

function plotsample!(f, ctxs::Tuple{PlotND, PlotText}, blocks, datas)
    f[1, 1] = ax = imageaxis(f)
    plotblock!(ax, blocks[1], datas[1])  # plots an N-D image
    plotblock!(ax, blocks[2], datas[2])  # appends to axis title
end

function plotsample!(f, ctxs::Tuple{PlotND{N}, PlotNDOverlay{N}}, blocks, datas) where N
    f[1, 1] = ax = imageaxis(f)
    plotblock!(ax, blocks[1], datas[1])  # plots an N-D image
    plotblock!(ax, blocks[2], datas[2])  # overlays transparently over image
end

function plotsample!(f, ctxs::Tuple{PlotND{N}, PlotND{N}}, blocks, datas) where N
    f[1, 1] = ax1 = imageaxis(f)
    plotblock!(ax1, blocks[1], datas[1])  # plots an image
    f[1, 2] = ax2 = imageaxis(f)
    plotblock!(ax, blocks[2], datas[2])  # plots an image
end

# plotxy!

function plotsample!(f, method::BlockMethod, input, target)
    contexts = plotcontext(method.blocks)
    plotsample!(f, contexts, method.blocks, (input, target))
end

function plotprediction!(f, ctxs::Tuple{Plot2D, PlotText, PlotText}, blocks, datas)
    # image with title showing ground truth and prediction
end

function plotxy!(f, ctxs::Tuple{Plot2D, PlotText, PlotText}, blocks, datas)
    # image with title showing ground truth and prediction
end

function plotprediction!(f, xctx::Plot2D, yctx::Plot2D, blocks, datas)
    xblock, ŷblock, yblock = blocks
    x, ŷ, y = datas
    f[1, 1] = ax1 = imageaxis(f, title="Image")
    f[2, 1] = ax2 = imageaxis(f, title="Pred")
end


function plotxy!(f, method::BlockMethod, x, y)
    xyblocks = encodedblock(method.encodings, method.blocks)
    input, target = decode(method.encodings, xyblocks, (x, y))
    plotsample!(f, method, input, target)
end
