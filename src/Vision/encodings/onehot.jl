
# Extending `OneHot` for `Mask`

encodedblock(::OneHot, block::Mask{N, T}) where {N, T} = OneHotTensor{N, T}(block.classes)
decodedblock(::OneHot, block::OneHotTensor{N, T}) where {N, T} = Mask{N, T}(block.classes)

function encode(enc::OneHot, context, block::Mask, data)
    tfm = DataAugmentation.OneHot{enc.T}()
    return apply(tfm, DataAugmentation.MaskMulti(data, block.classes)) |> DataAugmentation.itemdata
end

function decode(::OneHot, context, block::OneHotTensor, data)
    Tidx = length(block.classes) >= 255 ? UInt16 : UInt8
    classidxs = reshape(
        map(I -> Tidx(I.I[end]), argmax(data; dims = ndims(data))),
        size(data)[1:end-1])
    return IndirectArray(classidxs, block.classes)
end


# ## Loss function

function blocklossfn(outblock::OneHotTensor{N}, yblock::OneHotTensor{N}) where N
    outblock.classes == yblock.classes || error("Classes of $outblock and $yblock differ!")
    return _segmentationloss
end


# Arrays have to be reshaped to 3D array since
# `logitcrossentropy(...; dims = 3)` doesn't work on GPU:

function _segmentationloss(ypreds, ys; kwargs...)
    sz = size(ypreds)
    ypreds = reshape(ypreds, :, sz[end-1], sz[end])
    ys = reshape(ys, :, size(ys, 3), size(ys, 4))
    Flux.Losses.logitcrossentropy(ypreds, ys; dims = 2, kwargs...)
end
