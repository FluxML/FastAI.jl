"""
    blocklossfn(predblock, yblock)

Construct a loss function that compares a batch of model outputs
`ŷs` and encoded targets `ys` and returns a scalar loss.

For example for `block = OneHotTensor{1}(classes)` (i.e. an encoded
`Label(classes)`), we have
`blocklossfn(block, block) == Flux.Losses.logitcrossentropy`.
"""
function blocklossfn end


blocklossfn(predblock) = blocklossfn(predblock, predblock)

function blocklossfn(outblock::OneHotTensor{0}, yblock::OneHotTensor{0})
    outblock.classes == yblock.classes || error("Classes of $outblock and $yblock differ!")
    return Flux.Losses.logitcrossentropy
end

function blocklossfn(outblock::OneHotTensorMulti{0}, yblock::OneHotTensorMulti{0})
    outblock.classes == yblock.classes || error("Classes of $outblock and $yblock differ!")
    return Flux.Losses.logitbinarycrossentropy
end

function blocklossfn(outblock::OneHotTensor{N}, yblock::OneHotTensor{N}) where N
    outblock.classes == yblock.classes || error("Classes of $outblock and $yblock differ!")
    return segmentationloss
end


function segmentationloss(ypreds, ys; kwargs...)
    # Has to be reshaped to 3D array since `logitcrossentropy(...; dims = 3)`
    # doesn't work on GPU
    sz = size(ypreds)
    ypreds = reshape(ypreds, :, sz[end-1], sz[end])
    ys = reshape(ys, :, size(ys, 3), size(ys, 4))
    Flux.Losses.logitcrossentropy(ypreds, ys; dims = 2, kwargs...)
end

function blocklossfn(outblock::KeypointTensor{N}, yblock::KeypointTensor{N}) where N
    outblock.sz == yblock.sz || error("Sizes of $outblock and $yblock differ!")
    return Flux.Losses.mse
end

function blocklossfn(outblock::Continuous, yblock::Continuous)
    outblock.n == yblock.n || error("Sizes of $outblock and $yblock differ!")
    return Flux.Losses.mse
end
