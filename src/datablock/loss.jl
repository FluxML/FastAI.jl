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

function blocklossfn(outblock::Continuous, yblock::Continuous)
    outblock.size == yblock.size || error("Sizes of $outblock and $yblock differ!")
    return Flux.Losses.mse
end
