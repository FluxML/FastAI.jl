"""
    blocklossfn(predblock, yblock)

Construct a loss function that compares a batch of model outputs
`ŷs` and encoded targets `ys` and returns a scalar loss.

For example for `block = OneHotLabel(classes)` (i.e. an encoded
`Label(classes)`), we have
`blocklossfn(block, block) == Flux.Losses.logitcrossentropy`.
"""
function blocklossfn end


blocklossfn(predblock) = blocklossfn(predblock, predblock)
