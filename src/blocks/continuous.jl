
"""
    Continuous(size) <: Block

`Block` for collections of numbers. `obs` is a valid observation
if it's length is `size` and contains `Number`s.
"""
struct Continuous <: Block
    size::Int
end

function checkblock(block::Continuous, x)
    block.size == length(x) && eltype(x) <: Number
end

mockblock(block::Continuous) = rand(block.size)


function blocklossfn(outblock::Continuous, yblock::Continuous)
    outblock.size == yblock.size || error("Sizes of $outblock and $yblock differ!")
    return Flux.Losses.mse
end
