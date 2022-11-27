
"""
    ConvFeatures{N}(n) <: Block
    ConvFeatures(n, size)

Block representing features from a convolutional neural network backbone
with `n` feature channels and `N` spatial dimensions.
"""
struct ConvFeatures{N} <: Block
    n::Int
    size::NTuple{N, DimSize}
end

ConvFeatures{N}(n) where {N} = ConvFeatures{N}(n, ntuple(_ -> :, N))

function FastAI.checkblock(block::ConvFeatures{N}, a::AbstractArray{T, M}) where {M, N, T}
    M == N + 1 || return false
    return checksize(block.size, size(a))
end

function FastAI.mockblock(block::ConvFeatures)
    rand(Float32, map(l -> l isa Colon ? 8 : l, block.size)..., block.n)
end
