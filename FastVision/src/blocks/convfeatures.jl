
"""
    ConvFeatures{N}(n) <: Block
    ConvFeatures(n, size)

Block representing features from a convolutional neural network backbone
with `n` feature channels and `N` spatial dimensions.

For example, a 2D ResNet's convolutional layers may produce a `h`x`w`x`ch` output
that is passed further to the classifier head.

## Examples

A feature block with 512 channels and variable spatial dimensions:

```julia
FastVision.ConvFeatures{2}(512)
# or equivalently
FastVision.ConvFeatures(512, (:, :))
```

A feature block with 512 channels and fixed spatial dimensions:

```julia
FastVision.ConvFeatures(512, (4, 4))
```

"""
struct ConvFeatures{N} <: Block
    n::Int
    size::NTuple{N, DimSize}
end

ConvFeatures{N}(n) where {N} = ConvFeatures{N}(n, ntuple(_ -> :, N))

function FastAI.checkblock(block::ConvFeatures{N}, a::AbstractArray{T, M}) where {M, N, T}
    M == N + 1 || return false
    return checksize(block.size, size(a)[begin:N])
end

function FastAI.mockblock(block::ConvFeatures)
    rand(Float32, map(l -> l isa Colon ? 8 : l, block.size)..., block.n)
end


@testset "ConvFeatures [block]" begin
    @test ConvFeatures(16, (:, :)) == ConvFeatures{2}(16)
    @test checkblock(ConvFeatures(16, (:, :)), rand(Float32, 2, 2, 16))
    @test checkblock(ConvFeatures(16, (:, :)), rand(Float32, 3, 2, 16))
    @test checkblock(ConvFeatures(16, (2, 2)), rand(Float32, 2, 2, 16))
    @test !checkblock(ConvFeatures(16, (2, :)), rand(Float32, 3, 2, 16))
end
