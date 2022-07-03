
"""
    Keypoints{N}(sz) <: Block

A block representing an array of size `sz` filled with keypoints of type
`SVector{N}`.
"""
struct Keypoints{N, M} <: Block
    sz::NTuple{M, DimSize}
end
Keypoints{N}(n::Int) where {N} = Keypoints{N, 1}((n,))
Keypoints{N}(sz::Tuple) where {N} = Keypoints{N, length(sz)}(sz)

function checkblock(b::Keypoints{N, M},
                    ks::AbstractArray{<:Union{<:SVector{N}, Nothing}, M}) where {M, N}
    return checksize(b.sz, size(ks))
end

mockblock(block::Keypoints{N}) where {N} = mockarray(SVector{N, Float32}, block.sz)

# ## Visualization

function showblock!(io, ::ShowText, block::Keypoints{2}, obs)
    print(io, UnicodePlots.scatterplot(first.(obs), last.(obs), marker = :cross))
end

function showblock!(io, ::ShowText, block::Bounded{2, <:Keypoints{2}}, obs)
    h, w = block.size
    plot = UnicodePlots.scatterplot(first.(obs), last.(obs),
                                    xlim = (0, w), ylim = (0, h), marker = :cross)
    print(io, plot)
end

@testset "Keypoints [block]" begin
    block = Keypoints{2}((10, 10))
    @test checkblock(block, rand(SVector{2}, 10, 10))

    block = Keypoints{2}((10, :))
    @test checkblock(block, rand(SVector{2}, 10, 10))

    ks = map(k -> (rand() > 0.5) ? k : nothing, rand(SVector{2}, 10, 10))
    @test checkblock(block, ks)
end
