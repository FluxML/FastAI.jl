"""
    Bounded(block, size) <: [WrapperBlock](#)

A wrapper block for annotating spatial data blocks with
size information for their spatial bounds. As an example,
`Image{2}()` doesn't carry any size information since it
supports variable-size images, but sometimes it can be
useful to have the exact size as information where it can
be known.
"""
struct Bounded{N, B<:AbstractBlock} <: WrapperBlock
    block::B
    size::NTuple{N, Int}
end


function Bounded(bounded::Bounded{M}, size::NTuple{N, Int}) where {N, M}
    N == M || error("Cannot rewrap a `Bounded` with different dimensionalities $N and $M")
    Bounded(wrapped(bounded), size)
end


InlineTest.@testset "Bounded [block, wrapper]" begin
    @test_nowarn Bounded(Image{2}(), (16, 16))
    bounded = Bounded(Image{2}(), (16, 16))
    @test Bounded(bounded, (16, 16)) == bounded
end
