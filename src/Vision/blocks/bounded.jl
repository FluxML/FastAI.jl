"""
    Bounded(block, size) <: WrapperBlock

A [`WrapperBlock`](#) for annotating spatial data blocks with
size information for their spatial bounds. As an example,
`Image{2}()` doesn't carry any size information since it
supports variable-size images, but sometimes it can be
useful to have the exact size as information where it can
be known.

Encoding using [`ProjectiveTransforms`](#) returns `Bounded`s since it
crops any input to the same size.

## Examples

```julia
block = Image{2}()  # a 2D-image block with out size informatio
wrapper = Bounded(Image{2}(), (128, 128))  # a 2D-image block with fixed size

@test checkblock(block, rand(10, 10))
@test !checkblock(wrapper, rand(10, 10))  # Expects size `(128, 128)`
```

Wrapping a `Bounded` into another `Bounded` with the same dimensionality
will update the bounds:

```julia
block = Image{2}()
Bounded(Bounded(block, (16, 16)), (8, 8)) == Bounded(block, (8, 8))
```



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
