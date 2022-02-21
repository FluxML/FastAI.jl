const DimSize = Union{Int, Colon}

function checksize(targetsz::Tuple, sz::Tuple)
    length(targetsz) == length(sz) || return false
    return all(map(_checksizedim, targetsz, sz))
end


_checksizedim(l1::Int, l2::Int) = l1 == l2
_checksizedim(l1::Colon, l2::Int) = true

mockarray(T, sz) = rand(T, map(l -> l isa Colon ? rand(8:16) : l, sz))


@testset "checksize" begin
    @test checksize((10, 1), (10, 1))
    @test !checksize((100, 1), (10, 1))
    @test checksize((:, :, :), (1, 2, 3))
    @test !checksize((:, :, :), (1, 2))
    @test checksize((10, :, 1), (10, 20, 1))
    @test !checksize((10, :, 2), (10, 20, 1))
end


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
    size::NTuple{N, DimSize}
end


function Bounded(bounded::Bounded{M}, size::NTuple{N, DimSize}) where {N, M}
    N == M || error("Cannot rewrap a `Bounded` with different dimensionalities $N and $M")
    Bounded(wrapped(bounded), size)
end


function checkblock(bounded::Bounded{N}, a::AbstractArray{N}) where N
    return checksize(bounded.size, size(a)) && checkblock(parent(bounded), a)
end

@testset "Bounded [block, wrapper]" begin
    @test_nowarn Bounded(Image{2}(), (16, 16))
    bounded = Bounded(Image{2}(), (16, 16))
    @test checkblock(bounded, rand(16, 16))

    # composition
    @test Bounded(bounded, (16, 16)) == bounded
end
