
# ## Many

"""
    Many(block) <: WrapperBlock

`Many` indicates that you can variable number of instances for
`block`. Consider a bounding box detection task where there may be any
number of targets in an image and this number varies for different
samples. The blocks `(Image{2}(), BoundingBox{2}()` imply that there is exactly
one bounding box for every image, which is not the case. Instead you
would want to use `(Image{2}(), Many(BoundingBox{2}())`.
"""
struct Many{B <: AbstractBlock} <: WrapperBlock
    block::B
end

FastAI.checkblock(many::Many, obss) = all(checkblock(parent(many), obs) for obs in obss)
FastAI.mockblock(many::Many) = [mockblock(parent(many)) for _ in 1:rand(1:3)]

function FastAI.encode(enc::Encoding, ctx, many::Many, obss)
    return map(obss) do obs
        encode(enc, ctx, parent(many), obs)
    end
end

function FastAI.decode(enc::Encoding, ctx, many::Many, obss)
    return map(obss) do obs
        decode(enc, ctx, parent(many), obs)
    end
end

# ## Tests

@testset "Many [block]" begin
    enc = OneHot()
    block = Many(Label(1:10))
    FastAI.testencoding(enc, block)
end
