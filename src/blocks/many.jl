
# ## Many

"""
    Many(block) <: WrapperBlock

`Many` indicates that you can variable number of data instances for
`block`. Consider a bounding box detection task where there may be any
number of targets in an image and this number varies for different
samples. The blocks `(Image{2}(), BoundingBox{2}()` imply that there is exactly
one bounding box for every image, which is not the case. Instead you
would want to use `(Image{2}(), Many(BoundingBox{2}())`.
"""
struct Many{B<:AbstractBlock} <: WrapperBlock
    block::B
end

FastAI.checkblock(many::Many, datas) =
    all(checkblock(wrapped(many), data) for data in datas)
FastAI.mockblock(many::Many) = [mockblock(wrapped(many)), mockblock(wrapped(many))]

function FastAI.encode(enc::Encoding, ctx, many::Many, datas)
    return map(datas) do data
        encode(enc, ctx, wrapped(many), data)
    end
end

function FastAI.decode(enc::Encoding, ctx, many::Many, datas)
    return map(datas) do data
        decode(enc, ctx, wrapped(many), data)
    end
end


# ## Tests

@testset "`Many`" begin
    enc = ImagePreprocessing()
    block = Many(Image{2}())
    FastAI.testencoding(enc, block)
end
