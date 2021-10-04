

"""
    showblockinterpretable(backend, encodings, block, data)

Decode `block` successively by applying `encodings` until a block is gotten
that can be shown by `backend`. Useful to visualize encoded data that
is not directly interpretable, for example an `Image{2}` representing an
encoded `Image`.

## Examples

```julia
using FastAI
encodings = (ImagePreprocessing(),)
block = ImageTensor{2}(3)
x = FastAI.mockblock(block)

showblockinterpretable(ShowText(), encodings, block, x)  # will decode to an `Image`
@test_throws showblock(ShowText(), encodings, block, x)  # will error
```

"""
function showblockinterpretable(backend::ShowBackend, encodings, block, data)
    block_, data_ = decodewhile(
        block -> !_isshowable(backend, block),
        encodings,
        Validation(),
        block,
        data)
    @show block_
    return showblock(backend, block_, data_)
end


"""
    showblocksinterpretable(backend, encodings, block, datas)

Multi-sample version [`showblockinterpretable`](#).
"""
function showblocksinterpretable(backend::ShowBackend, encodings, block, data::AbstractVector)
    blockdatas_ = [decodewhile(
        block -> !_isshowable(backend, block),
        encodings,
        Validation(),
        block,
        data) for data in datas]
    block_ = first(first(blockdatas))
    datas_ = last.(blockdatas)
    return showblock(backend, block_, data_)
end


# Helpers


function _isshowable(backend::S, block::B) where {S<:ShowBackend, B<:AbstractBlock}
    hasmethod(FastAI.showblock!, (Any, S, B, Any))
end

"""
    decodewhile(f, encodings, ctx, block, data) -> (block', data')

Decode `block` by successively applying `encodings` to decode in
reverse order until `f(block') == false`.

"""
function decodewhile(f, encodings, ctx, block::FastAI.AbstractBlock, data)
    if f(block)
        return decodewhile(
            f,
            encodings[1:end-1],
            ctx,
            decodedblock(encodings[end], block, true),
            decode(encodings[end], ctx, block, data),
        )
    else
        return (block, data)
    end
end

function decodewhile(f, encodings, ctx, blocks::Tuple, datas::Tuple)
    results = Tuple(decodewhile(f, encodings, ctx, block, data)
                for (block, data) in zip(blocks, datas))
    blocks = first.(results)
    datas = last.(results)
    return blocks, datas
end
