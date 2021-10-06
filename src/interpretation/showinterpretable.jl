

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
    res = decodewhile(
        block -> !isshowable(backend, block),
        encodings,
        Validation(),
        block,
        data)
    isnothing(res) && error("Could not decode to an interpretable block representation!")
    block_, data_ = res
    showblock(backend, block_, data_)
end


"""
    showblocksinterpretable(backend, encodings, block, datas)

Multi-sample version [`showblockinterpretable`](#).
"""
function showblocksinterpretable(backend::ShowBackend, encodings, block, datas::AbstractVector)
    blockdatas = [decodewhile(
        block -> !isshowable(backend, block),
        encodings,
        Validation(),
        block,
        data) for data in datas]
    isnothing(res) && error("Could not decode to an interpretable block representation!")
    block_ = first(first(blockdatas))
    datas_ = last.(blockdatas)
    showblocks(backend, block_, datas_)
end


# Helpers


function isshowable(backend::S, block::B) where {S<:ShowBackend, B<:AbstractBlock}
    hasmethod(FastAI.showblock!, (Any, S, B, Any))
end

"""
    decodewhile(f, encodings, ctx, block, data) -> (block', data')

Decode `block` by successively applying `encodings` to decode in
reverse order until `f(block') == false`.
"""
function decodewhile(f, encodings, ctx, block::AbstractBlock, data)
    encodings === () && return nothing
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
    encodings === () && return nothing
    results = Tuple(decodewhile(f, encodings, ctx, block, data)
                for (block, data) in zip(blocks, datas))
    any(isnothing, results) && return nothing
    blocks = first.(results)
    datas = last.(results)
    return blocks, datas
end


function decodewhile(f, encodings, ctx, (title, block)::Pair, data)
    encodings === () && return nothing
    res = decodewhile(f, encodings, ctx, block, data)
    isnothing(res) && return nothing
    block_, data_ = res
    (title => block_, data_)
end
