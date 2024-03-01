
"""
    showblockinterpretable(backend, encodings, block, obs)

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
function showblockinterpretable(backend::ShowBackend, encodings, block, obs)
    invariant_checkblock(block)(Exception, obs)
    res = decodewhile(block -> !isshowable(backend, block),
                      encodings,
                      Validation(),
                      block,
                      obs)
    isnothing(res) && error("Could not decode to an interpretable block representation!")
    block_, obs_ = res
    showblock(backend, block_, obs_)
end

"""
    showblocksinterpretable(backend, encodings, block, obss)

Multi-sample version [`showblockinterpretable`](#).
"""
function showblocksinterpretable(backend::ShowBackend, encodings, block,
                                 obss::AbstractVector)
    blockobss = [decodewhile(block -> !isshowable(backend, block),
                             encodings,
                             Validation(),
                             block,
                             obs) for obs in obss]
    any(isnothing, blockobss) &&
        error("Could not decode to an interpretable block representation!")
    block_ = first(first(blockobss))
    obss_ = last.(blockobss)
    showblocks(backend, block_, obss_)
end

# Helpers

function isshowable(backend::S, block::B) where {S <: ShowBackend, B <: AbstractBlock}
    hasmethod(FastAI.showblock!, (Any, S, B, typeof(mockblock(block))))
end

"""
    decodewhile(f, encodings, ctx, block, obs) -> (block', obs')

Decode `block` by successively applying `encodings` to decode in
reverse order until `f(block') == false`.
"""
function decodewhile(f, encodings, ctx, block::AbstractBlock, obs)
    if f(block)
        encodings === () && return nothing
        return decodewhile(f,
                           encodings[1:(end - 1)],
                           ctx,
                           decodedblockfilled(encodings[end], block),
                           decode(encodings[end], ctx, block, obs))
    else
        return (block, obs)
    end
end

function decodewhile(f, encodings, ctx, blocks::Tuple, obss::Tuple)
    encodings === () && return nothing
    results = Tuple(decodewhile(f, encodings, ctx, block, obs)
                    for (block, obs) in zip(blocks, obss))
    any(isnothing, results) && return nothing
    blocks = first.(results)
    obss_ = last.(results)
    return blocks, obss_
end

function decodewhile(f, encodings, ctx, (title, block)::Pair, obs)
    encodings === () && return nothing
    res = decodewhile(f, encodings, ctx, block, obs)
    isnothing(res) && return nothing
    block_, obs_ = res
    (title => block_, obs_)
end
