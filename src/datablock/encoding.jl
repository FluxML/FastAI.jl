
"""
    abstract type Encoding

Transformation of `Block`s. Can encode some `Block`s ([`encode`]), and optionally
decode them [`decode`]

## Interface

- `encode(::E, ::Context, block::Block, data)` encodes `block` of `data`.
    The default is to do nothing. This should be overloaded for an encoding `E`,
    concrete `Block` types and possibly a context.
- `decode(::E, ::Context, block::Block, data)` decodes `block` of `data`. This
    should correspond as closely as possible to the inverse of `encode(::E, ...)`.
    The default is to do nothing, as not all encodings can be reversed. This should
    be overloaded for an encoding `E`, concrete `Block` types and possibly a context.
- `encodedblock(::E, block::Block) -> block'` returns the block that is obtained by
    encoding `block` with encoding `E`. This needs to be constant for an instance of `E`,
    so it cannot depend on the sample or on randomness. The default is to return `nothing`,
    meaning the same block is returned and not changed. Encodings that return the same
    block but change the data (e.g. `ProjectiveTransforms`) should return `block`.
- `decodedblock(::E, block::Block) -> block'` returns the block that is obtained by
    decoding `block` with encoding `E`. This needs to be constant for an instance of `E`,
    so it cannot depend on the sample or on randomness. The default is to return `nothing`,
    meaning the same block is returned and not changed.
- `encode!(buf, ::E, ::Context, block::Block, data)` encodes `data` inplace.
- `decode!(buf, ::E, ::Context, block::Block, data)` decodes `data` inplace.

"""
abstract type Encoding end


"""
    fillblock(inblocks, outblocks)

Replaces all `nothing`s in outblocks with the corresponding block in `inblocks`.
`outblocks` may be obtained by
"""
fillblock(inblocks::Tuple, outblocks::Tuple) =
    map(fillblock, inblocks, outblocks)
fillblock(inblock::AbstractBlock, ::Nothing) = inblock
fillblock(inblocks::Tuple, ::Nothing) = inblocks
fillblock(::AbstractBlock, outblock::AbstractBlock) = outblock

encodedblockfilled(enc, block) = fillblock(block, encodedblock(enc, block))
decodedblockfilled(enc, block) = fillblock(block, decodedblock(enc, block))
# ## `encode` methods

# By default an encoding doesn't change the data
encode(encoding::Encoding, ctx, block::Block, data; kwargs...) =
    isempty(kwargs) ? data : encode(encoding, ctx, block, data)

# By default, a tuple of encodings encodes by encoding the data one encoding
# after the other
"""
    encode(encoding, context, block, data)
    encode(encoding, context, blocks, data)
    encode(encodings, context, blocks, data)
"""
function encode(encodings::NTuple{N, Encoding}, context, blocks, data) where N
    for encoding in encodings
        data = encode(encoding, context, blocks, data)
        blocks = encodedblockfilled(encoding, blocks)
    end
    return data
end

# By default, an encoding encodes every element in a tuple separately
function encode(encoding::Encoding, context, blocks::Tuple, datas::Tuple)
    @assert length(blocks) == length(datas)
   return map(
        (block, data) -> encode(encoding, context, block, data),
        blocks, datas
    )
end

# Named tuples of data are handled like tuples, but the keys are preserved
function encode(encoding::Encoding, context, blocks::NamedTuple, datas::NamedTuple)
    @assert length(blocks) == length(datas)
    return NamedTuple(
        zip(keys(datas), encode(encoding, context, values(blocks), values(datas))),
    )
end

# ## `decode` methods

# By default an encoding doesn't change the data when decoding
decode(encoding::Encoding, ctx, block::Block, data; kwargs...) =
    isempty(kwargs) ? data : decode(encoding, ctx, block, data)

# By default, a tuple of encodings decodes by decoding the data one encoding
# after the other, with encodings iterated in reverse order
function decode(encodings::NTuple{N,Encoding}, context, blocks, data) where {N}
    for encoding in Iterators.reverse(encodings)
        data = decode(encoding, context, blocks, data)
        blocks = decodedblockfilled(encoding, blocks)
    end
    return data
end


# By default, an encoding decodes every element in a tuple separately
function decode(encoding::Encoding, context, blocks::Tuple, datas::Tuple)
    @assert length(blocks) == length(datas)
    return map(
        (block, data) -> decode(encoding, context, block, data),
        blocks, datas
    )
end

# Named tuples of data are handled like tuples, and the keys are preserved
function decode(encoding::Encoding, context, blocks::NamedTuple, datas::NamedTuple)
    @assert length(blocks) == length(datas)
    return NamedTuple(
        zip(keys(datas), decode(encoding, context, values(blocks), values(datas))),
    )
end


"""
    encodedblock(encoding, block)
    encodedblock(encoding, blocks)
    encodedblock(encodings, blocks)

Return the block that is obtained by encoding `block` with encoding `E`.
This needs to be constant for an instance of `E`, so it cannot depend on the
sample or on randomness. The default is to return `nothing`,
meaning the same block is returned and not changed. Encodings that return the same
block but change the data (e.g. `ProjectiveTransforms`) should return `block`.
"""
encodedblock(::Encoding, ::Block) = nothing

function encodedblock(encoding::Encoding, blocks::Tuple)
    map(block -> encodedblock(encoding, block), blocks)
end
function encodedblock(encodings::Tuple, blocks)
    encoded = false
    for encoding in encodings
        encoded = encoded || !isnothing(encodedblock(encoding, blocks))
        blocks = encodedblockfilled(encoding, blocks)
    end
    return encoded ? blocks : nothing
end

"""
    decodedblock(encoding, block)
    decodedblock(encoding, blocks)
    decodedblock(encodings, blocks)

Return the block that is obtained by decoding `block` with encoding `E`.
This needs to be constant for an instance of `E`, so it cannot depend on the
sample or on randomness. The default is to return `nothing`,
meaning the same block is returned and not changed. Encodings that return the same
block but change the data when decoding should return `block`.
"""
decodedblock(::Encoding, ::Block) = nothing

function decodedblock(encoding::Encoding, blocks::Tuple)
    map(block -> decodedblock(encoding, block), blocks)
end
function decodedblock(encodings, blocks)
    decoded = false
    for encoding in Iterators.reverse(encodings)
        decoded = decoded || !isnothing(decodedblock(encoding, blocks))
        blocks = decodedblockfilled(encoding, blocks)
    end
    return decoded ? blocks : nothing
end


"""
    abstract type StatefulEncoding <: Encoding

Encoding that needs to compute some state from the whole sample, even
if it only transforms some of the blocks. This could be random state
for stochastic augmentations that needs to be the same for every block
that is encoded.

The state is created by calling `encodestate(encoding, context, blocks, sample)`
and passed to recursive calls with the keyword argument `state`.
As a result, you need to implement `encode`, `decode`, `encode!`, `decode!` with a
keyword argument `state` that defaults to the above call.

Same goes for `decode`, which should accept a `state` keyword argument defaulting
to `decodestate(encoding, context, blocks, sample)`
"""
abstract type StatefulEncoding <: Encoding end

encodestate(encoding, context, blocks, data) = nothing
decodestate(encoding, context, blocks, data) = nothing

function encode(
    encoding::StatefulEncoding,
    context,
    blocks::Tuple,
    datas::Tuple;
    state = encodestate(encoding, context, blocks, datas),
)
    return map(
        (block, data) -> encode(encoding, context, block, data; state = state),
        blocks, datas)
end

function decode(
    encoding::StatefulEncoding,
    context,
    blocks::Tuple,
    datas::Tuple;
    state = decodestate(encoding, context, blocks, datas),
)

    @assert length(blocks) == length(datas)
    return Tuple(
        decode(encoding, context, block, data; state = state) for
        (block, data) in zip(blocks, datas)
    )
end


"""
    testencoding(encoding, block, data)

Performs some tests that the encoding interface is set up properly for
`encoding` and `block`. Tests that

- `data` is a valid `block`
- `encode` returns a valid `encodedblock(encoding, block)`
- `decode` returns a valid `decodedblock(encoding, encodedblock(encoding, block))`
    and that the block is identical to `block`
"""
function testencoding(encoding, block, data = mockblock(block))
    Test.@testset "Encoding `$(typeof(encoding))` for block `$block`" begin
        # Test that `data` is a valid instance of `block`
        Test.@test checkblock(block, data)
        Test.@test !isnothing(encodedblock(encoding, block))
        outblock = encodedblockfilled(encoding, block)
        outdata = encode(encoding, Training(), block, data)
        # The encoded data should be a valid instance for the `encodedblock`
        Test.@test checkblock(outblock, outdata)

        # Test decoding (if supported) works correctly
        if (outblock isa Tuple)
            for idx in 1:length(outblock)
                inblock = decodedblock(encoding, outblock[idx])
                if !isnothing(inblock)
                    Test.@test wrapped(block[idx]) == wrapped(inblock)
                    indata = decode(encoding, Training(), outblock[idx], outdata[idx])
                    Test.@test checkblock(inblock, indata)
                end
            end
        else
            inblock = decodedblock(encoding, outblock)
            if !isnothing(inblock)
                Test.@test wrapped(block) == wrapped(inblock)
                indata = decode(encoding, Training(), outblock, outdata)
                Test.@test checkblock(inblock, indata)
            end
        end
    end
end
