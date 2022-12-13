
"""
    abstract type Encoding

Transformation of `Block`s. Can encode some `Block`s ([`encode`]), and optionally
decode them [`decode`]. `Encoding`s describe data transformations that are applied
to `Block` data. Together `Encoding`s and `Block`s, are used to construct complex
data preprocessing pipelines for training loops.

Encodings operate on two levels:

- On the value level, an encoding transforms an observation.
- On the `Block` level, applying an encoding to a block tells you what the output
    block is. For example, the [`OneHot`](#) encoding turns a [`Label`](#) block
    into a [`OneHotLabel`](#) block.

    By introspecting the block-level transformation

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

invertible(enc::Encoding, block::AbstractBlock) =
    !isnothing(decodedblock(enc, encodedblockfilled(enc, block)))

encodingname(::E) where E<:Encoding = nameof(E)
encodingname(t::Tuple) = map(encodingname, t)

"""
    fillblock(inblocks, outblocks)

Replaces all `nothing`s in outblocks with the corresponding block in `inblocks`.
`outblocks` may be obtained by
"""
fillblock(inblocks::Tuple, outblocks::Tuple) = map(fillblock, inblocks, outblocks)
fillblock(inblock::AbstractBlock, ::Nothing) = inblock
fillblock(inblocks::Tuple, ::Nothing) = inblocks
fillblock(::AbstractBlock, outblock::AbstractBlock) = outblock

encodedblockfilled(enc, block) = fillblock(block, encodedblock(enc, block))
decodedblockfilled(enc, block) = fillblock(block, decodedblock(enc, block))
# ## `encode` methods

# By default an encoding doesn't change the data
function encode(encoding::Encoding, ctx, block::Block, obs; kwargs...)
    isempty(kwargs) ? obs : encode(encoding, ctx, block, obs)
end

# By default, a tuple of encodings encodes by encoding the data one encoding
# after the other
"""
    encode(encoding, context, block, obs)
    encode(encoding, context, blocks, obss)
    encode(encodings, context, blocks, obss)

Apply one or more [`Encoding`](#)s to observation(s) `obs`.
"""
function encode(encodings::NTuple{N, Encoding}, context, blocks, data) where {N}
    for encoding in encodings
        data = encode(encoding, context, blocks, data)
        blocks = encodedblockfilled(encoding, blocks)
    end
    return data
end

# By default, an encoding encodes every element in a tuple separately

function encode(encoding::Encoding, context, blocks::Tuple, obss::Tuple)
    @assert length(blocks) == length(obss)
    return map((block, obs) -> encode(encoding, context, block, obs),
               blocks, obss)
end

# Named tuples of data are handled like tuples, but the keys are preserved
function encode(encoding::Encoding, context, blocks::NamedTuple, obss::NamedTuple)
    @assert length(blocks) == length(obss)
    return NamedTuple(zip(keys(obss),
                          encode(encoding, context, values(blocks), values(obss))))
end

# ## `decode` methods

# By default an encoding doesn't change the data when decoding
function decode(encoding::Encoding, ctx, block::Block, obs; kwargs...)
    isempty(kwargs) ? obs : decode(encoding, ctx, block, obs)
end

# By default, a tuple of encodings decodes by decoding the data one encoding
# after the other, with encodings iterated in reverse order
function decode(encodings::NTuple{N, Encoding}, context, blocks, obs) where {N}
    for encoding in Iterators.reverse(encodings)
        obs = decode(encoding, context, blocks, obs)
        blocks = decodedblockfilled(encoding, blocks)
    end
    return obs
end

# By default, an encoding decodes every element in a tuple separately
function decode(encoding::Encoding, context, blocks::Tuple, obss::Tuple)
    @assert length(blocks) == length(obss)
    return map((block, obs) -> decode(encoding, context, block, obs),
               blocks, obss)
end

# Named tuples of data are handled like tuples, and the keys are preserved
function decode(encoding::Encoding, context, blocks::NamedTuple, obss::NamedTuple)
    @assert length(blocks) == length(obss)
    return NamedTuple(zip(keys(obss),
                          decode(encoding, context, values(blocks), values(obss))))
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

encodestate(encoding, context, blocks, obs) = nothing
decodestate(encoding, context, blocks, obs) = nothing

function encode(encoding::StatefulEncoding,
                context,
                blocks::Tuple,
                obss::Tuple;
                state = encodestate(encoding, context, blocks, obss))
    return map((block, obs) -> encode(encoding, context, block, obs; state = state),
               blocks, obss)
end

function decode(encoding::StatefulEncoding,
                context,
                blocks::Tuple,
                obss::Tuple;
                state = decodestate(encoding, context, blocks, obss))
    @assert length(blocks) == length(obss)
    return Tuple(decode(encoding, context, block, obs; state = state)
                 for
                 (block, obs) in zip(blocks, obss))
end

"""
    testencoding(encoding, block[, obs])

Performs some tests that the encoding interface is set up properly for
`encoding` and `block`. Tests that

- `obs` is a valid instance `block`
- `encode` returns a valid `encodedblock(encoding, block)`
- `decode` returns a valid `decodedblock(encoding, encodedblock(encoding, block))`
    and that the block is identical to `block`
"""
function testencoding(encoding, block, obs = mockblock(block))
    Test.@testset "Encoding `$(typeof(encoding))` for block `$block`" begin
        inv = invariant_encoding(encoding, block)
        @test_nowarn inv(Exception, obs)
    end
    return
    Test.@testset "Encoding `$(typeof(encoding))` for block `$block`" begin
        # Test that `obs` is a valid instance of `block`
        Test.@test checkblock(block, obs)
        Test.@test !isnothing(encodedblock(encoding, block))
        outblock = encodedblockfilled(encoding, block)
        outobs = encode(encoding, Training(), block, obs)
        # The encoded data should be a valid instance for the `encodedblock`
        Test.@test checkblock(outblock, outobs)

        # Test decoding (if supported) works correctly
        if (outblock isa Tuple)
            for idx in 1:length(outblock)
                inblock = decodedblock(encoding, outblock[idx])
                if !isnothing(inblock)
                    Test.@test wrapped(block[idx]) == wrapped(inblock)
                    inobs = decode(encoding, Training(), outblock[idx], outobs[idx])
                    Test.@test checkblock(inblock, inobs)
                end
            end
        else
            inblock = decodedblock(encoding, outblock)
            if !isnothing(inblock)
                Test.@test wrapped(block) == wrapped(inblock)
                inobs = decode(encoding, Training(), outblock, outobs)
                Test.@test checkblock(inblock, inobs)
            end
        end
    end
end


function invariant_encoding(encoding, block; context = Validation(), encvar = "encoding", blockvar = "block", obsvar = "obs")
    B = blockname(block)
    E = encodingname(encoding)

    encobs(obs) = encode(encoding, context, block, obs)
    encblock() = encodedblock(encoding, block)
    encblockfilled() = encodedblockfilled(encoding, block)
    decblock() = decodedblock(encoding, encblockfilled())
    decblockfilled() = decodedblockfilled(encoding, encblockfilled())
    decobs(obs) = decode(encoding, context, encblockfilled(), encobs(obs))

    return invariant(
        "Encoding `$E` is implemented for block `$B`",
        [
            invariant_checkblock(block; blockvar, obsvar, description="""
                Before checking that the encoding is properly implemented for the block,
                we need to check that the observation `$obsvar` is a valid instance of
                `$blockvar <: $B.`

                """ |> md),
            invariant("`$encvar <: $E` is implemented for `$blockvar <: $B`") do _
                if isnothing(encblock())
                    return """Expected `encodedblock($encvar::$E, $blockvar::$B)` to return a block,
                    indicating that the encoding does transform observations for block
                    `$blockvar`. Instead, it returned `nothing` which indicates that the
                    encoding does not transform observations of block `$B`.

                    If the encoding should modify the block, this may mean that a method
                    for `FastAI.encodedblock` is missing. To fix this, implement the following
                    method, returning a block from it:

                    ```julia
                    FastAI.encodedblock(::$E, ::$B)
                    ```
                    """ |> md
                end
            end,
            invariant(invariant_checkblock(encblockfilled();
                                           blockvar = "enc$blockvar", obsvar = "enc$obsvar");
                      title = "Encoded `$obsvar` is a valid instance of encoded `$blockvar`",
                      inputfn = encobs,
                      description = """The encoded observation
                      `encobs = encode($encvar, $context, $blockvar, $obsvar)`
                      should be a valid observation for the encoded block
                      `enc$blockvar = encodedblock($encvar, $blockvar)`.
                      """ |> md),
            invariant(
                "If `$encvar <: $E` is invertible, decoding is implemented",
                [
                    invariant("`$encvar <: $E` is not invertible") do _
                        if invertible(encoding, block)
                            return "The encoding *is* invertible." |> md
                        end
                    end,
                    invariant("Decoding is implemented") do obs
                        if isnothing(decblock())
                            return """
                                `decodedblock(encoding, encodedblock(encoding, block))` returned
                                `nothing`, indicating that the encoding `$encvar <: $E` does not implement
                                a decoding step.

                                This can mean that either the encoding is not invertible, or `decodedblock`
                                was not implemented for block `$blockvar <: $B`. To fix this, implement EITHER

                                - `decodedblock(::$E, ::$B)` and return a non-`nothing` block value from it; OR
                                - `invertible(::$E, ::$B) = false` if the encoding is not invertible.
                                """ |> md
                        end
                    end,
                    invariant_checkblock(
                        decblockfilled(),
                        inputfn = decobs,
                        title = "Decoded `encobs` is a valid instance of `$blockvar <: $B`",
                        description="""Decoding the encoded observation should return a valid observation.""")
                ],
                any,
            )
        ],
        description = """
            This invariant checks that the encoding `$encvar <: $E` is properly implemented
            for `$blockvar <: $B`. Type `?FastAI.Encoding` to get an overview of the
            interface for `Encoding`s.
            """ |> md)
end
