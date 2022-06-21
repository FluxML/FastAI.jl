
"""
    Named(name, block)

Wrapper `Block` to attach a name to a block. Can be used in conjunction
with [`Only`](#) to apply encodings to specific blocks only.
"""
struct Named{Name, B <: AbstractBlock} <: WrapperBlock
    block::B
end
Named(name::Symbol, block::B) where {B <: AbstractBlock} = Named{name, B}(block)

setwrapped(named::Named{name}, block) where {name} = Named(name, block)

# The name is preserved through encodings and decodings, which is the
# behavior of `propagatewrapper(::W) = PropagateAlways()` so it does
# not need to be overwritten

"""
    Only(fn, encoding)
    Only(BlockType, encoding)
    Only(name, encoding)

Wrapper that applies the wrapped `encoding` to a `block` if
`fn(block) === true`. Instead of a function you can also pass in
a type of block `BlockType` or the `name` of a `Named` block.
"""
struct Only{E <: Encoding} <: StatefulEncoding
    fn::Any
    encoding::E
end

function Only(name::Symbol, encoding::Encoding)
    return Only(Named{name}, encoding)
end

function Only(B::Type{<:AbstractBlock}, encoding::Encoding)
    return Only(block -> block isa B, encoding)
end

function encodedblock(only::Only, block::Block)
    only.fn(block) ? encodedblock(only.encoding, block) : nothing
end
function encodedblock(only::Only, block::WrapperBlock)
    only.fn(block) ? encodedblock(only.encoding, block) : nothing
end
function encodedblock(only::Only, block::Named)
    only.fn(block) ? encodedblock(only.encoding, block) : nothing
end

function decodedblock(only::Only, block::Block)
    inblock = decodedblock(only.encoding, block)
    only.fn(inblock) || return nothing
    return inblock
end
function decodedblock(only::Only, block::WrapperBlock)
    inblock = decodedblock(only.encoding, block)
    only.fn(inblock) || return nothing
    return inblock
end
function decodedblock(only::Only, block::Named)
    inblock = decodedblock(only.encoding, block)
    only.fn(inblock) || return nothing
    return inblock
end

encodestate(only::Only, args...) = encodestate(only.encoding, args...)
decodestate(only::Only, args...) = decodestate(only.encoding, args...)

function encode(only::Only, ctx, block::Block, obs; kwargs...)
    _encode(only, ctx, block, obs; kwargs...)
end
function encode(only::Only, ctx, block::WrapperBlock, obs; kwargs...)
    _encode(only, ctx, block, obs; kwargs...)
end
function _encode(only, ctx, block, obs; kwargs...)
    return only.fn(block) ? encode(only.encoding, ctx, block, obs; kwargs...) : obs
end

function decode(only::Only, ctx, block::Block, obs; kwargs...)
    _decode(only, ctx, block, obs; kwargs...)
end
function decode(only::Only, ctx, block::WrapperBlock, obs; kwargs...)
    _decode(only, ctx, block, obs; kwargs...)
end
function _decode(only, ctx, block, obs; kwargs...)
    return only.fn(decodedblock(only.encoding, block)) ?
           decode(only.encoding, ctx, block, obs; kwargs...) : obs
end

# ## Test

InlineTest.@testset "Only [block]" begin
    encx = Only(:x, OneHot())
    inblock = Label(1:100)
    inblocknamed = Named(:x, inblock)
    obs = mockblock(inblock)
    encobs = encode(OneHot(), Training(), inblock, obs)

    @test encodedblock(encx, inblock) === nothing
    @test encodedblock(encx, inblocknamed) isa Named{:x}
    @test encodedblock(Only(Named, OneHot()), inblock) === nothing
    @test encodedblock(Only(Named, OneHot()), inblocknamed) isa Named{:x}

    outblock = encodedblock(OneHot(), inblock)
    outblocknamed = encodedblock(OneHot(), inblocknamed)
    @test decodedblock(encx, outblock) === nothing
    @test decodedblock(encx, outblocknamed) isa Named{:x}
    @test decodedblock(Only(Named, OneHot()), outblock) === nothing
    @test decodedblock(Only(Named, OneHot()), outblocknamed) isa Named{:x}

    @test encode(encx, Training(), inblock, obs) == obs
    @test encode(encx, Training(), inblocknamed, obs) != obs

    @test decode(encx, Training(), outblock, encobs) == encobs
    @test decode(encx, Training(), outblocknamed, encobs) != encobs

    tfm = OneHot()
    only = Only(:name, tfm)
    block = Named(:name, Label(["cat", "dog"]))
    obs = mockblock(block)
    testencoding(only, block, obs)
    testencoding(only, (block, wrapped(block)), (obs, obs))
    @test encodedblock(only, wrapped(block)) isa Nothing
end
