
"""
    Named(name, block)

Wrapper `Block` to attach a name to a block. Can be used in conjunction
with [`Only`](#) to apply encodings to specific blocks only.
"""
struct Named{Name,B<:AbstractBlock} <: WrapperBlock
    block::B
end
Named(name::Symbol, block::B) where {B<:AbstractBlock} = Named{name,B}(block)

setwrapped(named::Named{name}, block) where name = Named(name, block)


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
struct Only{E<:Encoding} <: StatefulEncoding
    fn::Any
    encoding::E
end

function Only(name::Symbol, encoding::Encoding)
    return Only(Named{name}, encoding)
end

function Only(B::Type{<:AbstractBlock}, encoding::Encoding)
    return Only(block -> block isa B, encoding)
end


encodedblock(only::Only, block::Block) =
    only.fn(block) ? encodedblock(only.encoding, block) : nothing
encodedblock(only::Only, block::WrapperBlock) =
    only.fn(block) ? encodedblock(only.encoding, block) : nothing
encodedblock(only::Only, block::Named) =
    only.fn(block) ? encodedblock(only.encoding, block) : nothing

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


function encode(only::Only, ctx, block::Block, data; kwargs...)
    _encode(only, ctx, block, data; kwargs...)
end
function encode(only::Only, ctx, block::WrapperBlock, data; kwargs...)
    _encode(only, ctx, block, data; kwargs...)
end
function _encode(only, ctx, block, data; kwargs...)
    return only.fn(block) ? encode(only.encoding, ctx, block, data; kwargs...) : data
end

function decode(only::Only, ctx, block::Block, data; kwargs...)
    _decode(only, ctx, block, data; kwargs...)
end
function decode(only::Only, ctx, block::WrapperBlock, data; kwargs...)
    _decode(only, ctx, block, data; kwargs...)
end
function _decode(only, ctx, block, data; kwargs...)
    return only.fn(decodedblock(only.encoding, block)) ?
           decode(only.encoding, ctx, block, data; kwargs...) : data
end

# ## Test

InlineTest.@testset "Only [block]" begin
    encx = Only(:x, OneHot())
    inblock = Label(1:100)
    inblocknamed = Named(:x, inblock)
    data = mockblock(inblock)
    encdata = encode(OneHot(), Training(), inblock, data)

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

    @test encode(encx, Training(), inblock, data) == data
    @test encode(encx, Training(), inblocknamed, data) != data

    @test decode(encx, Training(), outblock, encdata) == encdata
    @test decode(encx, Training(), outblocknamed, encdata) != encdata

    tfm = OneHot()
    only = Only(:name, tfm)
    block = Named(:name, Label(["cat", "dog"]))
    data = mockblock(block)
    testencoding(only, block, data)
    testencoding(only, (block, wrapped(block)), (data, data))
    @test encodedblock(only, wrapped(block)) isa Nothing
end
