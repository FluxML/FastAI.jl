# # Wrapper blocks

abstract type WrapperBlock <: AbstractBlock end

wrapped(w::WrapperBlock) = w.block
function setwrapped(w::WrapperBlock, b)
    return Setfield.@set w.block = b
end
mockblock(w::WrapperBlock) = mockblock(wrapped(w))
checkblock(w::WrapperBlock, data) = checkblock(wrapped(w), data)

# If not overwritten, encodings are applied to the wrapped block

function encodedblock(enc::Encoding, wrapper::WrapperBlock)
    inner = encodedblock(enc, wrapped(wrapper))
    return isnothing(inner) ? nothing : setwrapped(wrapper, inner)
end
function decodedblock(enc::Encoding, wrapper::WrapperBlock)
    inner = decodedblock(enc, wrapped(wrapper))
    return isnothing(inner) ? nothing : setwrapped(wrapper, inner)
end
function encode(enc::Encoding, ctx, wrapper::WrapperBlock, data; kwargs...)
    return encode(enc, ctx, wrapped(wrapper), data; kwargs...)
end
function decode(enc::Encoding, ctx, wrapper::WrapperBlock, data; kwargs...)
    return decode(enc, ctx, wrapped(wrapper), data; kwargs...)
end

# ## Named

"""
    Named(name, block)

Wrapper `Block` to attach a name to a block. Can be used in conjunction
with [`Only`](#) to apply encodings to specific blocks only.
"""
struct Named{Name, B<:AbstractBlock} <: WrapperBlock
    block::B
end
Named(name::Symbol, block::B) where {B<:AbstractBlock} = Named{name, B}(block)


# the name is preserved through encodings and decodings
function encodedblock(enc::Encoding, named::Named{Name}) where Name
    outblock = encodedblock(enc, wrapped(named))
    return isnothing(outblock) ? nothing : Named(Name, outblock)
end

function decodedblock(enc::Encoding, named::Named{Name}) where Name
    outblock = decodedblock(enc, wrapped(named))
    return isnothing(outblock) ? nothing : Named(Name, outblock)
end

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

FastAI.checkblock(many::Many, datas) = all(checkblock(wrapped(many), data) for data in datas)
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


# # Wrapper encodings

"""
    Only(name, encoding)
    Only(block, encoding)

Wrapper that conditionally applies `encoding` only if the block
equals `block` or is a `Named{name}`.
"""
struct Only{Name, E<:Encoding} <: StatefulEncoding
    encoding::E
end

function Only(name::Symbol, encoding::E) where E
    return Only{name, E}(encoding)
end


encodedblock(only::Only{Name}, block::Named{Name}) where Name = encodedblock(only.encoding, block)
decodedblock(only::Only{Name}, block::Named{Name}) where Name = decodedblock(only.encoding, block)

encodestate(only::Only, args...) = encodestate(only.encoding, args...)
decodestate(only::Only, args...) = decodestate(only.encoding, args...)


function encode(
        only::Only{Name},
        context,
        block::Named{Name},
        data;
        state=encodestate(only, context, block, data)) where Name
    return encode(only.encoding, context, block, data; state = state)
end


function decode(
        only::Only{Name},
        context,
        block::Named{Name},
        data;
        state=decodestate(only, context, block, data)) where Name
    return decode(only.encoding, context, block, data; state = state)
end
