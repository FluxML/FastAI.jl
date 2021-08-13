abstract type WrapperBlock <: AbstractBlock end

wrapped(w::WrapperBlock) = w.block
function setwrapped(w::WrapperBlock, b)
    return Setfield.@set w.block = b
end
mockblock(w::WrapperBlock) = mockblock(wrapped(w))
checkblock(w::WrapperBlock, data) = checkblock(wrapped(w), data)

# If not overwritten, encodings are applied to the wrapped block

function encodedblock(enc::Encoding, wrapper::W) where {W<:WrapperBlock}
    W(encodedblock(enc, wrapped(many)))
end
decodedblock(enc::Encoding, wrapper::WrapperBlock) = Many(decodedblock(enc, wrapped(many)))
function encode(enc::Encoding, ctx, wrapper::WrapperBlock, data; kwargs...)
    return encode(enc, ctx, wrapped(wrapper), data; kwargs...)
end
function decode(enc::Encoding, ctx, wrapper::WrapperBlock, data; kwargs...)
    return decode(enc, ctx, wrapped(wrapper), data; kwargs...)
end


"""
    Named(name, block)

Wrapper `Block` to attach a name to a block. Can be used in conjunction
with [`Only`](#) to apply encodings to specific blocks only.
"""
struct Named{Name, B<:Block} <: WrapperBlock
    block::B
end
Named(name::Symbol, block::B) where {B<:Block} = Named{name, B}(block)


# the name is preserved through encodings and decodings
function encodedblock(enc::Encoding, named::Named{Name}) where Name
    outblock = encodedblock(enc, wrapped(named))
    return isnothing(outblock) ? nothing : Named(Name, outblock)
end

function decodedblock(enc::Encoding, named::Named{Name}) where Name
    outblock = decodedblock(enc, wrapped(named))
    return isnothing(outblock) ? nothing : Named(Name, outblock)
end

# Wrapper encodings

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
