# # Wrapper blocks

abstract type WrapperBlock <: AbstractBlock end

wrapped(w::WrapperBlock) = w.block
wrapped(b::Block) = b
function setwrapped(w::WrapperBlock, b)
    return Setfield.@set w.block = b
end
mockblock(w::WrapperBlock) = mockblock(wrapped(w))
checkblock(w::WrapperBlock, data) = checkblock(wrapped(w), data)

# If not overwritten, encodings are applied to the wrapped block
"""
    abstract type PropagateWrapper

Defines the default propagation behavior of a `WrapperBlock` when
an encoding is applied to it.

Propagation refers to what happens when an encoding is applied to
a `WrapperBlock`. If no `encode` method is defined for a wrapper block
`wrapper`, `encode` is instead called on the wrapped block.
Propagating the wrapper block means that the block resulting from
encoding the wrapped block is rewrapped in `wrapper.`.

```
wrapper = Wrapper(block)
# propagate
encodedblock(enc, wrapper) = Wrapper(encodedblock(enc, wrapped(wrapper)))

# don't propagate
encodedblock(enc, wrapper) = encodedblock(enc, wrapped(wrapper))
```

The following wrapping behaviors exist:

- `PropagateAlways`: Always propagate. This is the default behavior.
- `PropagateNever`: Never propagate
- `PropagateSameBlock`: Only propagate if the wrapped block is unchanged
"""
abstract type PropagateWrapper end

struct PropagateAlways <: PropagateWrapper end
struct PropagateSameBlock <: PropagateWrapper end
struct PropagateNever <: PropagateWrapper end

propagatewrapper(::WrapperBlock) = PropagateAlways()

encodedblock(enc::Encoding, wrapper::WrapperBlock) =
    encodedblock(enc, wrapper, propagatewrapper(wrapper))

function encodedblock(enc::Encoding, wrapper::WrapperBlock, ::PropagateAlways)
    inner = encodedblock(enc, wrapped(wrapper))
    return isnothing(inner) ? nothing : setwrapped(wrapper, inner)
end

function encodedblock(enc::Encoding, wrapper::WrapperBlock, ::PropagateNever)
    return encodedblock(enc, wrapped(wrapper))
end

function encodedblock(enc::Encoding, wrapper::WrapperBlock, ::PropagateSameBlock)
    inner = encodedblock(enc, wrapped(wrapper))
    inner == wrapped(block) && return setwrapped(wrapper, inner)
    return inner
end

decodedblock(enc::Encoding, wrapper::WrapperBlock) =
    decodedblock(enc, wrapper, propagatewrapper(wrapper))

function decodedblock(enc::Encoding, wrapper::WrapperBlock, ::PropagateAlways)
    inner = decodedblock(enc, wrapped(wrapper))
    return isnothing(inner) ? nothing : setwrapped(wrapper, inner)
end

function decodedblock(enc::Encoding, wrapper::WrapperBlock, ::PropagateNever)
    return decodedblock(enc, wrapped(wrapper))
end

function decodedblock(enc::Encoding, wrapper::WrapperBlock, ::PropagateSameBlock)
    inner = decodedblock(enc, wrapped(wrapper))
    inner == wrapped(block) && return setwrapped(wrapper, inner)
    return inner
end


# Encoding and decoding, if not overwritten for specific wrapper, are fowarded
# to wrapped block.

function encode(enc::Encoding, ctx, wrapper::WrapperBlock, data; kwargs...)
    return encode(enc, ctx, wrapped(wrapper), data; kwargs...)
end

function decode(enc::Encoding, ctx, wrapper::WrapperBlock, data; kwargs...)
    return decode(enc, ctx, wrapped(wrapper), data; kwargs...)
end


# Training interface

blockbackbone(wrapper::WrapperBlock) = blockbackbone(wrapped(wrapper))
blockmodel(wrapper::WrapperBlock, out, args...) = blockmodel(wrapped(wrapper), out, args...)
blockmodel(in::Block, out::WrapperBlock, args...) = blockmodel(in, wrapped(out), args...)

blocklossfn(wrapper::WrapperBlock, out) = blocklossfn(wrapped(wrapper), out)
blocklossfn(in::Block, out::WrapperBlock) = blocklossfn(in, wrapped(out))

# ## Named

"""
    Named(name, block)

Wrapper `Block` to attach a name to a block. Can be used in conjunction
with [`Only`](#) to apply encodings to specific blocks only.
"""
struct Named{Name,B <: AbstractBlock} <: WrapperBlock
    block::B
end
Named(name::Symbol, block::B) where {B <: AbstractBlock} = Named{name,B}(block)


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
struct Many{B <: AbstractBlock} <: WrapperBlock
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

Wrapper that conditionally applies `encoding` only if the block
is a `Named{name}`.
"""
struct Only{Name,E <: Encoding} <: StatefulEncoding
    encoding::E
end

function Only(name::Symbol, encoding::E) where E
    return Only{name,E}(encoding)
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
    return encode(only.encoding, context, block, data; state=state)
end


function decode(
        only::Only{Name},
        context,
        block::Named{Name},
        data;
        state=decodestate(only, context, block, data)) where Name
    return decode(only.encoding, context, block, data; state=state)
end
