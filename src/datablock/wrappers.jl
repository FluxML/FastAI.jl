# # Wrapper blocks

"""
    abstract type WrapperBlock

Supertype for blocks that "wrap" an existing block, inheriting its
functionality, allowing you to override just parts of its interface.

For examples of `WrapperBlock`, see [`Bounded`](#)

"""
abstract type WrapperBlock <: AbstractBlock end

Base.parent(w::WrapperBlock) = w.block
Base.parent(b::Block) = b
wrapped(w::WrapperBlock) = wrapped(parent(w))
wrapped(b::Block) = b
function setwrapped(w::WrapperBlock, b)
    # TODO: make recursive
    return Setfield.@set w.block = b
end
mockblock(w::WrapperBlock) = mockblock(parent(w))
checkblock(w::WrapperBlock, obs) = checkblock(parent(w), obs)

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
    by the encoding
"""
abstract type PropagateWrapper end

struct PropagateAlways <: PropagateWrapper end
struct PropagateSameBlock <: PropagateWrapper end
struct PropagateNever <: PropagateWrapper end

# If not overwritten, encodings are applied to the wrapped block
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
    inner == wrapped(wrapper) && return setwrapped(wrapper, inner)
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

function encode(enc::Encoding, ctx, wrapper::WrapperBlock, obs; kwargs...)
    return encode(enc, ctx, wrapped(wrapper), obs; kwargs...)
end

function decode(enc::Encoding, ctx, wrapper::WrapperBlock, obs; kwargs...)
    return decode(enc, ctx, wrapped(wrapper), obs; kwargs...)
end


# ### Training interface
#
# The default behavior for `WrapperBlock`s is to forward behavior that is not
# explicitly overwritten to the wrapped block. So the same is done here for
# the training interface.

blockbackbone(wrapper::WrapperBlock) = blockbackbone(wrapped(wrapper))
blockmodel(wrapper::WrapperBlock, out, args...) = blockmodel(wrapped(wrapper), out, args...)
blockmodel(in::Block, out::WrapperBlock, args...) = blockmodel(in, wrapped(out), args...)

blocklossfn(wrapper::WrapperBlock, out) = blocklossfn(wrapped(wrapper), out)
blocklossfn(in::Block, out::WrapperBlock) = blocklossfn(in, wrapped(out))
