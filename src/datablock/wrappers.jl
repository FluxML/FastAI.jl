# # Wrapper blocks

abstract type WrapperBlock <: AbstractBlock end

Base.parent(w::WrapperBlock) = w.block
Base.parent(b::Block) = b
wrapped(w::WrapperBlock) = wrapped(parent(w))
wrapped(b::Block) = b
function setwrapped(w::WrapperBlock, b)
    return Setfield.@set w.block = b
end
mockblock(w::WrapperBlock) = mockblock(wrapped(w))
checkblock(w::WrapperBlock, obs) = checkblock(wrapped(w), obs)

function blockname(wrapper::WrapperBlock)
    w = string(nameof(typeof(wrapper)))
    b = blockname(parent(wrapper))
    return "$w($b)"
end


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
    by the encoding
"""
abstract type PropagateWrapper end

"""
    struct PropagateAlways <: PropagateWrapper end

Always propagate a wrapper type.

See [`propagate`](#) for more information.
"""

struct PropagateAlways <: PropagateWrapper end
propagate(::PropagateAlways, _, _) = true
propagatedecode(::PropagateAlways, _, _) = true

"""
    struct PropagateNever <: PropagateWrapper end

Never propagate a wrapper type.

See [`propagate`](#) for more information.
"""
struct PropagateNever <: PropagateWrapper end
propagate(::PropagateNever, _, _) = false
propagatedecode(::PropagateNever, _, _) = false

"""
    struct PropagateSameBlock <: PropagateWrapper end

Propagate a wrapper type only if the encoded block is same,
ignoring any wrappers.

See [`propagate`](#) for more information.
"""
struct PropagateSameBlock <: PropagateWrapper end
propagate(::PropagateSameBlock, encoding, block) =
    wrapped(encodedblock(encoding, block)) == wrapped(block)
propagatedecode(::PropagateSameBlock, encoding, block) =
    wrapped(decodedblock(encoding, block)) == wrapped(block)

"""
    struct PropagateSameWrapper <: PropagateWrapper end

Propagate a wrapper type only if the encoded block is the exact same,
including any wrappers.

See [`propagate`](#) for more information.
"""
struct PropagateSameWrapper <: PropagateWrapper end
propagate(::PropagateSameWrapper, encoding, block) =
    encodedblock(encoding, block) == block
propagatedecode(::PropagateSameWrapper, encoding, block) =
    decodedblock(encoding, block) == block


PropagateWrapper(::WrapperBlock) = PropagateAlways()


"""
    propagate(wrapper::WrapperBlock, encoding::Encoding) -> true|false

Whether the wrapper type should be kept after encoding the wrapped block with `encoding`.
"""
propagate(wrapper::WrapperBlock, encoding::Encoding) =
    propagate(PropagateWrapper(wrapper), encoding, parent(wrapper))

"""
    propagatedecode(wrapper::WrapperBlock, encoding::Encoding) -> true|false

Whether the wrapper type should be kept after decoding the wrapped block with `encoding`.
"""
propagatedecode(wrapper::WrapperBlock, encoding::Encoding) =
    propagatedecode(PropagateWrapper(wrapper), encoding, parent(wrapper))


function encodedblock(encoding::Encoding, wrapper::WrapperBlock)
    encblock = encodedblock(encoding, parent(wrapper))
    isnothing(encblock) && return nothing
    return if propagate(wrapper, encoding)
        setwrapped(wrapper, encblock)
    else
        encblock
    end
end

function decodedblock(encoding::Encoding, wrapper::WrapperBlock)
    decblock = decodedblock(encoding, parent(wrapper))
    isnothing(decblock) && return nothing
    return if propagatedecode(wrapper, encoding)
        setwrapped(wrapper, decblock)
    else
        decblock
    end
end

# Encoding and decoding, if not overwritten for specific wrapper, are fowarded
# to wrapped block.

function encode(enc::Encoding, ctx, wrapper::WrapperBlock, obs; kwargs...)
    return encode(enc, ctx, parent(wrapper), obs; kwargs...)
end

function decode(enc::Encoding, ctx, wrapper::WrapperBlock, obs; kwargs...)
    return decode(enc, ctx, parent(wrapper), obs; kwargs...)
end

encodestate(enc, ctx, w::WrapperBlock, obs) = encodestate(enc, ctx, parent(w), obs)
decodestate(enc, ctx, w::WrapperBlock, obs) = decodestate(enc, ctx, parent(w), obs)

function encode(enc::StatefulEncoding,
                ctx,
                w::WrapperBlock,
                obs;
                state = encodestate(enc, ctx, w, obs))
    encode(enc, ctx, parent(w), obs; state)
end

function decode(enc::StatefulEncoding,
                ctx,
                w::WrapperBlock,
                obs;
                state = decodestate(enc, ctx, w, obs))
    decode(enc, ctx, parent(w), obs; state)
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


struct TestWrapper{B<:AbstractBlock, P<:PropagateWrapper} <: WrapperBlock
    block::B
    propagation::P
end

PropagateWrapper(w::TestWrapper) = w.propagation

@testset "Wrapper propagation" begin
    block = Label(1:10)
    enc = OneHot()
    encblock = encodedblock(enc, block)

    w = TestWrapper(block, PropagateAlways())
    @test propagate(w, enc)

    w = TestWrapper(block, PropagateNever())
    @test !propagate(w, enc)

    w = TestWrapper(block, PropagateSameBlock())
    @test !propagate(w, enc)

    w = TestWrapper(block, PropagateSameWrapper())
    @test !propagate(w, enc)

    @test encodedblock(enc, TestWrapper(block, PropagateAlways())) isa TestWrapper
    @test decodedblock(enc, TestWrapper(encblock, PropagateAlways())) isa TestWrapper

    @test !(encodedblock(enc, TestWrapper(block, PropagateNever())) isa TestWrapper)
    @test !(decodedblock(enc, TestWrapper(encblock, PropagateNever())) isa TestWrapper)
end
