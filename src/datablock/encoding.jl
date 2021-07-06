

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
    so it cannot depend on the sample or on randomness. The default is to return `block`,
    since the default `encode` doesn't do anything.
- `decodedblock(::E, block::Block) -> block'` returns the block that is obtained by
    decoding `block` with encoding `E`. This needs to be constant for an instance of `E`,
    so it cannot depend on the sample or on randomness. The default is to return `block`,
    since the default `decode` doesn't do anything.
- `encode!(buf, ::E, ::Context, block::Block, data)` encodes `data` inplace.
- `decode!(buf, ::E, ::Context, block::Block, data)` decodes `data` inplace.

"""
abstract type Encoding end


"""
    abstract type StatefulEncoding <: Encoding

Encoding that needs to compute some state from the whole sample, even
if it only transforms some of the blocks. This could be random state
for stochastic augmentations that needs to be the same for every block
that is encoded.

The state is created by calling `samplestate(encoding, context, blocks, sample)`
and passed to recursive calls with the keyword argument `state`.
As a result, you need to implement `encode`, `decode`, `encode!`, `decode!` with a
keyword argument `state` that defaults to the above call.
"""
abstract type StatefulEncoding end


"""
    ImagePreprocessing(C, T, stats)

Encodes `Image`s by converting them to a common color type `C`,
expanding the color channels and normalizing the channel values.

Encodes `Image{N}` -> `ImageTensor{N}` and decodes the reverse.
"""
struct ImagePreprocessing <: Encoding
    # intermediate color type to convert to
    C::
end

function encodedblock(ip::ImagePreprocessing, ::Image{N}) where N
    return ImageTensor{N}(colorchannels(ip.C))
end

decodedblock(::ImagePreprocessing, ::ImageTensor{N}) where N = Image{N}()

function encode()
    # see imagepreprocessing.jl
end


"""
    OneHot()

One-hot encodes data.
"""

encodedblock(::OneHot, label::Label{T}) = OneHotTensor{1}(label.classes)
decodedblock(::OneHot, onehot::OneHotTensor{1, T}) = Label{T}(onehot.classes)

encodedblock(::OneHot, mask::Mask{T}) = OneHotTensor{3}(label.classes)
decodedblock(::OneHot, onehot::OneHotTensor{3, T}) = Mask{T}(label.classes)
