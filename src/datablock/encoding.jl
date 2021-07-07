

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
    encode(encoding, context, block, data)
    encode(encoding, context, blocks, data)
    encode(encodings, context, blocks, data)


"""
function encode(encodings::NTuple{N, <:Encoding}, context, blocks, data) where N
    for encoding in encodings
        data = encode(encoding, context, blocks, data)
        blocks = encodedblock(encoding, blocks)
    end
    return data
end

function encode(encoding::Encoding, context, blocks::NTuple{N}, datas::NTuple{N}) where N
    return Tuple(encode(encoding, context, block, data)
                    for (block, data) in zip(blocks, datas))
end


function decode(encodings::NTuple{N, <:Encoding}, context, blocks, data) where N
    for encoding in Iterators.reverse(encodings)
        data = decode(encoding, context, blocks, data)
        blocks = decodedblock(encoding, blocks)
    end
    return data
end

function decode(encoding::Encoding, context, blocks::NTuple{N}, datas::NTuple{N}) where N
    return Tuple(decode(encoding, context, block, data)
                    for (block, data) in zip(blocks, datas))
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
    Tuple(encodedblock(encoding, block) for block in blocks)
end
function encodedblock(encodings::NTuple{N, <:Encoding}, blocks) where N
    encoded = false
    for encoding in encodings
        outblocks = encodedblock(encoding, blocks)
        blocks = isnothing(outblocks) ? blocks : outblocks
        encoded = encoded || !isnothing(outblocks)
    end
    return encoded ? blocks : nothing
end

"""
    decodedblock(encoding, block)
    decodedblock(encoding, blocks)

Return the block that is obtained by decoding `block` with encoding `E`.
This needs to be constant for an instance of `E`, so it cannot depend on the
sample or on randomness. The default is to return `nothing`,
meaning the same block is returned and not changed. Encodings that return the same
block but change the data when decoding should return `block`.
"""
decodedblock(::Encoding, ::Block) = nothing
function decodedblock(encoding::Encoding, blocks::Tuple)
    Tuple(decodedblock(encoding, block) for block in blocks)
end
function decodedblock(encodings::NTuple{N, <:Encoding}, blocks) where N
    decoded = false
    for encoding in Iterators.reverse(encodings)
        outblocks = decodedblock(encoding, blocks)
        blocks = isnothing(outblocks) ? blocks : outblocks
        decoded = decoded || !isnothing(outblocks)
    end
    return decoded ? blocks : nothing
end

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
abstract type StatefulEncoding <: Encoding end


"""
    ImagePreprocessing(C, T, stats)

Encodes `Image`s by converting them to a common color type `C`,
expanding the color channels and normalizing the channel values.

Encodes `Image{N}` -> `ImageTensor{N}` and decodes the reverse.
"""
struct ImagePreprocessing <: Encoding
    # intermediate color type to convert to
    C
    # number type of output image tensor
    T
end

function ImagePreprocessing(; C = RGB{N0f8}, T = Float32)

end

function encodedblock(ip::ImagePreprocessing, ::Image{N}) where N
    return ImageTensor{N}(colorchannels(ip.C))
end

decodedblock(::ImagePreprocessing, ::ImageTensor{N}) where N = Image{N}()

function encode(::ImagePreprocessing)
    # see imagepreprocessing.jl
end


"""
    OneHot()

One-hot encodes data.
"""

encodedblock(::OneHot, label::Label{T}) where T = OneHotTensor{1, T}(label.classes)
decodedblock(::OneHot, onehot::OneHotTensor{1, T}) where T = Label{T}(onehot.classes)

encodedblock(::OneHot, mask::Mask{2, T}) where T = OneHotTensor{3, T}(label.classes)
decodedblock(::OneHot, onehot::OneHotTensor{3, T}) where T = Mask{2, T}(label.classes)


"""
    checkencodings()

Check that `encodings` can be sequenced, i.e. given input `blocks`, the
`encodedblock`s of every encoding can be fed into the next.
"""
function checkencodings(encodings, blocks)

end

"""
Some printing of the steps taken in a full pipeline would be nice. Should
highlight which blocks change.
Can also check that every encoding is applied to at least one block.


- INPUT:                (Image{2}(),           Label(classes))

- ImagePreprocessing:   (**ImageTensor{2}()**, Label(classes))
- OneHot:               (ImageTensor{2}(),     **OneHot{1}(classes)**)

- OUTPUT:               (ImageTensor{2}(),     OneHot{1}(classes))


"""
