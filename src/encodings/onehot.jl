

struct OneHotTensor{N, T} <: Block
    classes::AbstractVector{T}
end

function checkblock(block::OneHotTensor{N}, a::AbstractArray{T, M}) where {M, N, T}
    return N + 1 == M && last(size(a)) == length(block.classes)
end

mockblock(block::OneHotTensor{0}) = encode(
    OneHot(), Validation(), Label(block.classes), rand(block.classes))

function mockblock(block::OneHotTensor{N}) where N
    maskblock = Mask{N}(block.classes)
    return encode(OneHot(), Validation(), maskblock, mockblock(maskblock))
end

struct OneHotTensorMulti{N, T} <: Block
    classes::AbstractVector{T}
end

function checkblock(block::OneHotTensorMulti{N}, a::AbstractArray{T, M}) where {M, N, T}
    return N + 1 == M && last(size(a)) == length(block.classes)
end

"""
    OneHot()
    OneHot(T, threshold)

`Encoding` that turns categorical labels into one-hot encoded arrays of type `T`.

Encodes
```
      `Mask{N, U}` -> `OneHotTensor{N, T}`
`LabelMulti{N, U}` -> `OneHotTensorMulti{N, T}`
        `Label{U}` -> `OneHotTensor{N, T}`
```
"""
struct OneHot{TT<:Type} <: Encoding
    T::TT
    threshold::Float32
end

OneHot() = OneHot(Float32, 0.5f0)


# ### `Label` implementation
encodedblock(::OneHot, block::Label{T}) where T = OneHotTensor{0, T}(block.classes)
decodedblock(::OneHot, block::OneHotTensor{0}) = Label(block.classes)

function encode(enc::OneHot, context, block::Label, data)
    idx = findfirst(isequal(data), block.classes)
    isnothing(idx) && error("$data could not be found in `block.classes`: $(block.classes).")
    return DataAugmentation.onehot(enc.T, idx, length(block.classes))
end


function decode(::OneHot, context, block::OneHotTensor{0}, data)
    return block.classes[argmax(data)]
end

# ### `LabelMulti` implementation

encodedblock(::OneHot, block::LabelMulti{T}) where T = OneHotTensorMulti{0, T}(block.classes)
decodedblock(::OneHot, block::OneHotTensorMulti{0}) = LabelMulti(block.classes)

function encode(enc::OneHot, context, block::LabelMulti, data)
    return collect(enc.T, (c in data for c in block.classes))
end

function decode(enc::OneHot, context, block::OneHotTensorMulti{0}, data)
    return block.classes[softmax(data) .> enc.threshold]
end

# ### `Mask` implementation

encodedblock(::OneHot, block::Mask{N, T}) where {N, T} = OneHotTensor{N, T}(block.classes)
decodedblock(::OneHot, block::OneHotTensor{N, T}) where {N, T} = Mask{N, T}(block.classes)

function encode(enc::OneHot, context, block::Mask, data)
    tfm = DataAugmentation.OneHot{enc.T}()
    return apply(tfm, DataAugmentation.MaskMulti(data, block.classes)) |> DataAugmentation.itemdata
end

function decode(enc::OneHot, context, block::OneHotTensor, data)
    Tidx = length(block.classes) >= 255 ? UInt16 : UInt8
    classidxs = reshape(
        map(I -> Tidx(I.I[end]), argmax(data; dims = ndims(data))),
        size(data)[1:end-1])
    return IndirectArray(classidxs, block.classes)
end
