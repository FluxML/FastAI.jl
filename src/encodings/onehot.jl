

struct OneHotTensor{N, T} <: Block
    classes::AbstractVector{T}
end

checkblock(block::OneHotTensor{N}, a::AbstractArray{T, N}) where {N, T} = true
mockblock(block::OneHotTensor{1}) = encode(OneHot(), Validation(), Label(block.classes), rand(block.classes))

struct OneHotTensorMulti{N, T} <: Block
    classes::AbstractVector{T}
end

checkblock(block::OneHotTensorMulti{N}, a::AbstractArray{T, N}) where {N, T} = true

"""
    OneHot()
    OneHot(T, threshold)

`Encoding` that turns categorical labels into one-hot encoded arrays of type `T`.

Encodes
```
     `Mask{N, U}` -> `OneHotTensor{N, T}`
`MaskMulti{N, U}` -> `OneHotTensorMulti{N, T}`
       `Label{U}` -> `OneHotTensor{N, T}`
```
"""
struct OneHot{TT<:Type} <: Encoding
    T::TT
    threshold::Float32
end

OneHot() = OneHot(Float32, 0.5f0)


# ### `Label` implementation
encodedblock(::OneHot, block::Label{T}) where T = OneHotTensor{1, T}(block.classes)
decodedblock(::OneHot, block::OneHotTensor{1}) = Label(block.classes)

function encode(enc::OneHot, context, block::Label, data)
    idx = findfirst(isequal(data), block.classes)
    isnothing(idx) && error("$data could not be found in `block.classes`: $(block.classes).")
    return DataAugmentation.onehot(enc.T, idx, length(block.classes))
end


function decode(::OneHot, context, block::OneHotTensor{1}, data)
    return block.classes[argmax(data)]
end

# ### `LabelMulti` implementation

encodedblock(::OneHot, block::LabelMulti{T}) where T = OneHotTensorMulti{1, T}(block.classes)
decodedblock(::OneHot, block::OneHotTensorMulti{1}) = LabelMulti(block.classes)

function encode(enc::OneHot, context, block::LabelMulti, data)
    return collect(enc.T, (c in data for c in block.classes))
end

function decode(enc::OneHot, context, block::OneHotTensorMulti{1}, data)
    return block.classes[softmax(data) .> enc.threshold]
end

# ### `Mask` implementation

encodedblock(::OneHot, block::Mask{N, T}) where {N, T} = OneHotTensor{N+1, T}(block.classes)
decodedblock(::OneHot, block::OneHotTensor{N, T}) where {N, T} = Mask{N-1, T}(block.classes)

function encode(enc::OneHot, context, block::Mask, data)
    tfm = DataAugmentation.OneHot{enc.T}()
    return apply(tfm, DataAugmentation.MaskMulti(data)) |> DataAugmentation.itemdata
end

function decode(enc::OneHot, context, block::OneHotTensor, data)
    return reshape(
        map(I -> block.classes[I.I[end]], argmax(data; dims=ndims(data))),
        size(data)[1:end-1])
end
