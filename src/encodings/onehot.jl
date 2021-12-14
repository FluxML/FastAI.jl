

"""
    OneHotTensor{N, T}(classes) <: Block

A block representing a one-hot encoded, N-dimensional array
categorical variable. For example, a single categorical label
is a `OneHotTensor{0, T}` (aliased to `OneHotTensor{T}`).

Use the [`OneHot`](#) encoding to one-hot encode [`Label`](#)s
or [`LabelMulti`](#)s.
"""
struct OneHotTensor{N, T} <: Block
    classes::AbstractVector{T}
end

const OneHotLabel{T} = OneHotTensor{0, T}

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

function mockblock(block::OneHotTensorMulti{0}) where N
    labelblock = LabelMulti(block.classes)
    return encode(OneHot(), Validation(), labelblock, mockblock(labelblock))
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
decodedblock(::OneHot, block::OneHotLabel) = Label(block.classes)

function encode(enc::OneHot, context, block::Label, data)
    idx = findfirst(isequal(data), block.classes)
    isnothing(idx) && error("$data could not be found in `block.classes`: $(block.classes).")
    return DataAugmentation.onehot(enc.T, idx, length(block.classes))
end


function decode(::OneHot, context, block::OneHotLabel, data)
    return block.classes[argmax(data)]
end

# ### `LabelMulti` implementation

encodedblock(::OneHot, block::LabelMulti{T}) where T = OneHotTensorMulti{0, T}(block.classes)
decodedblock(::OneHot, block::OneHotTensorMulti{0}) = LabelMulti(block.classes)

function encode(enc::OneHot, _, block::LabelMulti, data)
    return collect(enc.T, (c in data for c in block.classes))
end

function decode(enc::OneHot, _, block::OneHotTensorMulti{0}, data)
    return block.classes[softmax(data) .> enc.threshold]
end


function blocklossfn(outblock::OneHotTensor{0}, yblock::OneHotTensor{0})
    outblock.classes == yblock.classes || error("Classes of $outblock and $yblock differ!")
    return Flux.Losses.logitcrossentropy
end

function blocklossfn(outblock::OneHotTensorMulti{0}, yblock::OneHotTensorMulti{0})
    outblock.classes == yblock.classes || error("Classes of $outblock and $yblock differ!")
    return Flux.Losses.logitbinarycrossentropy
end
