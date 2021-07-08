"""
    abstract type Block

Represents a kind of data used in machine learning.

- Does not hold the data itself
- If it has any fields, they should be metadata that cannot be
  derived from the data itself and is constant for every sample in
  the dataset. For example `Label` holds all possible classes which
  are constant for the learning problem.


## Interface

- `checkblock(block, data)` checks if `data` is compatible with `block`. For example
    `checkblock(Image{2}(), img)` checks that `img` is a matrix with numeric or color
    values. Defaults to `false`.
"""
abstract type Block end


"""
    checkblock(block, data)
    checkblock(blocks, datas)

Check that `data` is compatible with `block`.
"""
checkblock(::Block, data) = false

function checkblock(blocks::Tuple, datas::Tuple)
    @assert length(blocks) == length(datas)
    return all(checkblock(block, data) for (block, data) in zip(blocks, datas))
end


abstract type AbstractLabel{T} <: Block end

"""
    Label(classes)

`Block` for a categorical label in a single-class context.
`data` is valid for `Label(classes)` if `data ∈ classes`.
"""
struct Label{T} <: AbstractLabel{T}
    classes::AbstractVector{T}
end

checkblock(label::Label{T}, data::T) where T = data ∈ label.classes
"""
    LabelMulti(classes[; thresh = 0.5])

`Block` for a categorical label in a multi-class context.
`data` is valid for `Label(classes)` if `data ∈ classes`.
"""
struct LabelMulti{T} <: AbstractLabel{T}
    classes::AbstractVector{T}
    thresh::Float32
end
LabelMulti(classes; thresh=0.5f0) = LabelMulti(classes, thresh)

function checkblock(label::LabelMulti{T}, v::AbstractVector{T}) where T
    return all(map(x -> x ∈ label.classes, v))
end


"""
    Image{N}() <: Block

`Block` for an N-dimensional mask. `data` is valid for `Image{N}()`
if it is an N-dimensional array with color or number element type.
"""
struct Image{N} <: Block end

checkblock(::Image{N}, ::AbstractArray{T, N}) where {T<:Union{Colorant, Number}, N} = true

"""
    Mask{N, T}(classes) <: Block

Block for an N-dimensional categorical mask. `data` is valid for
`Mask{N, T}(classes)`
if it is an N-dimensional array with every element in `classes`.
"""
struct Mask{N,T} <: Block
    classes::AbstractVector{T}
end
Mask{N}(classes::AbstractVector{T}) where {N,T} = Mask{N,T}(classes)

function checkblock(block::Mask{N,T}, a::AbstractArray{T,N}) where {N,T}
    return all(map(x -> x ∈ block.classes, a))
end
