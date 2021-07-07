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

"""
    Label(classes; multi = false) <: Block

`Block` for a categorical label in a single-class or multi-class
context, depending on `multi`.
`data` is valid for `Label(classes)` if `data ∈ classes`.
"""
struct Label{T} <: Block
    classes::AbstractVector{T}
    multi::Bool
end
Label(classes; multi = false) = Label(classes, multi)
checkblock(label::Label{T}, data::T) where T = data ∈ label.classes


"""
    Image{N}() <: Block

`Block` for an N-dimensional mask. `data` is valid for `Image{N}()`
if it is an N-dimensional array with color or number element type.
"""
struct Image{N} <: Block end

"""
    Mask{N, T}(classes) <: Block

Block for an N-dimensional categorical mask. `data` is valid for
`Mask{N, T}(classes)`
if it is an N-dimensional array with every element in `classes`.
"""
struct Mask{N, T} <: Block
    classes::AbstractVector{T}
end

struct ImageTensor{N} <: Block
    nchannels::Int
end


struct OneHotTensor{N, T} <: Block
    classes::AbstractVector{T}
end
