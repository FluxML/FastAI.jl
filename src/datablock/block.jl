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

Check that `data` is compatible with `block`.
"""
function checkblock(blocks::Tuple, datas::Tuple)
    @assert length(blocks) == length(datas)
    return all(checkblock(block, data) for (block, data) in zip(blocks, datas))
end

"""
    Label(classes)

`Block` for a categorical label in a single-class context, i.e.
there is only one correct class.
"""
struct Label{T} <: Block
    classes::AbstractVector{T}
end

"""
    LabelMulti(classes)

`Block` for a categorical label in a multi-class context, i.e.
there can be multiple correct classes.
"""
struct LabelMulti{T} <: Block
    classes::AbstractVector{T}
end

checkblock(block::Label{T}, label::T) where T = label in block.labels
checkblock(block::LabelMulti{T}, label::T) where T = label in block.labels
