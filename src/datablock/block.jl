abstract type AbstractBlock end



"""
    abstract type Block

Represents a kind of data used in machine learning.

- Does not hold the data itself
- If it has any fields, they should be metadata that cannot be
  derived from the data itself and is constant for every sample in
  the dataset. For example `Label` holds all possible classes which
  are constant for the learning problem.


## Interface

Required:

- `checkblock(block, data)` checks if `data` is compatible with `block`. For example
    `checkblock(Image{2}(), img)` checks that `img` is a matrix with numeric or color
    values. Defaults to `false`.

Optional:

- For plotting interface:
    - `plotblock!(f, block)`. Plot `block` on Makie figure `f`.
- For testing interface:
    - `mockblock(block)`. Randomly generate an instance of `block`. Needed
    to derive the testing interface for `BlockMethod`.
- For training interface:
    - [`blocklossfn`](#)`(pred_block, true_block)`
    - [`blockmodel`](#)`(inblock, outblock, backbone)`

"""
abstract type Block <: AbstractBlock end


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
    mockblock(block)
    mockblock(blocks)

Randomly generate an instance of `block`.
"""
mockblock(blocks::Tuple) = map(mockblock, blocks)


# ## Utilities

typify(T::Type) = T
typify(t::Tuple) = Tuple{map(typify, t)...}
typify(block::FastAI.AbstractBlock) = typeof(block)


# ## Block implementations

abstract type AbstractLabel{T} <: Block end

# Label

"""
    Label(classes)

`Block` for a categorical label in a single-class context.
`data` is valid for `Label(classes)` if `data ∈ classes`.
"""
struct Label{T} <: AbstractLabel{T}
    classes::AbstractVector{T}
end

checkblock(label::Label{T}, data::T) where T = data ∈ label.classes
mockblock(label::Label) = rand(label.classes)


# LabelMulti

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

mockblock(label::LabelMulti) =
    unique([rand(label.classes) for _ in 1:rand(1:length(label.classes))])


# Image

"""
    Image{N}() <: Block

`Block` for an N-dimensional mask. `data` is valid for `Image{N}()`
if it is an N-dimensional array with color or number element type.
"""
struct Image{N} <: Block end

checkblock(::Image{N}, ::AbstractArray{T,N}) where {T <: Union{Colorant,Number},N} = true
mockblock(::Image{N}) where N = rand(RGB{N0f8}, ntuple(_ -> 16, N))

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

mockblock(mask::Mask{N}) where N = rand(mask.classes, ntuple(_ -> 16, N))

# Keypoints

"""

    Keypoints{N}(sz) <: Block

A block representing an array of size `sz` filled with keypoints of type
`SVector{N}`.
"""
struct Keypoints{N, M} <: Block
    sz::NTuple{M, Int}
end
Keypoints{N}(n::Int) where N = Keypoints{N, 1}((n,))
Keypoints{N}(t::NTuple{M, Int}) where {N, M} = Keypoints{N, M}(t)

function checkblock(
        block::Keypoints{N,M},
        a::AbstractArray{<:Union{SVector{N,T},Nothing},M}) where {M,N,T}
    return true
end

mockblock(block::Keypoints{N}) where N = rand(SVector{N, Float32}, block.sz)


# TableRow

"""
    TableRow{M, N}(catcols, contcols, categorydict) <: Block

`Block` for table rows with M categorical and N continuous columns. `data`
is valid if it satisfies the `AbstractRow` interface in Tables.jl, values 
present in indices for categorical and continuous columns are consistent, 
and `data` is indexable by the elements of `catcols` and `contcols`.
"""
struct TableRow{M, N, T} <: Block
    catcols::NTuple{M}
    contcols::NTuple{N}
    categorydict::T
end

function TableRow(catcols, contcols, categorydict)
    TableRow{length(catcols), length(contcols)}(catcols, contcols, categorydict)
end

function checkblock(block::TableRow, x)
    columns = Tables.columnnames(x)
    (all(col -> col ∈ columns, (block.catcols..., block.contcols...)) &&
    all(col -> haskey(block.categorydict, col) && 
        (ismissing(x[col]) || x[col] ∈ block.categorydict[col]), block.catcols) &&
    all(col -> ismissing(x[col]) || x[col] isa Number, block.contcols))
end

function mockblock(block::TableRow)
    cols = (block.catcols..., block.contcols...)
    vals = map(cols) do col
        col in block.catcols ? 
            rand(block.categorydict[col]) : rand()
    end
    return NamedTuple(zip(cols, vals))
end

# Continous

"""
    Continuous(size) <: Block

`Block` for collections of numbers. `data` is valid if it's 
length is `size` and contains `Number`s.
"""

struct Continuous <: Block
    size::Int
end

function checkblock(block::Continuous, x)
    block.size == length(x) && eltype(x) <: Number
end

mockblock(block::Continuous) = rand(block.size)
