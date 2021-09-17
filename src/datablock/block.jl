"""
    abstract type AbstractBlock

Abstract supertype of all blocks. You should not subtype form this,
but instead from [`Block`](#) or [`WrapperBlock`](#).
"""
abstract type AbstractBlock end



"""
    abstract type Block

A block describes the meaning of a piece of data in the context of a learning task.
For example, for supervised learning tasks, there is an input and a target
and we want to learn to predict targets from inputs. Learning to predict a
cat/dog label from 2D images is a supervised image classification task that can
be represented with the `Block`s `Image{2}()` and `Label(["cat", "dog"])`.

`Block`s are used in virtually every part of the high-level interfaces, from data
processing over model creation to visualization.

## Extending

Consider the following when subtyping `Block`. A block

- Does not hold observation data itself. Instead they are used in conjunction with
  data to annotate it with some meaning.
- If it has any fields, they should be metadata that cannot be
  derived from the data itself and is constant for every sample in
  the dataset. For example `Label` holds all possible classes which
  are constant for the learning problem.

### Interfaces

There are many interfaces that can be implemented for a `Block`. See the docstrings
of each function for more info about how to implement it.

- [`checkblock`](#)`(block, data)`: check whether a piece of data is a valid block
- [`mockblock`](#)`(block)`: randomly generate a piece of data
- [`blocklossfn`](#)`(predblock, yblock)`: loss function for comparing two blocks
- [`blockmodel`](#)`(inblock, outblock[, backbone])`: construct a task-specific model
- [`blockbackbone`](#)`(inblock)`: construct a backbone model that takes in specific data
- [`plotblock!`](#)`(block, data)`: visualize block data

"""
abstract type Block <: AbstractBlock end


"""
    checkblock(block, data)
    checkblock(blocks, datas)

Check whether `data` is compatible with `block`, returning a `Bool`.

## Examples

```julia
checkblock(Image{2}(), rand(RGB, 16, 16)) == true
```

```julia
checkblock(
    (Image{2}(),        Label(["cat", "dog"])),
    (rand(RGB, 16, 16), "cat"                ),
) == true
```

## Extending

An implementation of `checkblock` should be as specific as possible. The
default method returns `false`, so you only need to implement methods for valid types
and return `true`.
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


"""
    setup(Block, data)

Create an instance of block type `Block` from data container `data`.

## Examples

```julia
setup(Label, ["cat", "dog", "cat"]) == Label(["cat", "dog"])
```
"""
function setup end


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

setup(::Type{Label}, data) = Label(unqiue(eachobs(data)))

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


setup(::Type{LabelMulti}, data) = Label(unqiue(eachobs(data)))

# Image

"""
    Image{N}() <: Block

`Block` for an N-dimensional mask. `data` is valid for `Image{N}()`
if it is an N-dimensional array with color or number element type.
"""
struct Image{N} <: Block end

checkblock(::Image{N}, ::AbstractArray{T,N}) where {T <: Union{Colorant,Number},N} = true
mockblock(::Image{N}) where N = rand(RGB{N0f8}, ntuple(_ -> 16, N))

setup(::Type{Image}, data) = Image{ndims(getobs(data, 1))}()

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

"""
    setup(TableRow, data[; catcols, contcols])

Create a `TableRow` block from data container `data::TableDataset`. If the
categorical and continuous columns are not specified manually, try to
guess them from the dataset's column types.
"""
function setup(::Type{TableRow}, data; catcols=nothing, contcols=nothing)
    catcols_, contcols_ = getcoltypes(data)
    catcols = isnothing(catcols) ? catcols_ : catcols
    contcols = isnothing(contcols) ? contcols_ : contcols

    return TableRow(
        catcols,
        contcols,
        gettransformdict(data, DataAugmentation.Categorify, catcols))
end

function Base.show(io::IO, block::TableRow)
    print(io, ShowCase(block, (:catcols, :contcols), show_params=false, new_lines=true))
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
