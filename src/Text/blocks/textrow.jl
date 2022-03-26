

# TextRow

"""
    TextRow{M, N}(catcols, contcols, categorydict) <: Block

`Block` for table rows with M categorical and N continuous columns. `data`
is valid if it satisfies the `AbstractRow` interface in Tables.jl, values
present in indices for categorical and continuous columns are consistent,
and `data` is indexable by the elements of `catcols` and `contcols`.
"""
struct TextRow{M,N} <: Block
    catcols::NTuple{M}
    contcols::NTuple{N}
    categorydict::T
end

function TextRow(catcols, contcols)
    TextRow{length(catcols),length(contcols)}(catcols, contcols, categorydict)
end

function checkblock(block::TextRow, x)
    columns = Tables.columnnames(x)
    (
        all(col -> col ∈ columns, (block.catcols..., block.contcols...)) &&
        all(
            col ->
                haskey(block.categorydict, col) &&
                    (ismissing(x[col]) || x[col] ∈ block.categorydict[col]),
            block.catcols,
        ) &&
        all(col -> ismissing(x[col]) || x[col] isa Number, block.contcols)
    )
end

function mockblock(block::TextRow)
    cols = (block.catcols..., block.contcols...)
    vals = map(cols) do col
        col in block.catcols ? rand(block.categorydict[col]) : rand()
    end
    return NamedTuple(zip(cols, vals))
end

"""
    setup(TextRow, data[; catcols, contcols])

Create a `TextRow` block from data container `data::TextDataset`. If the
categorical and continuous columns are not specified manually, try to
guess them from the dataset's column types.
"""
function setup(::Type{TextRow}, data; catcols=nothing, contcols=nothing)
    catcols_, contcols_ = getcoltypes(data)
    catcols = isnothing(catcols) ? catcols_ : catcols
    contcols = isnothing(contcols) ? contcols_ : contcols

    return TextRow(
        catcols,
        contcols,
        gettransformdict(data, DataAugmentation.Categorify, catcols),
    )
end

function Base.show(io::IO, block::TextRow)
    print(io, ShowCase(block, (:catcols, :contcols), show_params=false, new_lines=true))
end

# ## Interpretation

# function showblock!(io, ::ShowText, block::TextRow, obs) end