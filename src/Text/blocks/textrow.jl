

# TextRow

"""
    TextRow{M, N}(catcols, contcols, categorydict) <: Block

`Block` for table rows with M categorical and N continuous columns. `data`
is valid if it satisfies the `AbstractRow` interface in Tables.jl, values
present in indices for categorical and continuous columns are consistent,
and `data` is indexable by the elements of `catcols` and `contcols`.
"""

struct TextRow{M,N,T} <: Block
    catcols::NTuple{M}
    contcols::NTuple{N}
    categorydict::T
end

function TextRow(catcols, contcols, categorydict)
    TextRow{length(catcols),length(contcols)}(catcols, contcols, categorydict)
end

function checkblock(block::TextRow{M,N}, x where {M,N,T<:Number}
end

function mockblock(block::TextRow)
end

function setup(::Type{TextRow}, data)
end

# ## Interpretation

function showblock!(io, ::ShowText, block::TextRow, obs)
end
