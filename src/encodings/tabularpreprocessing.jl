struct RawContinuousBlock{N, T, M} <: Block
    columns
    allcols
end

RawContinuousBlock(columns, allcols) = RawContinuousBlock{length(columns), eltype(columns), length(allcols)}(columns, allcols)

struct RawCategoricalBlock{N, T, M} <: Block 
    columns
    allcols
    categorydict
end

RawCategoricalBlock(columns, allcols, categorydict) = RawCategoricalBlock{length(columns), eltype(columns), length(allcols)}(columns, categorydict)

function checkblock(::Union{RawContinuousBlock{N}, RawCategoricalBlock{N}}, x) where N
    N == length(x)
end

# function checkblock(block::RawCategoricalBlock{1, T, M}, ::Flux.OneHotVector{S, O}) where {T, M, S, O}
#     (length(block.categorydict[block.columns[1]]) == O)
# end

struct TabularTransform <: Encoding
	tfms
end

# function encode(tt::TabularTransform, _, block::Tuple{TableRow, RegressionBlock}, row)
#     tfmrow = DataAugmentation.apply(tt.tfms, DataAugmentation.TabularItem(row, block[1].columns)).data
#     x = (
#              [Int32(tfmrow[col]) for col in block[1].catcols], 
#              [tfmrow[col] for col in block[1].contcols]
#     )
#     y = [tfmrow[col] for col in block[2].columns]
#     (x, y)
# end

# function encode(tt::TabularTransform, _, block::Tuple{TableRow, ClassificationBlock}, row)
#     tfmrow = DataAugmentation.apply(tt.tfms, DataAugmentation.TabularItem(row, block[1].columns)).data
#     x = (
#              [Int32(tfmrow[col]) for col in block[1].catcols], 
#              [tfmrow[col] for col in block[1].contcols]
#     )
#     y = Flux.onehot(tfmrow[block[2].column], block[2].classes)
#     (x, y)
# end

# function encodedblock(tt::TabularTransform, block)
#     return block
# end

function encode(tt::TabularTransform, _, block::Union{ContinuousBlock, CategoricalBlock}, row)
    tfmrow = DataAugmentation.apply(
            tt.tfms, 
            DataAugmentation.TabularItem(row, block.allcols)
        ).data
    return map(col -> tfmrow[col], block.columns)
end

function encode(tt::TabularTransform, context, blocks::Tuple, row)
    return map(b -> encode(tt, context, b, row), blocks)
end

function encodedblock(::TabularTransform, block::ContinuousBlock{N, T, M}) where {N, T, M}
    return RawContinuousBlock{N, T, M}(block.columns, block.allcols)
end

function encodedblock(::TabularTransform, block::CategoricalBlock{N, T, M}) where {N, T, M}
    return RawCategoricalBlock{N, T, M}(block.columns, block.allcols, block.categorydict)
end

function encodedblock(tt::TabularTransform, blocks::Tuple)
    return map(b -> encodedblock(tt, b), blocks)
end


# function encode()
