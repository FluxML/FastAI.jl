struct EncodedTableRow{M, N} <: Block
    catcols
    contcols
    categorydict
end

function EncodedTableRow(catcols, contcols, categorydict)
    EncodedTableRow{length(catcols), length(contcols)}(catcols, contcols, categorydict)
end

function checkblock(::EncodedTableRow{M, N}, x) where {M, N}
    length(x[1]) == M && length(x[2]) == N
end

struct TabularTransform <: Encoding
	tfms
end

function encodedblock(::TabularTransform, block::TableRow)
    EncodedTableRow(block.catcols, block.contcols, block.categorydict)
end

function encode(tt::TabularTransform, _, block::TableRow, row)
    columns = Tables.columnnames(row)
    usedrow = NamedTuple(filter(
            x -> x[1] ∈ block.catcols || x[1] ∈ block.contcols, 
            collect(zip(columns, row))
        ))
    tfmrow = DataAugmentation.apply(
        tt.tfms, 
        DataAugmentation.TabularItem(usedrow, keys(usedrow))
    ).data
    catvals = map(col -> tfmrow[col], block.catcols)
    contvals = map(col -> tfmrow[col], block.contcols)
    (catvals, contvals)
end
