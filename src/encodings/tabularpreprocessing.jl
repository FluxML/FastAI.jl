"""
    EncodedTableRow{M, N} <: Block

Block for processed rows having a tuple of M categorical and 
N continuous value collections.
"""

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

"""
    TabularTransform <: Encoding

Encodes `TableRow`s by applying a composition of various preprocessing
steps available in `DataAugmentation.jl`. Currently, preprocessing could 
consist of
- [`DataAugmentation.NormalizeRow`](#) (for normalizing a row of data for continuous columns)
- [`DataAugmentation.FillMissing`](#) (for filling missing values)
- [`DataAugmentation.Categorify`](#) (for label encoding categorical columns, which can be later used for indexing into embedding matrices)
or a sequence of these transformations.

"""

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

"""
The helper functions defined below can be used for quickly constructing a dictionary,
which will be required for creating various tabular transformations available in DataAugmentation.jl.

These functions assume that the table in the TableDataset object td has Tables.jl columnaccess interface defined.
"""

function gettransformationdict(td, ::Type{DataAugmentation.NormalizeRow}, cols)
    dict = Dict()
    for col in cols
        vals = skipmissing(Tables.getcolumn(td.table, col))
        dict[col] = (Statistics.mean(vals), Statistics.std(vals))
    end
    dict
end

function gettransformationdict(td, ::Type{DataAugmentation.FillMissing}, cols)
    dict = Dict()
    for col in cols
        vals = skipmissing(Tables.getcolumn(td.table, col))
        dict[col] = Statistics.median(vals)
    end
    dict
end

function gettransformationdict(td, ::Type{DataAugmentation.Categorify}, cols)
    dict = Dict()
    for col in cols
        vals = Tables.getcolumn(td.table, col)
        dict[col] = unique(vals)
    end
    dict
end
        