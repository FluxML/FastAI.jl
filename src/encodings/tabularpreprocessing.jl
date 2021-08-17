"""
    EncodedTableRow{M, N} <: Block

Block for processed rows having a tuple of M categorical and 
N continuous value collections.
"""

struct EncodedTableRow{M, N} <: Block
    catcols::NTuple{M}
    contcols::NTuple{N}
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

Encodes a `TableRow` by applying the following preprocessing steps:
- [`DataAugmentation.NormalizeRow`](#) (for normalizing a row of data for continuous columns)
- [`DataAugmentation.FillMissing`](#) (for filling missing values)
- [`DataAugmentation.Categorify`](#) (for label encoding categorical columns, which can be later used for indexing into embedding matrices)
or a sequence of these transformations.

"""
struct TabularTransform{T} <: Encoding
	tfms::T
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
    catvals = collect(map(col -> tfmrow[col], block.catcols))
    contvals = collect(map(col -> tfmrow[col], block.contcols))
    (catvals, contvals)
end

"""
The helper functions defined below can be used for quickly constructing a dictionary,
which will be required for creating various tabular transformations available in DataAugmentation.jl.

These functions assume that the table in the TableDataset object td has Tables.jl columnaccess interface defined.
"""
function gettransformationdict(td, ::Type{DataAugmentation.NormalizeRow}, cols)
    dict = Dict()
    map(cols) do col
        vals = skipmissing(Tables.getcolumn(td.table, col))
        dict[col] = (Statistics.mean(vals), Statistics.std(vals))
    end
    dict
end

function gettransformationdict(td, ::Type{DataAugmentation.FillMissing}, cols)
    dict = Dict()
    map(cols) do col
        vals = skipmissing(Tables.getcolumn(td.table, col))
        dict[col] = Statistics.median(vals)
    end
    dict
end

function gettransformationdict(td, ::Type{DataAugmentation.Categorify}, cols)
    dict = Dict()
    map(cols) do col
        vals = Tables.getcolumn(td.table, col)
        dict[col] = unique(vals)
    end
    dict
end

"""
The convenience functions defined below can be used to get the categorical and continuous 
columns present in a `TableDataset`, and quickly get transformations for it which can be 
passed in `TableTransform`.
"""

function getcoltypes(td::Datasets.TableDataset)
    schema = Tables.schema(td.table)
    contcols = filter(col->schema.types[findfirst(isequal(col), schema.names)] <: Union{Number, Missing}, schema.names)
    catcols = filter(col->!(col in contcols), schema.names)
    catcols, contcols
end

function gettransforms(td::Datasets.TableDataset)
    catcols, contcols = getcoltypes(td)
    normstats = FastAI.gettransformationdict(td, DataAugmentation.NormalizeRow, contcols)
    fmvals = FastAI.gettransformationdict(td, DataAugmentation.FillMissing, contcols)
    catdict = FastAI.gettransformationdict(td, DataAugmentation.Categorify, catcols)
    
    normalize = DataAugmentation.NormalizeRow(normstats, contcols)
    categorify = DataAugmentation.Categorify(catdict, catcols)
    fm = DataAugmentation.FillMissing(fmvals, contcols)
    
    return fm |> normalize |> categorify
end
        