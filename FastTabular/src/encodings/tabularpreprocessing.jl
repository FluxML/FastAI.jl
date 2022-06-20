"""
    EncodedTableRow{M, N} <: Block

Block for processed rows having a tuple of M categorical and
N continuous value collections.
"""
struct EncodedTableRow{M, N, T} <: Block
    catcols::NTuple{M}
    contcols::NTuple{N}
    categorydict::T
end

function EncodedTableRow(catcols, contcols, categorydict)
    EncodedTableRow{length(catcols), length(contcols)}(catcols, contcols, categorydict)
end

function checkblock(::EncodedTableRow{M, N}, x::Tuple{Vector, Vector}) where {M, N}
    length(x[1]) == M && length(x[2]) == N
end


function showblock!(io, ::ShowText, block::EncodedTableRow, obs)
    print(io, "EncodedTableRow(...)")
end


# ## Encoding

"""
    TabularPreprocessing <: Encoding

Encodes a `TableRow` by applying the following preprocessing steps:
- [`DataAugmentation.NormalizeRow`](#) (for normalizing a row of data for continuous columns)
- [`DataAugmentation.FillMissing`](#) (for filling missing values)
- [`DataAugmentation.Categorify`](#) (for label encoding categorical columns,
    which can be later used for indexing into embedding matrices)
or a sequence of these transformations.
"""
struct TabularPreprocessing{T} <: Encoding
	tfms::T
end

TabularPreprocessing(td::TableDataset) = TabularPreprocessing(gettransforms(td))

function encodedblock(::TabularPreprocessing, block::TableRow)
    EncodedTableRow(block.catcols, block.contcols, block.categorydict)
end

function encode(tt::TabularPreprocessing, _, block::TableRow, row)
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


function setup(::Type{TabularPreprocessing}, block::TableRow, data::TableDataset)
    return TabularPreprocessing(gettransforms(data, block.catcols, block.contcols))
end



# ## `blockmodel`


"""
    blockmodel(inblock::TableRow{M, N}, outblock::Union{Continuous, OneHotTensor{0}}, backbone=nothing) where {M, N}

Contruct a model for tabular classification or regression. `backbone` should be a
NamedTuple of categorical, continuous, and a finalclassifier layer, with
the first two taking in batches of corresponding row value matrices.
"""

"""
    blockmodel(::EncodedTableRow, ::OneHotTensor[, backbone])

Create a model for tabular classification. `backbone` should be named tuple
`(categorical = ..., continuous = ...)`. See [`TabularModel`](#) for more info.
"""
function blockmodel(inblock::EncodedTableRow, outblock::OneHotTensor{0}, backbone)
    TabularModel(
        backbone.categorical,
        backbone.continuous,
        Dense(100, length(outblock.classes))
    )
end


"""
    blockmodel(::EncodedTableRow, ::Continuous[, backbone])

Create a model for tabular regression. `backbone` should be named tuple
`(categorical = ..., continuous = ...)`. See [`TabularModel`](#) for more info.
"""
function blockmodel(inblock::EncodedTableRow, outblock::Continuous, backbone)
    TabularModel(
        backbone.categorical,
        backbone.continuous,
        Dense(100, outblock.size)
    )
end


function blockbackbone(inblock::EncodedTableRow{M, N}) where {M, N}
    embedszs = _get_emb_sz(collect(map(col->length(inblock.categorydict[col]), inblock.catcols)))
    catback = tabular_embedding_backbone(embedszs)
    contback = tabular_continuous_backbone(N)
    return (categorical = catback, continuous = contback)
end


# ## Utilities

"""
The helper functions defined below can be used for quickly constructing a dictionary,
which will be required for creating various tabular transformations available in DataAugmentation.jl.

These functions assume that the table in the TableDataset object td has Tables.jl columnaccess interface defined.
"""
function gettransformdict(td, ::Type{DataAugmentation.NormalizeRow}, cols)
    dict = Dict()
    map(cols) do col
        vals = skipmissing(Tables.getcolumn(td.table, col))
        dict[col] = (Statistics.mean(vals), Statistics.std(vals))
    end
    dict
end

function gettransformdict(td, ::Type{DataAugmentation.FillMissing}, cols)
    dict = Dict()
    map(cols) do col
        vals = skipmissing(Tables.getcolumn(td.table, col))
        dict[col] = Statistics.median(vals)
    end
    dict
end

function gettransformdict(td, ::Type{DataAugmentation.Categorify}, cols)
    dict = Dict()
    map(cols) do col
        vals = Tables.getcolumn(td.table, col)
        dict[col] = unique(vals)
    end
    dict
end

"""
    getcoltypes(td::Datasets.TableDataset)

Returns the categorical and continuous columns present in a `TableDataset`.
"""
function getcoltypes(td::TableDataset)
    schema = Tables.schema(td.table)

    contcols = Tuple(name for (name, T) in zip(schema.names, schema.types)
        if T <: Union{<:Number, <:Union{Missing, <:Number}})

    catcols = Tuple(name for name in schema.names if !(name in contcols))
    catcols, contcols
end

"""
    gettransforms(td::Datasets.TableDataset)

Returns a composition of basic tabular transformations constructed
for the given TableDataset.
"""
function gettransforms(td::TableDataset, catcols, contcols)
    normstats = gettransformdict(td, DataAugmentation.NormalizeRow, contcols)
    fmvals = gettransformdict(td, DataAugmentation.FillMissing, contcols)
    catdict = gettransformdict(td, DataAugmentation.Categorify, catcols)
    normalize = DataAugmentation.NormalizeRow(normstats, contcols)
    categorify = DataAugmentation.Categorify(catdict, catcols)
    fm = DataAugmentation.FillMissing(fmvals, contcols)

    return fm |> normalize |> categorify
end


gettransforms(td::TableDataset) = gettransforms(td, getcoltypes(td)...)


# ## Tests

@testset "TabularPreprocessing [encoding]" begin
    cols = [:col1, :col2, :col3, :col4, :col5]
    vals = [1, 2, 3, "a", "x"]
    row = NamedTuple(zip(cols, vals))

    catcols = (:col4, :col5)
    contcols = (:col1, :col2, :col3)

    col1_mean, col1_std = 10, 100
    col2_mean, col2_std = 100, 10
    col3_mean, col3_std = 15, 1

    normdict = Dict(
        :col1 => (col1_mean, col1_std),
        :col2 => (col2_mean, col2_std),
        :col3 => (col3_mean, col3_std)
    )

    tfm = TabularPreprocessing(
        DataAugmentation.NormalizeRow(normdict, contcols)
    )

    block = TableRow(
        catcols,
        contcols,
        Dict(:col4=>["a", "b"], :col5=>["x", "y", "z"])
    )

    testencoding(tfm, block, row)
    testencoding(setup(TabularPreprocessing, block, TableDataset(DataFrame([row, row]))), block, row)
end


@testset "blockbackbone" begin
    @test_nowarn FastAI.blockbackbone(EncodedTableRow((:x,), (:y,), Dict(:x => [1, 2])))
end
