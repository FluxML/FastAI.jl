

function TabularClassificationSingle(
        blocks::Tuple{<:TableRow, <:Label},
        data)
    tabledata, targetdata = data
    tabledata isa TableDataset || error("`data` needs to be a tuple of a `TableDataset` and targets")

    return BlockMethod(
        blocks,
        (
            setup(TabularPreprocessing, blocks[1], tabledata),
            OneHot()
        )
    )
end

"""
    TabularClassificationSingle(blocks, data)

Learning method for single-label tabular classification. Continuous columns are
normalized and missing values are filled, categorical columns are label encoded
taking into account any missing values which might be present. The target value
is predicted from `classes`. `blocks` should be an input and target block
`(TableRow(...), Label(...))`.

    TabularClassificationSingle(classes, tabledata [; catcols, contcols])

Construct learning method with `classes` to classify into and a `TableDataset`
`tabledata`. The column names can be passed in or guessed from the data.
"""
function TabularClassificationSingle(
        classes::AbstractVector,
        tabledata::TableDataset;
        catcols = nothing,
        contcols = nothing)

    blocks = (
        setup(TableRow, tabledata; catcols = catcols, contcols = contcols),
        Label(classes)
    )
    return TabularClassificationSingle(blocks, (tabledata, nothing))
end


function TabularRegression(
        blocks::Tuple{<:TableRow, <:Continuous},
        data)
    tabledata, targetdata = data
    tabledata isa TableDataset || error("`data` needs to be a tuple of a `TableDataset` and targets")
    return BlockMethod(
        blocks,
        (setup(TabularPreprocessing, blocks[1], tabledata),),
        outputblock=blocks[2]
    )
end

"""
    TabularRegression(blocks, data)

Learning method for tabular regression. Continuous columns are
normalized and missing values are filled, categorical columns are label encoded
taking into account any missing values which might be present.
 `blocks` should be an input and target block `(TableRow(...), Continuous(...))`.

    TabularRegression(n, tabledata [; catcols, contcols])

Construct learning method with `classes` to classify into and a `TableDataset`
`tabledata`. The column names can be passed in or guessed from the data. The
regression target is a vector of `n` values.
"""
function TabularRegression(
        n::Int,
        tabledata::TableDataset;
        catcols = nothing,
        contcols = nothing)
    blocks = (
        setup(TableRow, tabledata; catcols=catcols, contcols=contcols),
        Continuous(n)
    )
    return TabularRegression(blocks, (tabledata, nothing))
end
