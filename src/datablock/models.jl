


"""
    blockmodel(inblock, outblock[, backbone = blockbackbone(inblock)])

From a `backbone` model, construct a model suitable for learning
a mapping from `inblock` to `outblock`.
"""
function blockmodel end

blockmodel(inblock, outblock) = blockmodel(inblock, outblock, blockbackbone(inblock))

"""
    blockbackbone(inblock)

Create a default backbone that takes in block `inblock`.
"""
function blockbackbone end



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
    embedszs = Models.get_emb_sz(collect(map(col->length(inblock.categorydict[col]), inblock.catcols)))
    catback = Models.tabular_embedding_backbone(embedszs)
    contback = Models.tabular_continuous_backbone(N)
    return (categorical = catback, continuous = contback)
end
