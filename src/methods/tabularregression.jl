"""
    TabularRegression(tfms; contcols, catcols, targetcols, columns, [catdict]) <: LearningMethod

A learning method for single or multiple target regression: given a row of data from a table, determine the value of target column(s).
For example, predict the house cost from a row containing useful information about the house. 

The row values are preprocessed using `tfms` (see [`TabularTransforms`](#)).

## Keyword arguments

- `contcols`: A vector or tuple of continuous column names as `Symbol`.
- `catcols`: A vector or tuple of categorical column names as `Symbol`.
- `targetcols`: A vector or tuple of target column names as `Symbol`.
- `columns`: A vector or tuple of all the column names as `Symbol`. Each row of data has to have all these columns.
- `catdict`: A `Dict` like object which contains mappings from columns present in `catcols` to vector like objects containing unique column values.

## Learning method reference

This learning method implements the following interfaces:

{.tight}
- Core interface
- Training interface
- Testing interface

"""

struct TabularRegression <: LearningMethod
    tfms::TabularTransforms
    columns # Should be indexable using cols present below.
    contcols
    catcols
    targetcols
    catdict
end

function TabularRegression(
        tfms::TabularTransforms; 
        contcols=[], 
        catcols=[], 
        targetcols, 
        columns, 
        catdict=nothing, 
    )
    TabularRegression(tfms, columns, contcols, catcols, targetcols, catdict)
end

# Core Interface

function DLPipelines.encode(method::TabularRegression, context, input)
    tfminp = run(method.tfms, context, input)
    x = (
            [Int32(tfminp[col]) for col in method.catcols], 
            [tfminp[col] for col in method.contcols]
    )
    y = [tfminp[col] for col in method.targetcols]
    (x, y)
end

function DLPipelines.decodeŷ(method::TabularRegression, _, ŷ)
    ŷ
end

# Training Interface

DLPipelines.methodlossfn(::TabularRegression) = Flux.Losses.mse

# function DLPipelines.methodmodel(method::TabularRegression)
#     embedszs = Models.get_emb_sz(method.catdict, method.catcols)
#     embedbackbone = Models.embeddingbackbone(embedszs)
#     contbackbone = Models.continuousbackbone(length(method.contcols))
#     return Models.TabularModel(
#             embedbackbone,
#             contbackbone,
#             [200, 100],
#             n_cat=length(method.catcols),
#             n_cont=length(method.contcols),
#             out_sz = length(method.targetcols)
#         )
# end

# Testing Interface

function DLPipelines.mocksample(method::TabularRegression)
    input = [
        col in method.catcols ? method.catdict[col][rand(1:length(method.catdict[col]))] : rand() 
        for col in method.columns
    ]
    return input
end

function DLPipelines.mockmodel(method::TabularRegression)
    _ -> rand(length(method.targetcols), 1)
end

