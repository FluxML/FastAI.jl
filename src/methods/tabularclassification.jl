struct TabularClassification <: LearningMethod
    tfms::TabularTransforms
    columns # Should be indexable using cols present below.
    contcols
    catcols
    targetcol
    categorydict
    targetclasses
end

function TabularClassification(
        tfms::TabularTransforms; 
        contcols=[], 
        catcols=[], 
        targetcol, 
        columns, 
        targetclasses,
        categorydict=nothing, 
    )
    TabularClassification(tfms, columns, contcols, catcols, targetcol, categorydict, targetclasses)
end

# Core Interface

function DLPipelines.encode(method::TabularClassification, context, input)
    tfminp = run(method.tfms, context, input)
    x = (
            [Int32(tfminp[col]) for col in method.catcols], 
            [tfminp[col] for col in method.contcols]
    )
    y = Flux.onehot(tfminp[method.targetcol], method.targetclasses)
    (x, y)
end

function DLPipelines.decodeŷ(method::TabularClassification, _, ŷ)
    method.targetclasses[argmax(ŷ)]
end

# Training Interface

DLPipelines.methodlossfn(::TabularClassification) = Flux.Losses.logitcrossentropy

# function DLPipelines.methodmodel(method::TabularClassification)
#     embedszs = Models.get_emb_sz(method.categorydict, method.catcols)
#     embedbackbone = Models.embeddingbackbone(embedszs)
#     contbackbone = Models.continuousbackbone(length(method.contcols))
#     return Models.TabularModel(
#             embedbackbone,
#             contbackbone,
#             [200, 100],
#             n_cat=length(method.catcols),
#             n_cont=length(method.contcols),
#             out_sz = length(method.targetclasses)
#         )
# end

# Testing Interface

function DLPipelines.mocksample(method::TabularClassification)
    input = []
    for col in method.columns
        if col in method.catcols
            val = method.categorydict[col][rand(1:length(method.categorydict[col]))]
        elseif col == method.targetcol
            val = method.targetclasses[rand(1:length(method.targetclasses))]
        else
            val = rand()
        end
        push!(input, val)
    end
    return input
end

function DLPipelines.mockmodel(method::TabularClassification)
    _ -> rand(length(method.targetclasses), 1)
end

