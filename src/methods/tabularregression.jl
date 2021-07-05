struct TabularRegression <: LearningMethod
    tfms::TabularTransforms
    columns # Should be indexable using cols present below.
    contcols
    catcols
    targetcols
    catdict
    embsz_dict
end

function TabularRegression(
        tfms::TabularTransforms; 
        contcols, 
        catcols, 
        targetcols, 
        columns, 
        catdict=nothing, 
        embsz_dict=nothing
    )
    TabularRegression(tfms, columns, contcols, catcols, targetcols, catdict, embsz_dict)
end

# Core Interface

function DLPipelines.encode(method::TabularRegression, context, input)
    tempinp = (; zip(method.columns, [data for data in input])...)
    item = DataAugmentation.TabularItem(tempinp, method.columns)
    tfminp = run(method.tfms, context, item)
    x = Union{Vector{Int32}, Vector{Float64}}[[Int32(tfminp.data[col]) for col in method.catcols], [tfminp.data[col] for col in method.contcols]]
    y = [tfminp.data[col] for col in method.targetcols]
    (x, y)
end

function DLPipelines.decodeŷ(method::TabularRegression, _, ŷ)
    ŷ
end

# Training Interface

DLPipelines.methodlossfn(::TabularRegression) = Flux.Losses.mse

# function DLPipelines.methodmodel(method::TabularRegression)
#     embedszs = Models.get_emb_sz(method.catdict, method.contcols, method.embsz_dict)
#     return Models.TabularModel(
#             [200, 100], 
#             emb_szs=embedszs,
#             n_cont=length(method.contcols),
#             out_sz = length(method.targetcols),
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



