"""
    emb_sz_rule(n_cat)

Compute an embedding size corresponding to the number of classes for a 
categorical variable using the rule of thumb present in python fastai.
(see https://github.com/fastai/fastai/blob/2742fe844573d06e700f869839fb9ec5f3a9bca9/fastai/tabular/model.py#L12)
"""
emb_sz_rule(n_cat) = min(600, round(Int, 1.6 * n_cat^0.56))

"""
    get_emb_sz(cardinalities::AbstractVector, [size_overrides::AbstractVector])

Given a vector of `cardinalities` of each categorical column
(i.e. each element of `cardinalities` is the number of classes in that categorical column),
compute the output embedding size according to [`emb_sz_rule`](#).
Return a vector of tuples where each element is `(in_size, out_size)` for an embedding layer.

## Keyword arguments

- `size_overrides`: A collection of integers (or `nothing` to skip override) where the value present at any index 
    will be used to as the output embedding size for that column.
"""
get_emb_sz(cardinalities::AbstractVector{<:Integer}, size_overrides=fill(nothing, length(cardinalities))) =
    map(zip(cardinalities, size_overrides)) do (cardinality, override)
        emb_dim = isnothing(override) ? emb_sz_rule(cardinality + 1) : Int64(override)
        return (cardinality + 1, emb_dim)
    end

sigmoidrange(x, low, high) = @. Flux.sigmoid(x) * (high - low) + low

function tabular_embedding_backbone(embedding_sizes, dropout_rate=0.)
    embedslist = [Flux.Embedding(ni, nf) for (ni, nf) in embedding_sizes]
    emb_drop = iszero(dropout_rate) ? identity : Dropout(dropout_rate)
    Chain(
        x -> tuple(eachrow(x)...), 
        Parallel(vcat, embedslist), 
        emb_drop
    )
end

tabular_continuous_backbone(n_cont) = BatchNorm(n_cont)

"""
    TabularModel(catbackbone, contbackbone, [finalclassifier]; kwargs...)

Create a tabular model which operates on a tuple of categorical values 
(label or one-hot encoded) and continuous values.
The categorical backbones (`catbackbone`) and continuous backbone (`contbackbone`) operate on each element of the input tuple.
The output from these backbones is then passed through a series of linear-batch norm-dropout layers before a `finalclassifier` block.

## Keyword arguments

- `outsize`: The output size of the final classifier block. For single classification tasks, 
    this would be the number of classes, and for regression tasks, this would be the
    number of target continuous variables.
- `layersizes`: A vector of sizes for each hidden layer in the sequence of linear layers.
- `dropout_rates`: Dropout probabilities for the linear-batch norm-dropout layers.
    This could either be a single number which would be used for for all the layers,
    or a collection of numbers which are cycled through for each layer.
- `batchnorm`: Set to `false` to skip each batch norm in the linear-batch norm-dropout sequence.
- `activation`: The activation function to use in the classifier layers.
- `linear_first`: Controls if the linear layer comes before or after batch norm and dropout.
"""
function TabularModel(
        catbackbone, 
        contbackbone;
        outsize,
        layersizes=(200, 100),
        kwargs...)
    TabularModel(catbackbone, contbackbone, Dense(layersizes[end], outsize); layersizes=layersizes, kwargs...)
end

function TabularModel(
        catbackbone,
        contbackbone,
        finalclassifier;
        layersizes=(200, 100),
        dropout_rates=0.,
        batchnorm=true,
        activation=Flux.relu,
        linear_first=true)
    
    tabularbackbone = Parallel(vcat, catbackbone, contbackbone)

    classifierin = mapreduce(layer -> size(layer.weight)[1], +, catbackbone[2].layers;
                             init = contbackbone.chs)
    dropout_rates = Iterators.cycle(dropout_rates)
    classifiers = []

    first_ps, dropout_rates = Iterators.peel(dropout_rates)
    push!(classifiers, linbndrop(classifierin, first(layersizes);
                                 use_bn=batchnorm, p=first_ps, lin_first=linear_first, act=activation))

    for (isize, osize, p) in zip(layersizes[1:(end-1)], layersizes[2:end], dropout_rates)
        layer = linbndrop(isize, osize; use_bn=batchnorm, p=p, act=activation, lin_first=linear_first)
        push!(classifiers, layer)
    end
    
    Chain(
        tabularbackbone,
        classifiers...,
        finalclassifier
    )
end

"""
    TabularModel(n_cont, outsize, [layersizes; kwargs...])

Create a tabular model which operates on a tuple of categorical values 
(label or one-hot encoded) and continuous values. The default categorical backbone (`catbackbone`) is
a [`Flux.Parallel`](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Parallel) set of `Flux.Embedding` layers corresponding to each categorical variable.
The default continuous backbone (`contbackbone`) is a single [`Flux.BatchNorm`](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.BatchNorm).
The output from these backbones is concatenated then passed through a series of linear-batch norm-dropout layers before a `finalclassifier` block.

## Arguments

- `n_cont`: The number of continuous columns.
- `outsize`: The output size of the model.
- `layersizes`: A vector of sizes for each hidden layer in the sequence of linear layers.

## Keyword arguments

- `cardinalities`: A collection of sizes (number of classes) for each categorical column.
- `size_overrides`: An optional argument which corresponds to a collection containing 
    embedding sizes to override the value returned by the "rule of thumb" for a particular index 
    corresponding to `cardinalities`, or `nothing`.
"""
function TabularModel(
        n_cont::Number,
        outsize::Number,
        layersizes=(200, 100);
        cardinalities,
        size_overrides=fill(nothing, length(cardinalities)))
    embedszs = get_emb_sz(cardinalities, size_overrides)
    catback = tabular_embedding_backbone(embedszs)
    contback = tabular_continuous_backbone(n_cont)

    TabularModel(catback, contback; layersizes=layersizes, outsize=outsize)
end
